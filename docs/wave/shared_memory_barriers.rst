How are shared memory barriers inserted in wave
=============================================================

We want to automatically insert shared-memory barriers so that read/write/atomics on the same LDS (shared memory) region are ordered correctly.
To be more specific, cases like RAW (read after write), WAR (write after read), or touch memory via atomics, we must synchronize to avoid races and hangs. On gfx94, gfx95, we support basic shared memory barriers via `amdgpu.lds_barrier`; on gfx120x we prefer split barriers (Signal / Wait), split barriers are supported via lowering to `rocdl.s.wait_dscnt`, `rocdl.s.barrier.signal`, and `rocdl.s.barrier.wait`.

Glossary
--------------------
- Memory Access Types
    * Read (READ)
    * Write (WRITE)
    * Atomic (READ_WRITE)
    * GatherToLDS (READ_WRITE)

- Nested region op types
    * Iterate
    * Conditional

- Split barriers: two-part synchronization using signal (after producer) and wait (before consumer), supported on gfx12 targets.

- RAW / WAR hazards: classic shared memory hazards (Write→Read and Read→Write), this is represented as a closed interval in the pass (SyncRequirement).

**Note.** We will use producer to refer to an access type operator that takes ownership of a shared memory region for a period of time; consumer to refer to an access type operator that operates on the same shared memory region and therefore needs to synchronize before the producer releases it.

When do we need a barrier?
--------------------
- Basic shared memory barrier
    * If access types of producer and consumer are different. -> insert a barrier in front of the consumer node.

- GFX12 split barrier
    * If access types of producer and consumer are different. -> insert a signal after the producer and a wait before the consumer.

- READ_WRITE is involved {acts like both a producer and a consumer}
    * If it has a producer then this node will be treated like a consumer.
    * If it has a consumer then this node will be treated like a producer.
    * Either case, barriers are expected to be inserted with the logic described above.

Visualization: add_shared_memory_barriers
--------------------
- Basic barrier

.. image:: basic_barrier_vis.gif
    :width: 400
    :alt: Basic barrier GIF
    :align: center

The above gif is a visual illustration for inserting shared memory barriers between producers and consumers.

- Split barrier

.. image:: split_barrier_vis.gif
    :width: 400
    :alt: Split barrier GIF
    :align: center

The above gif is a visual illustration for inserting split barriers between producers and consumers.

Key Ideas:
--------------------
- A hazard is a producer–consumer relationship on shared memory that needs ordering.
- We model hazards as intervals (windows).
- minimize_placement_strategy is aimed at monolithic barriers (e.g., amdgpu.lds_barrier) and greedily places the fewest barriers before consumers by sorting on right endpoints.
- find_intersecting_interval_strategy is designed for split barriers: it coalesces hazard windows into the smallest feasible intersection and emits signal/wait pairs only when necessary.

Public API:
--------------------
.. code-block:: python

   add_shared_memory_barriers(trace: CapturedTrace, target: str = "") -> None

- Input: a captured FX trace and an optional target string.
- Effect: analyzes the trace, generates synchronization requirements, and emits barriers appropriate for the target.


Implementation:
--------------------
Target
--------------------
  TargetConfig describes what barriers to emit (monothlitic or split).

Analysis
--------------------
Core idea
- Track, per shared resource, a rolling “window” of producers (writers) followed by consumers (readers).
- When a new producer starts, the previous window (if it had both sides) is finalized into a barrier requirement.
add_sync_requirements
- Takes the last producer and the first consumer in the current window.
- Skips if their access types don’t actually require a barrier.
- Marks as loop carried if producer’s topo index is greater than consumer’s.
- Emits a SyncRequirement with resource, endpoints, positions, loop bounds; dedups before appending.
add_hazard_if_window_valid
- Finalizes the current window only if it contains at least one producer and one consumer.
- Calls add_sync_requirements, then resets the window.
handle_hazard
- For a single node: determine access kind and the shared resource (with depth for next iter).
- Attach optional graph_info (loop body [start, end]) to the window for later annotation.
- If node is a producer kind: flush current window, then record this producer.
- If node is a consumer kind and producers exist: record this consumer.
get_hazard_handle
- Returns a partially applied handle_hazard with chosen producer/consumer kinds (e.g., RAW or WAR).
get_barriers_analysis
- Assigns topo indices via pre order walk.
- Recursively walks the root graph and nested regions:
    - For Iterate: walk body twice (depth 0 for same iter, depth 1 for next iter) carrying [start, end].
    - For Conditional: walk its subgraph once.
- Runs two passes: WAR, then RAW; each pass maintains its own per resource windows and flushes after traversal.


Placement strategies
--------------------
We'll use `window` to refer to a SyncRequirement hazard interval, the interval is defined by the topological positions between a producer and a consumer.

2 strategies are currently available; both take a list of SyncRequirement and return a reduced set of barrier placement positions.
    - Minimize placement (minimize_placement_strategy):
        - Intent: Use the fewest possible barriers, placed before consumers.
        - Procedure:
            1. Sort all SyncRequirements by their endpoints. (earliest barrier requirements)
            2. Forward Hazard Sweep
                - For normal intervals (start < end), we maintain a single "last chosen position", initialize as -1 (an impossible topology position)
                - For each interval, if no existing barrier lies in (start, end], this mean the hazard window is not covered, a barrier is needed in that position, add one to list, and update `last_pos`.
            3. Cross-iteration (loop) hazards
                - For loop-carrierd intervals (start > end), each forms a circular interval on [graph_start, graph_end].
                - Check if an existing barrier lies in segments: (start, end] or (graph_start, end]
                - If neither segment contains a barrier, this position requires a barrier, add one to list

    - Find intersecting intervals (find_intersecting_interval_strategy):
        - Intent: Given a set of synchronization requirements (“hazard windows”) between a producer and a consumer operating on the same LDS (shared) memory region, we want to place one signal (after the producer) and one wait (before the consumer) so that: 1) Every consumer that could race with a producer waits for a matching signal, 2) No wait can appear before its corresponding signal, and 3) We use as few split barrier pairs as possible without over serializing the schedule.
        - Core: If two windows overlap, a single split barrier pair can satisfy both: signal after the latest producer seen so far (the largest L); wait before the earliest consumer (the smallest R).
            - Each window (hazard) corresponds to an interval (p, c) where p = producer topology position and c is the consumer topology position.
            - If p < c, we identify as normal hazard.
            - If p > c, we identify a loop-carrier hazard, normalize it by shifting c:= c + |body|, prepare it for the forward sweep.
        - Sweeping Procedure:
            - Sort by (c, p) and sweep once while maintaining a window.
                - max_p: the maximum producer position found so far.
                - min_c: the minimum consumer position found so far.
                - If the next hazard's p > min_c, we identify the smallest possible intersection, add a barrier to list and start a new window.
                - Otherwise, tighten the window.
            - If we add a cross-iteration barrier to graph, and all dependencies are intra-graph, then we emit a barrier surrounding the subgraph: (iterate.prev, iterate.next)

Emission
--------------------
A small dispatcher selects an emitter based on TargetConfig:
- LegacyBarrierEmitter (amdgpu.lds_barrier): emits monolithic SharedMemoryBarrier before the consumer.
- BasicSplitBarrierEmitter (rocdl.s.barrier.signal/rocdl.s.barrier.wait {barId: -1}) emits a signal after a producer and a wait before a consumer.
    - Verification: Scanning the pre-order traversal of the full nested graph.
        - no wait appears before its corresponding signal,
        - only a single barrier ID is used (-1),
        - no orphaned signals remain.

End-to-End flow
--------------------
- Build TargetConfig from target string.
- Walk the graph in the pre-order manner and assign _topo_location.
- Run get_barriers_analysis function to get a list of SyncRequirements.
- BarrierEmitter dispatch to choose Legacy or BasicSplit emitter.
- Optimize placements, then emit barriers.
- Run verify on the resulting graph to avoid GPU hangs.

Known Limitations / TODO
--------------------
- We are now propagating memory access of an op in the nested-region with `depth` set to 1
- Handle the hazard with space complexity O(1)
