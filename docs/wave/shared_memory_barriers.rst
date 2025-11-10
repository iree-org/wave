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

The above gif is an visual illustration for inserting shared memory barriers between producers and consumers.

- Split barrier

.. image:: split_barrier_vis.gif
    :width: 400
    :alt: Split barrier GIF
    :align: center

The above gif is an visual illustration for inserting split barriers between producers and consumers.

Public API:
--------------------
.. code-block:: python

   add_shared_memory_barriers(trace: CapturedTrace, target: str = "") -> None

- Input: a captured FX trace and an optional target string.
- Effect: analyzes the trace, generates synchronization requirements, and emits barriers appropriate for the target.


Implementation:
--------------------
### Target
  TargetConfig describes what the backend can emit

### Analysis
  We first compute a topological enumeration (_topo_location) across nodes, then scan for hazards over shared memory resources:
    1. Identify shared-memory accesses (Read, Write, Atomic, GatherToLDS) and classify them as READ, WRTIE, or READ_WRITE.
    2. Track producer/consumer "episodes" per memory resource.
    3. Create SyncRequirement records capturing:
        - the producer region and the consumer region
        - their topological positions
        - whether the hazard crosses an iteration boundary (is_loop)
        - boundary nodes of the surrounding graph
    4. RAW hazards are tagged with BarrierType.FILL, WAR with BarrierType.READY (for easier debugging and potential future guard usage)
            
   Cross-iteration hazards: The analysis duplicates node sequences and adjusts `depth` to propagate loop-carried variables, so it can spot dependencies like ```read(i) -> write(i+1)``` cleanly. The resulting SyncRequirement.is_loop distinguishes these from intra-iteration hazards.

### Placement strategies
We'll use `window` to refer to a SyncRequirement hazard interval, the interval is defined by the topological positions between a producer and a consumer.

2 strategies are currently available; both take a list of SyncRequirement and return a reduced set of the SyncRequirement.
    - Minimize placement (minimize_placement_strategy): 
        - Intent: Use the fewest possible barriers, placed before consumers.
        - Procedure:
            1. Sort all SyncRequirements by their right endpoint. (earliest barrier requirements)
            2. Walk all windows in 1. order and check
                - If  

    - Find intersecting intervals (find_intersecting_interval_strategy): an interval-merging approach for split barriers; it tracks smallest feasible wait positions and largest feasible signal positions and emits when needed.

### Emission
A small dispatcher selects an emitter based on TargetConfig:
- LegacyEmitter (amdgpu.lds_barrier): emits monolithic SharedMemoryBarrier before the consumer. 
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
- BarrierEmitter dispatch to chose Legacy or BasicSplit emitter.
- Optimize placements, then emit barriers.
- Run verify on the resulting graph to avoid GPU hangs.

Known Limitations / TODO
--------------------
We are now propagting memory access of an op in the nested-region with `depth` set to 1
