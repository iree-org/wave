How is shared memory barriers inserted in wave
=============================================================

We want to automatically insert shared-memory barriers so that read/write/atomics on the same LDS (shared memory) region are ordered correctly.
To be more specific, cases like RAW (read after write), WAR (write after read), or touch memory via atomics, we must synchronize to avoid races and hangs. On gfx94, gfx95, we support basic shared memory barrier via `amdgpu.lds_barrier`; on gfx120x we prefer split barriers (Signal / Wait), split barriers are supported via lowering to `rocdl.s.wait_dscnt`, `rocdl.s.barrier.signal`, and `rocdl.s.barrier.wait`.

Terminologies
--------------------
- Access types
    * Read (READ)
    * Write (WRITE)
    * Atomic (READ_WRITE)
    * GatherToLDS (Write but async)

- Nested region op types
    * Iterate
    * Conditional

**Note.** We will use producer to refer to an access type operator that takes ownership of a shared memory region for a period of time; consumer to refer to an access type operator that operates on the same shared memory region and therefore needs to synchronize utils the producer releases it.

When do we need a barrier?
--------------------
- Basic shared memory barrier
    * If access types of producer and consumer are different. -> insert a barrier in front of the consumer node.

- GFX12 split barrier
    * If access types of producer and consumer are different. -> insert a signal after the producer and a wait before the consumer

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

The above gif is an visaul illustration for inserting shared memory barriers between producers and consumers.

- Split barrier

.. image:: split_barrier_vis.gif
    :width: 400
    :alt: Split barrier GIF
    :align: center

The above gif is an visaul illustration for inserting split barriers between producers and consumers.

Heuristic: add_shared_memory_barriers
--------------------
The heuristic walks the graph in pre-order and proceeds as follows:

0. Walks the graph in pre-order, node by node.

1. Is this a shared_memory_op?
    * Yes: get a "memory key" (fx node object) representing the shared memory, this keeps track of the last op taking ownership of this memory region. - jump to step 2.
    * No: thank you, next. - jump to 0.

2. Do we need a barrier relative to the last op on this memory?
    * Yes: if a barrier already exists in between producer and consumer.
        * Yes: If a producer is an async op (GatherToLDS) -> we upgrade the barrier (setting ```wait_async_ops=True```).
        * No: Does this target support split barriers?
            * Yes:
                * Producer and consumer in a same graph: insert Signal after producer and wait before consumer.
                * Producer and consumer not in a same graph: defer split barrier insertion to the `add_signal_wait_to_subgraph` pass.
            * No: insert a single SharedMemoryBarrier before the consumer. Set `wait_async_ops` if needed.
    * No: noop
- end of step 2, jump to setp 3.

3. Update state
    * update the last op that is taking ownership of the memory region.
    * if we just saw a `GatherToLDS` op, set `state.is_async` to True, otherwise, after inserting a barrier, set it back to False.
- end of step 3, jump to step 4.

4. Is this op if of type NestedRegionOp (Iterate / Conditional)?
    * Yes:
        * Record a set of nodes that are currently taking ownership. This is used to compare if producers are updated in the subgraph.
        * Recurse into its subgraph. - jump to step 0, recurse on the subgraph.
        * After recursive call returns, there are some cases to consider: (ref. `should_insert_split_barrier_for_nested_region_op`)
            * case 1: split barrier is not supported - jump to step 1
            * case 2: producers are not updated in the subgraph - jump to step 1
            * case 3: `next-iteration check` mode is set (by the Iterate node) - jump to step 1
            * otherwise: calls `add_signal_wait_to_subgraph` pass for inserting signal at subgraph prolog and wait at subgraph epilog for synchronization.
    * No: noop
- end of step 4, jump to step 0.

- end of setp 0, jump to step 6.

6. Is this graph a reductin graph? (ref. `is_reduction_subgraph`)
    * Yes:
        * If we are not already checking the next iteration (i.e. `next-iteration check` mode is unset) -> run the pass again with `checking_next_iter` flag set. (This makes is_shared_memory_op look one level deeper so we catch hazards like **iter i+1 reads what iter i writes** and insert the necessary barriers.)
    * No: noop
- end of step 6, the end of `add_shared_memory_barriers` call.

Corner Cases for split barriers:
--------------------
Adding shared memory barriers when producer appear before consumer is straightforward. Things get tricky when nested region ops are involved and dependencies exist between root graph and subgraphs.
A table below shows how split barriers are inserted for those cases.

.. list-table::
    :header-rows: 1
    
    * - NestedRegionOp
      - Signal
      - Wait
      - When is barrier for subgraph inserted?
    * - ``Iterate``
      - subgraph prolog
      - subgraph epilog
      - when finish the second pass (exit check-next-iter mode) |
    * - ``Conditional``
      - subgraph prolog / epilog
      - subgraph epilog / epilog
      - when producers or consumers are in the graph 
