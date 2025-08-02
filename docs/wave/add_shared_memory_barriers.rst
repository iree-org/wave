Add Shared Memory Barriers
==========================

This document explains Wave's shared memory barriers insertion pass, which adds shared memory barriers to a computation graph to ensure proper synchronization between memory access operations.

Overview
--------

Barrier insertion is done in the **add_shared_memory_barriers** pass.

This pass implements a simple approach to insert shared memory barriers
between read and write operations. It walks through the graph, tracking the last
memory operation and inserting barriers when necessary to prevent race conditions.

Algorithm
---------

- Tracks the last read or write operation for each shared memory location
- Inserts barriers between conflicting memory access patterns (read-write, write-read)
- Handles nested regions and loop iterations with proper dependency management
- Manages asynchronous dependencies for operations like **GatherToLDS**


Async Dependencies
------------------

The pass also manages asynchronous dependencies for operations like **GatherToLDS**.
Async dependencies are handled by tracking all the async operations and adding them to the next barrier op as arguments.
This also requires inserting proper placeholders/get_results/iter_args ops to pass the async dependencies across the control flow graph.

There is also a special **NullAsyncDep** op, which is used to signify a no-op dependency when control flow is merged from multiple input nodes.
The usual case is when we have a loop inside the graph and we need to pass the async dependencies across the loop iterations.
In this case, you need to supply two sources for the async dependencies:

- Previous loop iteration
- Loop **init_args**

One of those dependencies can be a **NullAsyncDep** op, which means we don't have any async dependencies from this control flow path.

.. mermaid::
   :caption: Loop with async dependencies

   flowchart TD
    subgraph Loop
        IterArg-->Barrier
        Barrier~~~GatherToLDS
        GatherToLDS-->Output
        Output-->IterArg
    end
    init_args-->IterArg
    NullAsyncDep-->init_args


Async Dependencies Lowering
---------------------------

Async dependencies are implemented in AMDGPU hardware as special memory operation counters and **waitcnt** instruction.

Each memory operation increments the corresponding counter when issued and decrements it when the operation is completed.
**waitcnt** instruction wait until we have no more than N pending memory operations.
i.e., **waitcnt vmcnt(0)** waits until all global memory operations are completed.
Operations are completed in the order they are issued, which allows us to use **waitcnt** to wait for the specific operation to complete if we know how many other operations were in between.

We are using the **populate_barriers_counters** pass to calculate the required counter values for the barriers.

The algorithm is as follows:

1. Start with the barrier op with non-empty **async_deps** list.
2. Walk backwards through the control flow graph, visiting each operation until the end condition is met.
3. Increment the corresponding counter for each read/write operation which is not our target async op.
4. End condition is **target async op is reached** or **null async dep is reached** or **maximum value of counter is reached**.
5. The walk is split each time we encounter a control flow merge (e.g., loop iteration), which makes this algorithm exponential, so we choose maximum async values small enough for it to terminate early.
6. When a barrier has multiple async deps, we take the minimum value of the counters for each async dep as the conservative value.
