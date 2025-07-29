.. _add_shared_memory_barriers:

add_shared_memory_barriers
==========================

.. function:: add_shared_memory_barriers(trace, graph=None, info=None, checking_next_iter=False)

   Adds shared memory barriers to a computation graph to ensure proper synchronization
   between memory access operations.

   This pass implements a heuristic-based approach to insert shared memory barriers
   between read and write operations. It walks through the graph tracking the last
   memory operation and inserts barriers when necessary to prevent race conditions.

   **Algorithm:**

   - Tracks the last read or write operation for each shared memory location
   - Inserts barriers between conflicting memory access patterns (read-write, write-read)
   - Handles nested regions and loop iterations with proper dependency management
   - Manages asynchronous dependencies for operations like GatherToLDS
