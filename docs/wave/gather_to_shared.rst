.. _gather_to_shared:

Gather to Shared Memory Optimization
====================================

Overview
--------

The ``gather_to_shared`` pass enables direct memory loads from global memory to Local Data Store (LDS) without passing through registers, reducing data movement overhead.

This instruction is supported only on specific AMD GPU architectures (gfx94* and gfx95*).

Architecture Support
--------------------

- **gfx94**: Support 32-bit load/store widths
- **gfx95**: Support 32-bit, 96-bit, and 128-bit load/store widths

Both architectures also support 8 and 16 bit load widths, but they are zero/sign extended to 32 bit before store, which is not very useful for us.

Instruction Semantics
---------------------

``gather_to_shared`` is translated to ``amdgpu.gather_to_lds`` MLIR op, which is lowered to ``global_load_lds_*`` instructions.

Each thread reads 4, 12, or 16 bytes from arbitrary positions in global memory or buffer and writes them contiguously to LDS starting from the address specified in the first thread in wave.
Destination addresses in all other threads are ignored.

The operation is asynchronous and AMDGPU backend currently doesn't enforce any dependencies with other LDS access operations (which may be fixed in the future). Users need to manually insert ``waitcnt`` instruction to avoid data races.

This is handled in ``add_shared_memory_barriers`` pass. Currently, it will insert ``waitcnt`` instruction right before the ``amdgpu.lds_barrier`` instruction if it had any preceding ``amdgpu.gather_to_lds`` instructions.


Pass Description
----------------

``gather_to_shared`` pass works similarly to ``minimize_global_loads``, it takes number of elements that need to be transferred and then divides it by the total number of threads to determine number of elements to be transferred per thread.

Unlike ``minimize_global_loads`` it supports a very limited number of elements per thread and only supports a simple contiguous memory layout.

Also, as LDS writes are always contiguous, it doesn't support padding if the number of elements per wave crosses a row boundary and will undo any LDS padding present in this case.
