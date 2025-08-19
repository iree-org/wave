Buffer Loads, Stores, and L1 Cache Swizzling
============================================

This optimization replaces normal vector loads and stores with **AMDGPU buffer loads and stores** using the
`amdgpu.fat_raw_buffer_cast` operation.

Motivation
----------

On AMD GPUs, **buffer loads** are often faster than flat global loads that rely on each thread having a
VGPR-held address. With a buffer load, a **base address** is used and per-thread **offsets** are provided.
These offsets are applied relative to a **buffer resource descriptor**, reducing the overhead of address
computation and allowing the hardware to optimize memory access.

This change improves memory performance by:

- Using buffer loads/stores instead of flat global loads/stores.
- Optionally enabling the **L1 cache swizzle** feature to improve cache set utilization.

L1 Cache Swizzling on MI300
---------------------------

On AMD MI300-class architectures, the L1 cache is divided into **4 cache sets**.
The **cache index** is determined by bits 7 and 8 of the memory address being accessed.

If the access stride is a power of two greater than or equal to ``2^8`` (256 bytes),
all accesses may map to the same cache set (set 0), leading to **cache set conflicts** and reduced performance.

To mitigate this, the hardware provides an **L1 cache swizzle** mechanism:

- A stride value is provided to the hardware.
- The hardware uses additional, changing bits of the address to select the cache set.
- This distributes accesses more evenly across the 4 sets.

Implementation Details
----------------------

This optimization is implemented in `_cast_buffer_and_encode_stride`:

- If the stride of the second-to-last dimension is less than or equal to ``8192`` bytes
  then a **cache swizzle stride** is passed to mlir-op: `amdgpu.fat_raw_buffer_cast`.
- Otherwise, a normal ``fat_raw_buffer_cast`` is used without swizzling.
- In both cases, bounds checking is enabled, offsets are reset, and ``valid_bytes`` is set
  to the maximum byte range addressable from the buffer.

Example
-------

Without swizzling::

    %buf = amdgpu.fat_raw_buffer_cast %ptr
           bounds_check = true
           reset_offset = true
           valid_bytes = %valid_bytes

With swizzling::

    %buf = amdgpu.fat_raw_buffer_cast %ptr
           cache_swizzle_stride = %stride
           bounds_check = true
           reset_offset = true
           valid_bytes = %valid_bytes
