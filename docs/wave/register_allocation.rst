.. Copyright 2025 The IREE Authors
..
.. Licensed under the Apache License v2.0 with LLVM Exceptions.
.. See https://llvm.org/LICENSE.txt for license information.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

======================
Register Allocation IR
======================

This document describes the kernel-level intermediate representation (IR) and 
register allocation infrastructure used by the Wave ASM backend.

Overview
========

The ASM backend uses a structured IR to represent entire kernels before physical
register allocation. This enables:

1. **Global register allocation**: A single allocator manages all VGPRs and SGPRs
   across the entire kernel, enabling optimal register reuse.

2. **Liveness analysis**: SSA-based live range computation determines when 
   registers can be safely reused.

3. **Constraint-aware allocation**: The allocator respects alignment requirements,
   contiguous range needs (for MFMA accumulators), and ABI-mandated precoloring.

Architecture
============

The register allocation pipeline follows this flow::

    MLIR Operations
         │
         ▼
    ┌─────────────────┐
    │  kernel_ir.py   │  Virtual register IR (KVReg, KSReg, KInstr)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_liveness.py  │  Compute live ranges from SSA defs/uses
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_regalloc.py  │  Linear scan allocation
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_generator.py │  Substitute physical registers, emit assembly
    └─────────────────────┘

Kernel IR (kernel_ir.py)
========================

The kernel IR defines the instruction representation used for register allocation.

Virtual Register Types
----------------------

The IR uses virtual registers that are placeholders for physical registers:

- **KVReg**: Virtual vector register (maps to physical ``vN``)
- **KSReg**: Virtual scalar register (maps to physical ``sN``)
- **KPhysVReg**: Precolored physical VGPR (for ABI-mandated registers like ``v0``)
- **KPhysSReg**: Precolored physical SGPR (for ABI-mandated registers like ``s[0:1]``)

Example::

    kv0 = KVReg(0)     # Virtual VGPR, will be allocated to some vN
    ks0 = KSReg(0)     # Virtual SGPR, will be allocated to some sN
    v0 = KPhysVReg(0)  # Must be physical v0 (flat thread ID)

Register Ranges
---------------

Some instructions require contiguous register blocks. ``KRegRange`` represents
a range of consecutive registers:

- **Register pairs**: ``ds_read_b64`` requires 2 consecutive VGPRs
- **Register quads**: ``buffer_load_dwordx4`` requires 4 consecutive VGPRs
- **MFMA accumulators**: ``v_mfma_f32_16x16x16_f16`` requires 4 consecutive VGPRs

Example::

    # Allocate a pair of VGPRs with 2-alignment
    pair = KRegRange(base_reg=KVReg(0), count=2, alignment=2)
    
    # After allocation, this might become v[4:5] (base must be even)

Instruction Representation
--------------------------

Each instruction is represented as a ``KInstr`` with:

- **opcode**: The operation (``KOpcode`` enum)
- **defs**: Registers written by this instruction
- **uses**: Operands read by this instruction
- **constraints**: Allocation constraints (alignment, size, register class)
- **comment**: Optional debug comment

Example::

    # v_add_u32 kv2, kv0, kv1
    instr = KInstr(
        opcode=KOpcode.V_ADD_U32,
        defs=(KVReg(2),),
        uses=(KVReg(0), KVReg(1)),
    )

Supported Opcodes
-----------------

The ``KOpcode`` enum covers all instructions needed by the ASM backend:

**Register Moves**::

    V_MOV_B32, S_MOV_B32, S_MOV_B64

**Vector Arithmetic**::

    V_ADD_U32, V_SUB_U32, V_MUL_LO_U32, V_MUL_HI_U32
    V_AND_B32, V_OR_B32, V_XOR_B32
    V_LSHLREV_B32, V_LSHRREV_B32, V_ASHRREV_I32
    V_LSHL_ADD_U32, V_LSHL_OR_B32, V_ADD_LSHL_U32, V_OR3_B32
    V_BFE_U32

**Scalar Arithmetic**::

    S_ADD_U32, S_MUL_I32, S_LSHL_B32, S_LSHR_B32, S_AND_B32, S_OR_B32
    S_ADD_U64, S_LSHL_B64

**Memory Operations**::

    S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4
    BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX4
    GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX4
    DS_READ_B32, DS_READ_B64, DS_READ_B128
    DS_WRITE_B32, DS_WRITE_B64, DS_WRITE_B128

**MFMA Instructions**::

    V_MFMA_F32_16X16X16_F16, V_MFMA_F32_32X32X8_F16
    V_MFMA_F16_16X16X16_F16, V_MFMA_F16_32X32X8_F16

**Control Flow**::

    S_WAITCNT, S_BARRIER, S_NOP, S_ENDPGM

Kernel ABI
----------

The ``KernelABI`` class defines precolored registers mandated by the GPU ABI:

- ``v0``: Contains packed flat thread ID (tid_x, tid_y, tid_z)
- ``s[0:1]``: Kernarg pointer (64-bit address)
- ``s2, s3, s4``: Workgroup IDs (optional, depends on kernel)

These registers cannot be allocated to other values.

Kernel Builder
--------------

The ``KernelBuilder`` class provides helper methods for emitting common
instruction patterns::

    builder = KernelBuilder()
    
    # Emit v_mov_b32 and get destination register
    dst = builder.v_mov_b32(src=KImm(42))
    
    # Emit v_add_u32
    sum_reg = builder.v_add_u32(src1, src2, comment="Add operands")
    
    # Emit ds_read_b64 and get destination pair
    data = builder.ds_read_b64(addr_reg, offset=128)

Liveness Analysis (kernel_liveness.py)
======================================

Since MLIR input is SSA (each value defined exactly once), liveness analysis
is straightforward.

Live Range
----------

A ``LiveRange`` represents when a virtual register is "alive":

- **start**: Instruction index where the register is defined
- **end**: Instruction index of the last use
- **size**: Number of consecutive registers (for ranges)
- **alignment**: Required alignment (for ranges)

Example::

    # kv0 defined at instruction 5, last used at instruction 12
    LiveRange(reg=KVReg(0), start=5, end=12, size=1, alignment=1)

Computing Liveness
------------------

The ``compute_liveness()`` function:

1. Scans instructions to find definition points for each virtual register
2. Scans instructions to find use points for each virtual register
3. Creates live ranges from ``(def_point, last_use_point)``
4. Computes register pressure statistics

Example::

    program = KernelProgram()
    # ... emit instructions ...
    
    liveness = compute_liveness(program)
    
    print(f"Peak VGPR pressure: {liveness.max_vreg_pressure}")
    print(f"Peak SGPR pressure: {liveness.max_sreg_pressure}")
    
    # Get registers live at instruction 10
    live_at_10 = liveness.get_live_at(10, RegClass.VGPR)

Register Pressure
-----------------

Register pressure is computed using an event-based sweep algorithm:

1. Create events for each range start (+size) and end (-size)
2. Sweep through events in order
3. Track maximum cumulative pressure

This gives the minimum number of physical registers needed at any program point.

SSA Validation
--------------

The ``validate_ssa()`` function verifies:

1. Each virtual register is defined exactly once
2. Each use is dominated by its definition (def comes before use)

Register Allocation (kernel_regalloc.py)
========================================

The allocator uses the classic linear scan algorithm optimized for SSA programs.

Linear Scan Algorithm
---------------------

The algorithm processes live ranges in order of start point::

    1. Sort live ranges by start point
    2. For each range:
       a. Expire any ranges that have ended (free their registers)
       b. Allocate physical register(s) from free pool
       c. Add to active set
    3. Return virtual-to-physical mapping

Register Pool
-------------

The ``RegPool`` class manages available physical registers:

- **Single allocation**: Allocate one register from the free list
- **Range allocation**: Find contiguous block with required alignment
- **Free**: Return register(s) to the free list

Example::

    pool = RegPool(RegClass.VGPR, max_regs=256, reserved={0})  # v0 reserved
    
    # Allocate single VGPR
    r1 = pool.alloc_single()  # Returns e.g. 1
    
    # Allocate aligned quad
    base = pool.alloc_range(size=4, alignment=4)  # Returns e.g. 4
    
    # Free
    pool.free_single(r1)
    pool.free_range(base, 4)

Constraints
-----------

The allocator respects several types of constraints:

1. **Register class**: VGPRs and SGPRs are allocated from separate pools
2. **Alignment**: Some instructions require aligned register bases
3. **Size**: Range allocations need contiguous blocks
4. **Precoloring**: ABI registers must use specific physical registers
5. **Reserved**: Some registers cannot be allocated (already in use)

No Spilling
-----------

The allocator does not support spilling. If allocation fails, compilation
fails with a diagnostic showing:

- The range that couldn't be allocated
- All overlapping ranges (sorted by length)
- Current pool state

This is appropriate because GPU kernels have fixed register budgets, and
spilling to memory would severely impact performance.

Usage
-----

::

    from kernel_regalloc import allocate_kernel
    
    program = KernelProgram()
    # ... build program ...
    
    mapping, stats = allocate_kernel(
        program,
        reserved_vgprs={0},      # v0 for flat tid
        reserved_sgprs={0, 1},   # s[0:1] for kernarg
    )
    
    print(f"Peak VGPRs: {stats.peak_vgprs}")
    print(f"Peak SGPRs: {stats.peak_sgprs}")

Code Generation (kernel_generator.py)
=====================================

The ``KernelGenerator`` converts the allocated program to assembly text.

Physical Mapping
----------------

The ``PhysicalMapping`` class holds the virtual-to-physical register mappings::

    mapping = PhysicalMapping(
        vreg_map={0: 5, 1: 6, 2: 7},  # kv0->v5, kv1->v6, kv2->v7
        sreg_map={0: 8, 1: 9},         # ks0->s8, ks1->s9
    )

Generation
----------

The generator substitutes physical register numbers and formats instructions::

    generator = KernelGenerator(program, mapping)
    assembly = generator.generate_to_string()
    
    # Output:
    #     v_add_u32 v7, v5, v6
    #     ds_read_b64 v[8:9], v10 offset:128
    #     ...

Special formatting rules are applied for certain instructions:

- LDS operations use space-separated offsets (``offset:N``)
- Register ranges use bracket notation (``v[4:7]``)

Debugging
=========

Environment Variables
---------------------

- ``WAVE_KERNEL_ALLOC_DEBUG=1``: Print allocation/free operations
- ``WAVE_KERNEL_EMITTER_DEBUG=1``: Print emitter operations
- ``WAVE_KERNEL_CSE_DEBUG=1``: Print CSE cache hits/misses

Example Debug Output
--------------------

With ``WAVE_KERNEL_ALLOC_DEBUG=1``::

    [ALLOC] v1 (single)
    [ALLOC] v2 (single)
    [ALLOC] v[4:7] (range, align=4)
    [FREE] v1
    [ALLOC] v1 (single)  # Reused!

Integration with KernelEmitter
==============================

The ``KernelEmitter`` (in ``kernel_emitter.py``) uses the kernel IR infrastructure
indirectly by emitting physical instructions directly with streaming allocation.
For complex kernels that need full kernel-level allocation, the pipeline would:

1. Build a ``KernelProgram`` using ``KernelBuilder``
2. Run ``compute_liveness()`` to get live ranges
3. Run ``allocate_kernel()`` to get physical mapping
4. Run ``render_program()`` to emit final assembly

The current implementation uses a simpler streaming approach where physical
registers are allocated immediately during emission, which works well for
expression-level code.

Future Work
===========

Potential enhancements to the register allocation infrastructure:

1. **Coalescing**: Eliminate redundant moves when source and destination
   can share the same physical register

2. **Live range splitting**: Split long-lived ranges to reduce register
   pressure at high-pressure points

3. **Spilling support**: Add memory spill/reload for extreme register
   pressure cases (with appropriate performance warnings)

4. **AGPR support**: Add accumulator GPR (AGPR) class for MFMA operations
   on architectures that support it

5. **Integration with kernel pipeline**: Use full kernel-level allocation
   for entire kernels instead of expression-level streaming allocation

