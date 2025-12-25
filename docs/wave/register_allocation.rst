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
register allocation. This is the only compilation path - there is no legacy streaming
allocation mode. This enables:

1. **Global register allocation**: A single allocator manages all VGPRs and SGPRs
   across the entire kernel, enabling optimal register reuse.

2. **CFG-based liveness analysis**: Control flow graph construction with backward
   dataflow analysis correctly handles loops and live ranges across iterations.

3. **Constraint-aware allocation**: The allocator respects alignment requirements,
   contiguous range needs (for MFMA accumulators), loop SGPR reservation, and
   ABI-mandated precoloring.

4. **Peephole optimization**: Post-allocation instruction fusion for reduced VALU count.

Architecture
============

The register allocation pipeline follows this flow::

    MLIR Operations
         │
         ▼
    ┌─────────────────────────────────┐
    │  kernel_module_compiler.py      │  Entry point, orchestrates compilation
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │  kernel_compilation_context.py  │  KernelCompilationContext
    │  kernel_expr_emitter.py         │  KernelIRExprEmitter (scoped CSE)
    └────────┬────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  kernel_ir.py   │  Virtual register IR (KVReg, KSReg, KInstr)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_liveness.py  │  CFG-based backward dataflow liveness analysis
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_regalloc.py  │  Linear scan allocation with constraints
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ kernel_passes.py    │  Peephole optimization, hazard mitigation, ticketing
    └────────┬────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ kernel_generator.py          │  Physical register substitution
    │ instruction_formatter.py     │  Instruction formatting to assembly text
    └──────────────────────────────┘

Note: ``kernel_pipeline.py`` is now a thin wrapper that re-exports the main classes
for backward compatibility. The actual implementations have been split into the
modules shown above.

Kernel Compilation Context (kernel_compilation_context.py)
==========================================================

The ``KernelCompilationContext`` is the central orchestration point for kernel IR
compilation. It manages the entire compilation flow from MLIR to assembly.

Module Structure
----------------

The kernel compilation functionality is split across several focused modules:

- ``kernel_compilation_context.py``: Core context class (``KernelCompilationContext``)
- ``kernel_expr_emitter.py``: Expression emission with scoped CSE (``KernelIRExprEmitter``)
- ``kernel_expr_floor_ops.py``: Floor operation handling
- ``kernel_loops.py``: Loop management (``_LoopSupport``)
- ``kernel_mfma.py``: MFMA instruction emission (``_MFMASupport``)
- ``kernel_passes.py``: Finalization passes (``_CompilationPasses``)
- ``kernel_pipeline.py``: Backward compatibility wrapper (re-exports main classes)
- ``kernel_pipeline_shared.py``: Shared imports and helpers

Key Components
--------------

- **KernelIRExprEmitter**: Expression emitter with scoped CSE and algebraic simplification
- **KernelProgram**: Container for all kernel IR instructions
- **Symbol bounds tracking**: Provides bounds for tid_x, tid_y, wgid_x, etc. for simplification
- **Loop management**: begin_loop/end_loop with SGPR reservation

Expression Emitter
------------------

The ``KernelIRExprEmitter`` converts SymPy expressions to kernel IR instructions with
several key optimizations:

**Scoped CSE (Common Subexpression Elimination)**::

    # Global scope for loop-invariant expressions
    with emitter.scope("loop_body"):
        # Local scope that gets cleared after the loop
        addr = emitter.get_or_emit(base + tid_x * stride)
    # Loop-invariant expressions (tid_x, wgid_x) remain cached

**Loop-Invariant Detection**:

Expressions containing only these symbols are considered loop-invariant and cached globally:

- Thread IDs: ``tid_x``, ``tid_y``, ``tid_z``
- Workgroup IDs: ``wgid_x``, ``wgid_y``, ``wgid_z``
- Integer constants

**Algebraic Simplification**:

Uses symbol bounds from kernel info to simplify expressions::

    # When tid_x < 64 (single wave):
    floor(tid_x / 64) → 0  # Eliminated entirely

    # When bit ranges don't overlap:
    (row * 256) + col → (row << 8) | col  # ADD becomes OR

**Bit Range Analysis**:

The emitter tracks the effective bit range of expressions to detect non-overlapping
additions that can be converted to OR operations::

    # If tid_x uses bits 0-5 and (row * 256) uses bits 8+:
    # The addition can use v_or_b32 instead of v_add_u32
    # Or fuse into v_lshl_or_b32 if preceded by a shift

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
    V_READFIRSTLANE_B32

**Scalar Arithmetic**::

    S_ADD_U32, S_MUL_I32, S_LSHL_B32, S_LSHR_B32, S_AND_B32, S_OR_B32
    S_ADD_U64, S_LSHL_B64
    S_CMP_LT_U32

**Memory Operations**::

    S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4
    BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX4
    BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX4
    GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX4
    DS_READ_B32, DS_READ_B64, DS_READ_B128
    DS_WRITE_B32, DS_WRITE_B64, DS_WRITE_B128

**MFMA Instructions**::

    V_MFMA_F32_16X16X16_F16, V_MFMA_F32_32X32X8_F16
    V_MFMA_F16_16X16X16_F16, V_MFMA_F16_32X32X8_F16

**Control Flow**::

    S_WAITCNT, S_BARRIER, S_NOP, S_ENDPGM
    S_CBRANCH_SCC1, S_BRANCH
    LABEL

Kernel ABI
----------

The ``KernelABI`` class defines precolored registers mandated by the GPU ABI:

- ``v0``: Contains packed flat thread ID (tid_x, tid_y, tid_z)
- ``s[0:1]``: Kernarg pointer (64-bit address)
- ``s2, s3, s4``: Workgroup IDs (optional, depends on kernel)
- ``s24+``: Reserved for loop counter SGPRs

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

The liveness analysis uses CFG-based backward dataflow analysis to correctly
handle loops and compute accurate live ranges.

Control Flow Graph
------------------

The ``CFG`` class represents the control flow graph:

- **BasicBlock**: A straight-line sequence of instructions with a single entry/exit
- **Edges**: Successors and predecessors between blocks
- **Loop detection**: Identifies back-edges that indicate loop structures

CFG Construction::

    cfg = build_cfg(program.instructions)

    # cfg.blocks: List of BasicBlock
    # Each block has: start_idx, end_idx, successors, predecessors

Backward Dataflow Analysis
--------------------------

For each basic block B, the analysis computes:

- **use[B]**: Registers used before being defined in B
- **def[B]**: Registers defined in B
- **live_in[B]**: Registers live at the entry of B
- **live_out[B]**: Registers live at the exit of B

The dataflow equations::

    live_in[B] = use[B] ∪ (live_out[B] - def[B])
    live_out[B] = ∪ live_in[S] for all successors S of B

The algorithm iterates until a fixed point is reached.

Loop Handling
-------------

For loops, the analysis correctly extends live ranges:

1. **Back-edge detection**: Identifies edges from loop latch to loop header
2. **Live range extension**: Values live at the loop header are extended through all loop iterations
3. **Loop-carried values**: Accumulators and induction variables stay live across the entire loop

Example::

    # tid_x defined before loop, used in loop body
    # The live range extends from definition through all loop iterations

    loop_header:
        # tid_x is in live_in[loop_header]
        # ... use tid_x ...
    loop_latch:
        s_branch loop_header
        # tid_x is in live_out[loop_latch]

Live Range
----------

A ``LiveRange`` represents when a virtual register is "alive":

- **start**: Instruction index where the register is defined
- **end**: Instruction index of the last use (extended for loops)
- **size**: Number of consecutive registers (for ranges)
- **alignment**: Required alignment (for ranges)

Example::

    # kv0 defined at instruction 5, last used at instruction 12
    # But if inside a loop from 8-15, extended to 15
    LiveRange(reg=KVReg(0), start=5, end=15, size=1, alignment=1)

Computing Liveness
------------------

The ``compute_liveness()`` function:

1. Builds the CFG from the instruction stream
2. Detects if loops are present using back-edge analysis
3. Computes block-local use/def sets
4. Runs iterative dataflow to compute live_in/live_out
5. Extends live ranges based on CFG liveness
6. Computes register pressure statistics

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
2. Each use is dominated by its definition (CFG-dominance aware)

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
- **Reserved registers**: s24+ reserved for loop SGPRs, v0 for flat tid

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
5. **Reserved**: Some registers cannot be allocated (v0, s0-s1, s24+)

Loop SGPR Reservation
---------------------

Loop control SGPRs are reserved at s24 and above to prevent conflicts::

    s24: loop counter
    s25: loop step
    s26: loop upper bound
    s27+: additional loop state for nested loops

This reservation happens in ``KernelCompilationContext.finalize()`` before
register allocation.

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
        reserved_vgprs={0},           # v0 for flat tid
        reserved_sgprs={0, 1} | set(range(24, 32)),  # s[0:1] for kernarg, s24+ for loops
    )

    print(f"Peak VGPRs: {stats.peak_vgprs}")
    print(f"Peak SGPRs: {stats.peak_sgprs}")

Peephole Optimization (kernel_passes.py)
========================================

After register allocation, the ``_apply_peephole_optimizations()`` pass
performs local instruction fusion.

Instruction Fusion
------------------

**Shift + Add → Fused**::

    # Before:
    v_lshlrev_b32 v5, 8, v3
    v_add_u32 v6, v5, v4

    # After:
    v_lshl_add_u32 v6, v3, 8, v4

**Shift + Or → Fused**::

    # Before:
    v_lshlrev_b32 v5, 8, v3
    v_or_b32 v6, v5, v4

    # After:
    v_lshl_or_b32 v6, v3, 8, v4

These fusions are applied when:

1. The shift result is only used by the subsequent add/or
2. The shift amount is an immediate value
3. The instructions are adjacent after hazard mitigation

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
- s_waitcnt uses combined vmcnt/lgkmcnt format

Pseudo-Instruction Expansion
----------------------------

Some pseudo-instructions are expanded during generation:

- ``_g2s_srd_copy``: Expands to 4 s_mov_b32 + s_and_b32 + s_or_b32 for G2S SRD setup
- ``_init_acc_quad``: Expands to 4 v_mov_b32 for MFMA accumulator initialization
- ``LABEL``: Emits assembly label (e.g., ``loop_0_header:``)

Hazard Mitigation
=================

The ``_apply_hazard_mitigation()`` pass inserts s_nop instructions to prevent
hardware hazards.

gfx94x/gfx95x VALU Hazard
-------------------------

On these architectures, ``v_readfirstlane_b32`` reading a VGPR written by the
immediately preceding VALU instruction requires a 1-cycle wait.

The hazard detection is precise:

1. Checks if the previous instruction is a VALU (starts with ``v_``)
2. Checks if the current instruction is ``v_readfirstlane_b32``
3. Checks if the VGPR written by the VALU is read by readfirstlane
4. Only inserts ``s_nop 0`` if all conditions are met

This reduces s_nop count from ~46 (blanket insertion) to ~2 (precise insertion).

Debugging
=========

Environment Variables
---------------------

- ``WAVE_KERNEL_ALLOC_DEBUG=1``: Print allocation/free operations
- ``WAVE_KERNEL_EMITTER_DEBUG=1``: Print emitter operations
- ``WAVE_KERNEL_CSE_DEBUG=1``: Print CSE cache hits/misses
- ``WAVE_DEBUG_LIVENESS=1``: Print liveness analysis results
- ``WAVE_DEBUG_REGALLOC=1``: Print register allocation decisions
- ``WAVE_LDS_DSREAD_OFFSET_DEBUG=1``: Print ds_read offset optimization decisions

Example Debug Output
--------------------

With ``WAVE_KERNEL_ALLOC_DEBUG=1``::

    [ALLOC] v1 (single)
    [ALLOC] v2 (single)
    [ALLOC] v[4:7] (range, align=4)
    [FREE] v1
    [ALLOC] v1 (single)  # Reused!

With ``WAVE_DEBUG_LIVENESS=1``::

    [LIVENESS] Building CFG...
    [LIVENESS] Found 5 basic blocks
    [LIVENESS] Detected loop: blocks 2-4
    [LIVENESS] Extending live range for kv3: 10 -> 45 (loop-carried)

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

5. **Three-operand fusion**: Emit ``v_or3_b32`` when three OR operands
   are available to further reduce VALU instructions
