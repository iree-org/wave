ASM Backend
===========

The ASM backend is Wave's direct AMDGCN assembly code generation backend that compiles Wave kernels to native AMD GPU assembly instructions. This backend provides fine-grained control over GPU execution and enables advanced optimizations not possible with higher-level backends.

Overview
--------

The ASM backend transforms Wave kernels through the following pipeline:

1. **MLIR Generation**: Wave kernels are first compiled to MLIR (Multi-Level Intermediate Representation)
2. **MLIR Analysis**: The MLIR is analyzed to extract kernel information, memory access patterns, and thread organization
3. **Kernel IR Generation**: Virtual register instructions are emitted to the Kernel IR program
4. **Liveness Analysis**: CFG-based backward dataflow analysis computes register live ranges
5. **Register Allocation**: Linear scan allocator assigns physical registers with constraint handling
6. **Peephole Optimization**: Instruction fusion and other local optimizations are applied
7. **Hazard Mitigation**: Architecture-specific hazard detection and s_nop insertion
8. **Assembly Generation**: AMDGCN assembly is rendered from the allocated program
9. **Binary Compilation**: The assembly is compiled to HSACO (Heterogeneous System Architecture Code Object) binaries using AMD's toolchain

Architecture
------------

The ASM backend follows a modular architecture with clear separation of concerns between MLIR analysis, kernel IR generation, register allocation, and assembly rendering.

Key Components
~~~~~~~~~~~~~~

**MLIR Walker** (``mlir_walker.py``)
  Analyzes MLIR operations and extracts kernel information including:

  - Function arguments and their types
  - Memory access patterns (loads/stores)
  - Thread ID operations and bounds
  - Affine expressions and their simplifications
  - Binding subspan operations for memory mapping
  - Loop iteration arguments and result mappings

**Operation Handlers** (``handlers*.py``)
  Operation-specific handlers for MLIR operations, split into focused modules:

  - ``handlers.py``: Main handler coordinator and compatibility layer
  - ``handlers_memory.py``: Memory operations (loads, stores, SRD setup)
  - ``handlers_control.py``: Control flow (scf.for, gpu.barrier)
  - ``handlers_arith_affine.py``: Arithmetic and affine operations
  - ``handlers_shared.py``: Shared imports and helper functions

  Supported operations include:

  - Memory allocation (memref.alloc) including LDS staging
  - Memory views (memref.view) with offset tracking
  - Load/store operations from global and LDS memory
  - MFMA operations (amdgpu.mfma) with accumulator chaining
  - LDS read/write operations (ds_read_b64, ds_write_b64) with offset optimization
  - Loop operations (scf.for) with induction variables and accumulators

**MLIR Analysis** (``mlir_analysis.py``)
  Centralized MLIR parsing, walking, and kernel metadata extraction:

  - ``walk_ops_recursively()``: Recursive MLIR operation walker
  - ``detect_needed_workgroup_ids()``: Detects which workgroup IDs are used
  - ``extract_translation_info()``: Extracts wg_size and subgroup_size from MLIR attributes
  - ``should_skip_function()``: Explicit kernel selection policy with documented constants

**Kernel Module Compiler** (``kernel_module_compiler.py``)
  Canonical entry point for MLIR to AMDGCN assembly compilation:

  - ``KernelModuleCompiler``: Main compiler class
  - ``compile_mlir_string()``: Compiles MLIR string to assembly
  - Orchestrates the full compilation pipeline

**Kernel Compilation Context** (``kernel_compilation_context.py``)
  Central context for kernel IR compilation (extracted from kernel_pipeline.py):

  - ``KernelCompilationContext``: Main context managing virtual registers, SRDs, and emission
  - Symbol bounds tracking for expression simplification
  - Scoped CSE with loop-invariant caching
  - Loop management and finalization

**Expression Emitter** (``kernel_expr_emitter.py``)
  Expression emission with scoped CSE:

  - ``KernelIRExprEmitter``: Expression emitter with CSE and algebraic simplification
  - ``kernel_expr_floor_ops.py``: Floor operation handling

**Compilation Passes** (``kernel_passes.py``)
  Post-allocation optimization and correctness passes:

  - Hazard mitigation (s_nop insertion for VALU hazards)
  - Peephole optimizations (instruction fusion)
  - Ticketing-based waitcnt insertion

**Kernel IR** (``kernel_ir.py``)
  Instruction representation and virtual register types:

  - ``KVReg``/``KSReg``: Virtual VGPR/SGPR register types
  - ``KRegRange``: Contiguous register ranges with alignment
  - ``KInstr``: Instruction with opcode, defs, uses, and metadata
  - ``KImm``/``KMemOffset``/``KSpecialReg``: Operand types

**Liveness Analysis** (``kernel_liveness.py``)
  CFG-based backward dataflow analysis:

  - ``BasicBlock``/``CFG``: Control flow graph construction
  - ``compute_liveness()``: Live range computation with loop handling
  - ``compute_cfg_liveness()``: Iterative dataflow for live_in/live_out sets
  - ``compute_live_ranges_from_cfg()``: Extends ranges for loop-carried values
  - SSA validation with dominance awareness

**Register Allocation** (``kernel_regalloc.py``)
  Constraint-aware linear scan allocator:

  - Precoloring for ABI-mandated registers (v0 for flat tid, s[0:1] for kernarg)
  - Range allocation with alignment constraints
  - Loop SGPR reservation (s24+ for counters)
  - No spilling: fails compilation with diagnostic if allocation fails

**Kernel Generator** (``kernel_generator.py``)
  Assembly generation from allocated program:

  - Physical register substitution using ``InstructionFormatter``
  - Pseudo-instruction expansion (e.g., ``_g2s_srd_copy``, ``_init_acc_quad``)
  - Label emission for control flow

**Instruction Formatter** (``instruction_formatter.py``)
  Centralized instruction formatting to assembly text:

  - Single point for all physical instruction formatting
  - Operand validation (strict mode via ``WAVE_STRICT_FORMATTER``)
  - Special handling for buffer operations, LDS, and MFMA
  - Integration with ``InstructionRegistry`` for opcode metadata

**Metadata Emitter** (``metadata_emitter.py``)
  AMDGCN metadata directive generation:

  - Kernel prologue (``.amdgcn_target``, ``.amdhsa_kernel``, etc.)
  - Kernel epilogue (``.amdgpu_metadata`` YAML block)
  - Resource patching for dynamic register counts
  - Architecture-specific granularity handling

**Ticketing** (``ticketing.py``)
  Memory operation tracking and waitcnt coalescing:

  - Tracks outstanding VMEM and LGKM operations
  - Coalesces redundant ``s_waitcnt`` instructions
  - Integrated into kernel IR finalization pass

**ABI Policies** (``abi.py``)
  Centralized ABI-related policies:

  - ``get_system_vgpr_workitem_id_policy()``: Determines VGPR workitem ID requirements
  - Workgroup size normalization

**Gather-to-Shared Handler** (``gather_to_shared.py``)
  Handles gather_to_lds operations for direct global-to-LDS transfers:

  - ``G2SHandler`` class for buffer_load_dword...lds emission
  - ``analyze_g2s_region`` for finding gather_to_lds operations
  - ``precreate_g2s_srds`` for SRD pre-allocation before loops
  - SRD tracing through memref cast chains
  - VGPR offset computation for global memory addressing
  - LDS destination address (M0) computation via kernel IR

**Expression Simplification** (``expr_simplify.py``)
  Algebraic simplification using singledispatch for type-based rule dispatch:

  - Declarative rewrite rules with SymPy Wild patterns
  - ``@singledispatch`` for ``get_max_value``, ``simplify_expr``, ``combine_like_terms``
  - Floor/mod identity: ``floor(x/n)*n + Mod(x,n) → x``
  - Linear floor to mod: ``a*x - a*n*floor(x/n) → a*Mod(x,n)``
  - Redundant floor elimination when ``max(x) < n``
  - Power-of-2 shift combining: ``x * 2^a * 2^b → x * 2^(a+b)``

**Utils** (``utils.py``)
  Provides utility functions for:

  - MLIR type parsing and analysis
  - Affine expression simplification using SymPy
  - Thread ID bound analysis
  - Expression-to-assembly conversion
  - SymPy expression building from MLIR indices
  - Byte offset calculation for memory addressing

Features
--------

Kernel IR Compilation Path
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend uses a kernel IR compilation path that provides:

- **Virtual Registers**: All operations emit to virtual registers (KVReg/KSReg)
- **Whole-Program Analysis**: Complete liveness analysis across the kernel
- **Optimal Allocation**: Linear scan with constraint handling and precoloring
- **Loop Awareness**: CFG-based analysis correctly handles loop-carried values

This is the only compilation path - there is no legacy mode.

Advanced Optimizations
~~~~~~~~~~~~~~~~~~~~~~

The backend implements several optimization techniques:

- **Common Subexpression Elimination (CSE)**: Scoped caching with global scope for loop-invariant expressions
- **Loop-Invariant Caching**: Expressions using only tid_x/tid_y/wgid_* are cached globally and persist across loop iterations
- **Algebraic Simplification**: Uses symbol bounds (from workgroup/subgroup size) to simplify expressions like ``floor(tid_x/64) → 0`` when tid_x < 64
- **Bit Range Analysis**: Detects non-overlapping bit ranges and uses OR instead of ADD (e.g., ``(tid_x * 256) + col`` becomes OR when ranges don't overlap)
- **Instruction Fusion**: Peephole optimizer fuses ``v_lshlrev_b32 + v_add_u32`` → ``v_lshl_add_u32`` and ``v_lshlrev_b32 + v_or_b32`` → ``v_lshl_or_b32``
- **ds_read/ds_write Offset Optimization**: Uses instruction offset field (up to 8192 bytes) to reduce address computation
- **Power-of-2 Optimization**: Uses bit shifts instead of multiplication for power-of-2 constants
- **Constant Caching**: Large constants are cached to avoid redundant v_mov_b32 instructions

Memory Management
~~~~~~~~~~~~~~~~~

The ASM backend handles memory operations through:

- **Shader Resource Descriptors (SRDs)**: Proper setup for buffer access with lazy allocation
- **Vectorized Loads/Stores**: Efficient 16-byte aligned memory operations
- **Address Calculation**: Optimized offset computation using SymPy expressions
- **LDS (Local Data Share) Staging**: Automatic staging through shared memory for improved performance
- **LDS Offset Optimization**: Uses ds_read/ds_write offset field (0-8192 bytes) to reduce VALU instructions
- **Synchronization**: Proper wait instructions for memory consistency (vmcnt, lgkmcnt)

Hardware Accelerated Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend provides native support for AMD GPU specialized instructions:

- **MFMA (Matrix Multiply-Accumulate)**: Hardware-accelerated matrix operations on CDNA architectures using VGPR-variant instructions with accumulator chaining for K-loops
- **LDS Operations**: Fast shared memory operations (ds_read_b64, ds_write_b64) with offset field support for reduced address computation
- **Multi-Wave Support**: Automatic detection and handling of multi-wave workgroups with proper thread ID extraction
- **Multi-Workgroup Support**: Dynamic detection of workgroup ID usage and conditional SGPR allocation
- **Loop Support (scf.for)**: Native support for structured control flow loops with SGPR induction variables and VGPR accumulators

Architecture Support
~~~~~~~~~~~~~~~~~~~~

The ASM backend supports multiple AMD GPU architectures with architecture-specific optimizations:

- **CDNA3 (gfx942)**: MI300 series with VGPR granularity of 4, SGPR granularity of 8
- **gfx950**: Support with precise VALU hazard mitigation

Gather-to-LDS Operations
~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend supports direct global-to-LDS transfers using ``buffer_load_dword...lds`` instructions,
which bypass VGPRs and load data directly into LDS (Local Data Share):

**Key Features**

- **Direct Transfer**: Data flows from global memory directly to LDS without VGPR intermediaries
- **M0 Computation via Kernel IR**: M0 register value computed inline using expression emitter
- **Cache Swizzle Support**: Configurable cache swizzle stride for optimal LDS bank access
- **SRD Management**: Automatic SRD (Shader Resource Descriptor) setup with LDS-specific word3 (0x27000)
- **SRD Pre-creation**: All G2S SRDs are pre-created before loops to prevent overwrites

**Generated Assembly Pattern**

.. code-block:: asm

   # SRD setup for gather_to_lds (word3 = 0x27000 for LDS mode)
   s_mov_b32 s32, s12                    # SRD word0 (addr low)
   s_and_b32 s33, s13, 0xffff
   s_or_b32 s33, s33, 0x40200000         # cache swizzle enabled
   s_mov_b32 s34, 2147483645             # SRD word2 (max buffer)
   s_mov_b32 s35, 159744                 # SRD word3 (0x27000 = LDS mode)

   # M0 setup and buffer_load...lds
   s_mov_b32 m0, 512                     # LDS destination offset
   buffer_load_dword v4, s[32:35], 0 offen lds

   # Barrier synchronization
   s_waitcnt vmcnt(0)                    # Wait for global reads
   s_waitcnt lgkmcnt(0)                  # Wait for LDS writes
   s_barrier                             # Sync threads

   # Now LDS contains the loaded data
   ds_read_b64 v[60:61], v53             # Read from LDS for MFMA

**Enabling Gather-to-LDS**

Use the ``use_global_to_shared=True`` compile option:

.. code-block:: python

   options = WaveCompileOptions(
       subs={...},
       backend="asm",
       use_global_to_shared=True,  # Enable gather_to_lds
   )

Hardware Hazard Mitigation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend automatically detects and mitigates architecture-specific hardware hazards with precise insertion:

**gfx94x/gfx95x VALU Hazard**

On gfx94x/gfx95x, a VALU instruction writing to a VGPR followed by ``v_readfirstlane_b32`` reading that same VGPR requires a 1-cycle wait. The hazard mitigation is precise:

- Only inserts ``s_nop 0`` when ``v_readfirstlane_b32`` immediately follows a VALU instruction
- Only when the VALU writes to a VGPR that ``v_readfirstlane`` reads
- Reduces s_nop count from ~46 (blanket insertion) to ~2 (precise insertion)

.. code-block:: asm

   # Without mitigation (causes hazard on gfx950):
   v_add_u32 v7, v5, v6
   v_readfirstlane_b32 s10, v7    # HAZARD: reading v7 too soon

   # With automatic mitigation:
   v_add_u32 v7, v5, v6
   s_nop 0                         # 1-cycle wait inserted automatically
   v_readfirstlane_b32 s10, v7     # Safe: v7 is ready

Peephole Optimizations
~~~~~~~~~~~~~~~~~~~~~~

The backend applies peephole optimizations after hazard mitigation:

**Instruction Fusion**

1. ``v_lshlrev_b32 + v_add_u32`` → ``v_lshl_add_u32`` (saves 1 VALU instruction)
2. ``v_lshlrev_b32 + v_or_b32`` → ``v_lshl_or_b32`` (saves 1 VALU instruction)

**Bit Range Analysis**

When adding two values with non-overlapping bit ranges, the addition can be converted to OR:

.. code-block:: python

   # If tid_x is in bits 0-5 and (row * 256) is in bits 8+:
   # (row * 256) + tid_x  # Uses v_add_u32
   # becomes:
   # (row << 8) | tid_x   # Can use v_or_b32 or v_lshl_or_b32

The backend analyzes bit ranges of expressions using symbol bounds to detect this pattern automatically.

CFG-Based Liveness Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend uses CFG-based backward dataflow analysis for accurate liveness:

**Features**

- **BasicBlock/CFG Construction**: Builds control flow graph from labels and branches
- **Loop Detection**: Identifies back-edges to detect loops in the CFG
- **Backward Dataflow**: Computes live_in/live_out sets for each basic block
- **Live Range Extension**: Extends ranges for loop-carried values across iterations

**Algorithm**

For each basic block B:

- ``use[B]`` = registers used before being defined in B
- ``def[B]`` = registers defined in B
- ``live_in[B]`` = use[B] ∪ (live_out[B] - def[B])
- ``live_out[B]`` = ∪ live_in[S] for all successors S of B

This correctly handles loop-invariant values like ``tid_x`` and ``wgid_x`` that are defined once but used across multiple loop iterations.

Dynamic Register Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend features fully dynamic register allocation:

- **Automatic VGPR Allocation**: Computes required VGPRs based on actual usage
- **Automatic SGPR Allocation**: Computes required SGPRs based on actual usage
- **Conditional System Register Allocation**: Dynamically detects workgroup ID and thread ID usage from MLIR
- **Granularity Alignment**: Automatically rounds allocations to architecture-specific granularities
- **VGPR-Variant MFMA**: Uses MFMA instructions that write directly to VGPRs, eliminating accumulator complexity
- **Loop SGPR Reservation**: Reserves s24+ for loop counter SGPRs to prevent conflicts

Usage
-----

Basic Usage
~~~~~~~~~~~

To use the ASM backend, specify it in your Wave compilation options:

.. code-block:: python

   import wave_lang.kernel.lang as tkl
   import wave_lang.kernel.wave as tkw
   from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

   # Define your kernel
   @tkw.wave(constraints)
   def my_kernel(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
                 b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
       res = tkw.read(a)
       tkw.write(res, b)

   # Compile with ASM backend
   options = WaveCompileOptions(
       subs={M: 16, N: 16, ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value},
       backend="asm",
       wave_runtime=True,
       compile_to_mlir=False
   )

   compiled_kernel = wave_compile(options, my_kernel)

Compilation Options
~~~~~~~~~~~~~~~~~~~

The ASM backend supports several compilation options:

- **``backend="asm"``**: Enables the ASM backend
- **``wave_runtime=True``**: Uses Wave's C++ runtime for execution
- **``compile_to_mlir=False``**: Skips MLIR output, goes directly to assembly
- **``compile_to_asm=True``**: Generates raw assembly text (for debugging)
- **``use_global_to_shared=True``**: Enables gather_to_lds (buffer_load...lds) for direct global-to-LDS transfers

Example: Simple Copy Kernel
---------------------------

Here's a complete example of a copy kernel using the ASM backend:

.. code-block:: python

   import wave_lang.kernel.lang as tkl
   import wave_lang.kernel.wave as tkw
   from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
   from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
   from torch.testing import assert_close

   # Define symbolic dimensions
   M = tkl.sym.M
   N = tkl.sym.N
   ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

   # Hardware constraints
   constraints = [
       tkw.HardwareConstraint(
           threads_per_wave=64,
           vector_shapes={M: 16, N: 16}
       ),
       tkw.WorkgroupConstraint(M, 16, 0),
       tkw.WorkgroupConstraint(N, 16, 1),
       tkw.WaveConstraint(M, 16),
       tkw.WaveConstraint(N, 16)
   ]

   # Define the kernel
   @tkw.wave(constraints)
   def copy_kernel(
       a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
       b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]
   ):
       """Copy kernel that reads from input and writes to output."""
       res = tkw.read(a)
       tkw.write(res, b)

   # Create test data
   shape = (16, 16)
   a = device_randn(shape, dtype=torch.float16)
   b = device_zeros(shape, dtype=torch.float16)

   # Compile with ASM backend
   options = WaveCompileOptions(
       subs={
           M: shape[0],
           N: shape[1],
           ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value
       },
       canonicalize=True,
       backend="asm",
       wave_runtime=True,
       compile_to_mlir=False
   )

   # Compile and execute
   compiled_kernel = wave_compile(options, copy_kernel)
   compiled_kernel(a, b)

   # Verify results
   assert_close(a, b)
   print("Copy kernel executed successfully!")

Generated Assembly
~~~~~~~~~~~~~~~~~~

The ASM backend generates optimized AMDGCN assembly. For the copy kernel above, it produces:

.. code-block:: asm

   .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
   .text
   .protected copy_kernel
   .globl copy_kernel
   .p2align 8
   .type copy_kernel,@function

   .section .rodata,#alloc
   .p2align 6
   .amdhsa_kernel copy_kernel
     .amdhsa_user_sgpr_kernarg_segment_ptr 1
     .amdhsa_accum_offset 12
     .amdhsa_next_free_vgpr 12
     .amdhsa_next_free_sgpr 16
     .amdhsa_group_segment_fixed_size 0
     .amdhsa_private_segment_fixed_size 0
     .amdhsa_system_sgpr_workgroup_id_x 1
     .amdhsa_system_sgpr_workgroup_id_y 1
     .amdhsa_system_sgpr_workgroup_id_z 1
     .amdhsa_system_vgpr_workitem_id 0
     .amdhsa_float_denorm_mode_32 3
     .amdhsa_float_denorm_mode_16_64 3
   .end_amdhsa_kernel
   .text

   # SRD upper word (gfx9xx): data_format=4 => 0x20000
   .set Srd127_96, 0x20000

   copy_kernel:
       s_load_dwordx2 s[2:3], s[0:1], 0x0
       s_load_dwordx2 s[4:5], s[0:1], 0x8
       s_waitcnt lgkmcnt(0)

       # SRD for input buffer (arg0)
       s_mov_b32 s8, s2
       s_mov_b32 s9, s3
       s_mov_b32 s10, 2048
       s_mov_b32 s11, Srd127_96

       # Compute lane ID and vector offset
       v_mbcnt_lo_u32_b32 v0, -1, 0
       v_mbcnt_hi_u32_b32 v0, -1, v0
       v_lshlrev_b32 v2, 5, v0

       # Load data from input buffer
       buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen offset:0
       s_waitcnt vmcnt(0)
       buffer_load_dwordx4 v[8:11], v2, s[8:11], 0 offen offset:16
       s_waitcnt vmcnt(0)

       # SRD for output buffer (arg1)
       s_mov_b32 s12, s4
       s_mov_b32 s13, s5
       s_mov_b32 s14, 2048
       s_mov_b32 s15, Srd127_96

       # Store data to output buffer
       buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen offset:0
       buffer_store_dwordx4 v[8:11], v2, s[12:15], 0 offen offset:16

       s_endpgm

   .amdgpu_metadata
   ---
   amdhsa.version:
     - 1
     - 2
   amdhsa.kernels:
     - .name: copy_kernel
       .symbol: 'copy_kernel.kd'
       .language: OpenCL C
       .language_version: [2, 0]
       .args:
         - .name: arg0_ptr
           .size: 8
           .offset: 0
           .value_kind: global_buffer
           .value_type: i8*
         - .name: arg1_ptr
           .size: 8
           .offset: 8
           .value_kind: global_buffer
           .value_type: i8*
       .group_segment_fixed_size: 0
       .kernarg_segment_align: 8
       .kernarg_segment_size: 16
       .max_flat_workgroup_size: 64
       .private_segment_fixed_size: 0
       .sgpr_count: 16
       .sgpr_spill_count: 0
       .vgpr_count: 12
       .vgpr_spill_count: 0
       .wavefront_size: 64
   ...
   .end_amdgpu_metadata

Key Features Demonstrated
~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates several key features of the ASM backend:

1. **Automatic SRD Setup**: The backend automatically generates Shader Resource Descriptor setup for buffer access
2. **Efficient Memory Access**: Uses 16-byte aligned buffer load/store instructions for optimal performance
3. **Thread ID Management**: Automatically computes lane IDs and vector offsets for memory access
4. **Dynamic Register Allocation**: Intelligently allocates SGPRs and VGPRs based on actual usage
5. **Synchronization**: Proper wait instructions ensure memory consistency

Example: Matrix Multiply with MFMA
-----------------------------------

The ASM backend supports hardware-accelerated matrix operations using MFMA instructions on CDNA architectures:

.. code-block:: python

   import wave_lang.kernel.lang as tkl
   import wave_lang.kernel.wave as tkw
   from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

   # Define symbolic dimensions
   M = tkl.sym.M
   N = tkl.sym.N
   K = tkl.sym.K
   BLOCK_M = tkl.sym.BLOCK_M
   BLOCK_N = tkl.sym.BLOCK_N
   LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
   STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
   ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

   # Hardware constraints for MFMA
   constraints = [
       tkw.HardwareConstraint(
           threads_per_wave=64,
           waves_per_block=(1, 1, 1),
           vector_shapes={M: BLOCK_M, N: BLOCK_N}
       ),
       tkw.WorkgroupConstraint(M, BLOCK_M, 0),
       tkw.WorkgroupConstraint(N, BLOCK_N, 1),
       tkw.WaveConstraint(M, BLOCK_M / 1),
       tkw.WaveConstraint(N, BLOCK_N / 1)
   ]

   # Define MMA kernel with LDS staging
   @tkw.wave(constraints)
   def mma_kernel(
       a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
       b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
       c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32]
   ):
       """Matrix multiply kernel using MFMA with LDS staging."""
       # Allocate LDS for staging
       a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
       b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)

       # Perform MFMA operation
       acc = tkw.mma(a_reg, b_reg)

       # Write results
       tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

   # Compile with ASM backend
   options = WaveCompileOptions(
       subs={
           M: 16, N: 16, K: 16,
           BLOCK_M: 16, BLOCK_N: 16,
           LOAD_ELEMS_PER_THREAD: 4,
           STORE_ELEMS_PER_THREAD: 4,
           ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value
       },
       canonicalize=True,
       backend="asm",
       wave_runtime=True,
       compile_to_mlir=False
   )

   compiled_kernel = wave_compile(options, mma_kernel)

MFMA Key Features
~~~~~~~~~~~~~~~~~

The MFMA support includes:

1. **LDS Staging**: Automatically stages data through Local Data Share for optimal MFMA performance
2. **VGPR-Variant MFMA**: Uses MFMA instructions that write directly to VGPRs (not accumulators)
3. **Accumulator Chaining**: Supports chained MFMAs for K-loops with persistent accumulators
4. **Loop Integration**: Automatically uses loop ``iter_args`` as MFMA accumulators
5. **Synchronization**: Inserts ``s_waitcnt lgkmcnt(0)`` before MFMA to ensure LDS reads complete
6. **Multi-Wave Support**: Correctly handles multiple waves per workgroup with thread ID extraction
7. **Multi-Workgroup Support**: Automatically detects and allocates workgroup ID system SGPRs as needed
8. **Dynamic Metadata**: Computes ``vgpr_count``, ``sgpr_count``, and ``lds_size`` dynamically

Advanced Features
-----------------

Loop Support (scf.for)
~~~~~~~~~~~~~~~~~~~~~~

The ASM backend provides native support for MLIR structured control flow loops (``scf.for``), enabling K-loop tiling for GEMM and other iterative operations:

**Loop Structure**

The backend generates efficient loop control flow with:

1. **Loop Initialization**: Allocates SGPR for loop counter, step, and upper bound (reserved at s24+)
2. **Loop Header**: Emits comparison (``s_cmp_lt_u32``) and conditional branch (``s_cbranch_scc1``)
3. **Loop Body**: Contains the computation with access to loop induction variable
4. **Loop Latch**: Increments counter (``s_add_u32``) and branches back to header
5. **Loop Exit**: Continuation point after loop completes

**MFMA Accumulator Chaining**

For K-loops with MFMA operations, the backend automatically:

- Pre-allocates VGPR quads for loop accumulators
- Initializes accumulators to 0.0 before loop entry
- Chains MFMA operations using the same accumulator across iterations
- Maps loop ``iter_args`` to their corresponding VGPRs
- Tracks loop results for use after loop exit

**Example**: GEMM K-Loop

.. code-block:: python

   @tkw.wave(constraints)
   def gemm_kernel(a, b, c):
       c_reg = tkl.Register[M, N, tkl.f32](0.0)

       @tkw.iterate(K, init_args=[c_reg])
       def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
           a_reg = tkw.read(a)
           b_reg = tkw.read(b)
           acc = tkw.mma(a_reg, b_reg, acc)  # Chained MFMA
           return acc

       tkw.write(repeat, c)

**Generated Assembly Pattern**

.. code-block:: asm

   # Initialize loop counter and accumulator
   s_mov_b32 s24, 0                  # counter = 0
   s_mov_b32 s25, 1                  # step = 1
   s_mov_b32 s26, 2                  # upper_bound = K/BLOCK_K
   v_mov_b32 v4, 0                   # accumulator[0] = 0.0
   v_mov_b32 v5, 0                   # accumulator[1] = 0.0
   v_mov_b32 v6, 0                   # accumulator[2] = 0.0
   v_mov_b32 v7, 0                   # accumulator[3] = 0.0

   loop_0_header:
       s_cmp_lt_u32 s24, s26         # counter < upper_bound?
       s_cbranch_scc1 loop_0_body
       s_branch loop_0_exit

   loop_0_body:
       # Load A and B tiles
       buffer_load_dwordx2 ...
       # ...
       # Chained MFMA using same accumulator
       v_mfma_f32_16x16x16_f16 v[4:7], v[...], v[...], v[4:7]

   loop_0_latch:
       s_add_u32 s24, s24, s25       # counter += step
       s_branch loop_0_header

   loop_0_exit:
       # Store accumulated results
       buffer_store_dword v4, ...

**Key Features**

- **SGPR Induction Variables**: Loop counters stored in reserved SGPRs (s24+) for efficiency
- **VGPR Accumulators**: Loop-carried values (e.g., MFMA results) stay in VGPRs
- **Nested Loops**: Support for multiple active loop contexts via loop_stack
- **Unique Labels**: Each loop gets unique labels (loop_N_header, loop_N_body, etc.)
- **CFG-Based Liveness**: Live ranges correctly extended across loop iterations

Multi-Wave and Multi-Workgroup Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend automatically handles complex thread and workgroup configurations:

**Multi-Wave Kernels**

When a workgroup contains multiple waves (e.g., ``workgroup_size = [256, 4, 1]``), the backend:

1. **Detects Multi-Wave Configuration**: Analyzes workgroup size from MLIR ``translation_info``
2. **Requests System VGPRs**: Sets ``.amdhsa_system_vgpr_workitem_id 1`` to get flat thread ID in ``v0``
3. **Extracts Thread IDs**: Generates code to extract ``tid_x`` and ``tid_y`` from flat ID:

   - ``tid_x = v0 & 0x3ff`` (bits 0-9)
   - ``tid_y = (v0 >> 10) & 0x3ff`` (bits 10-19)

4. **Uses in Addressing**: Thread IDs are used in affine expressions for memory access
5. **Provides Bounds**: Symbol bounds from subgroup_size used for algebraic simplification

**Multi-Workgroup Kernels**

When a kernel is dispatched across multiple workgroups (e.g., ``grid = [16, 16, 1]``), the backend:

1. **Detects Workgroup ID Usage**: Scans MLIR for ``gpu.block_id`` operations
2. **Conditionally Requests System SGPRs**: Only requests needed workgroup IDs:

   - ``.amdhsa_system_sgpr_workgroup_id_x 1`` if ``gpu.block_id x`` is used
   - ``.amdhsa_system_sgpr_workgroup_id_y 1`` if ``gpu.block_id y`` is used
   - ``.amdhsa_system_sgpr_workgroup_id_z 1`` if ``gpu.block_id z`` is used

3. **Allocates SGPRs**: Places workgroup IDs at ``s2``, ``s3``, ``s4`` (after kernarg pointer)
4. **Uses in Addressing**: Workgroup IDs scale memory access for workgroup-local tiles

**Example**: 256x256 MMA with 4 workgroups (2x2 grid), single wave per workgroup:

- Each workgroup processes a 64x64 tile
- ``wgid_x`` and ``wgid_y`` are detected and allocated
- Global memory addresses: ``base + (wgid_x * 64 * 4) + (wgid_y * 64 * 256 * 4) + tid_x``

Algebraic Simplification with Symbol Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend uses SymPy for algebraic simplification with symbol bounds derived from kernel information:

**Symbol Bounds**

- ``tid_x``, ``tid_y``, ``tid_z``: Bounded by ``subgroup_size`` (e.g., 0-63 for single wave, 0-255 for 4 waves)
- ``wgid_x``, ``wgid_y``, ``wgid_z``: Bounded by grid dimensions

**Optimizations Enabled**

.. code-block:: python

   # When tid_x < 64:
   floor(tid_x / 64) → 0  # Eliminated entirely

   # When expression ranges don't overlap:
   (row * 256) + col → (row << 8) | col  # ADD becomes OR

**Top-Level Only Simplification**

To avoid exponential complexity, simplification is only applied to top-level expressions, not recursively on sub-expressions. This reduces simplification calls by ~18x while maintaining optimization benefits.

Expression Caching and CSE
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend uses a sophisticated caching system for Common Subexpression Elimination:

**Scoped Caching**

- **Global Scope**: Loop-invariant expressions (containing only tid_x, tid_y, wgid_*) are cached globally
- **Local Scopes**: Loop-varying expressions use scoped caches that are pushed/popped
- **Structural Keys**: Expressions are keyed by structure, not Python object identity

**Loop-Invariant Detection**

An expression is loop-invariant if it contains only:
- Thread IDs: ``tid_x``, ``tid_y``, ``tid_z``
- Workgroup IDs: ``wgid_x``, ``wgid_y``, ``wgid_z``
- Integer constants

Loop-invariant expressions are computed once before the loop and reused across all iterations.

Example: GEMM with K-Loop
---------------------------

The ASM backend supports efficient GEMM kernels with K-dimension tiling using ``scf.for`` loops:

.. code-block:: python

   import wave_lang.kernel.lang as tkl
   import wave_lang.kernel.wave as tkw
   from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

   # Define symbolic dimensions
   M = tkl.sym.M
   N = tkl.sym.N
   K = tkl.sym.K
   BLOCK_M = tkl.sym.BLOCK_M
   BLOCK_N = tkl.sym.BLOCK_N
   BLOCK_K = tkl.sym.BLOCK_K
   ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
   ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

   # Hardware constraints for multi-wave GEMM
   constraints = [
       tkw.WorkgroupConstraint(M, BLOCK_M, 0),
       tkw.WorkgroupConstraint(N, BLOCK_N, 1),
       tkw.TilingConstraint(K, BLOCK_K),  # K-dimension tiling
       tkw.WaveConstraint(M, 16),  # Each wave handles 16x16
       tkw.WaveConstraint(N, 16),
       tkw.HardwareConstraint(
           threads_per_wave=64,
           mma_type=tkw.MMAType.F32_16x16x16_F16,
       ),
   ]

   @tkw.wave(constraints)
   def gemm_kernel(
       a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
       b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
       c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
   ):
       """GEMM kernel: C = A @ B^T with K-loop."""
       c_reg = tkl.Register[M, N, tkl.f32](0.0)

       @tkw.iterate(K, init_args=[c_reg])
       def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
           a_reg = tkw.read(a)
           b_reg = tkw.read(b)
           acc = tkw.mma(a_reg, b_reg, acc)
           return acc

       tkw.write(repeat, c)

   # Compile with multi-wave configuration
   # 4 waves per workgroup (2x2), BLOCK_K=64
   options = WaveCompileOptions(
       subs={
           M: 64, N: 64, K: 128,
           BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 64,
           ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
           ADDRESS_SPACE_0: tkl.AddressSpace.GLOBAL_MEMORY.value,
       },
       backend="asm",
       wave_runtime=True,
       compile_to_mlir=False
   )

   compiled_kernel = wave_compile(options, gemm_kernel)

K-Loop Key Features
~~~~~~~~~~~~~~~~~~~

This example demonstrates several advanced features working together:

1. **Multi-Wave Execution**: BLOCK_M=32, BLOCK_N=32 with WAVE_M=16, WAVE_N=16 creates 4 waves per workgroup
2. **K-Dimension Tiling**: BLOCK_K=64 tiles the K dimension for cache efficiency
3. **MFMA Accumulator Chaining**: Accumulator persists across loop iterations
4. **LDS Staging**: Input tiles automatically staged through shared memory
5. **Loop Control Flow**: Efficient SGPR-based loop counter and comparison
6. **CFG-Based Liveness**: Loop-carried values correctly tracked across iterations

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the 64x64x128 GEMM with BLOCK_K=64 configuration:

- **Iterations**: K/BLOCK_K = 128/64 = 2 iterations
- **Waves per Workgroup**: (32/16) * (32/16) = 4 waves
- **MFMAs per Iteration**: 4 MFMAs (one per K-tile chunk)
- **Total MFMAs**: 2 iterations * 4 MFMAs = 8 chained MFMAs
- **Register Pressure**: ~60 VGPRs (accumulator + temps), ~32 SGPRs
- **LDS Usage**: Staging buffers for A and B tiles

Performance Considerations
----------------------------

The ASM backend is designed for performance-critical applications:

- **Direct Assembly**: Eliminates intermediate compilation steps
- **Common Subexpression Elimination**: Scoped caching with global scope for loop-invariant expressions
- **Algebraic Simplification**: Symbol bounds enable aggressive simplification (e.g., floor elimination)
- **Instruction Fusion**: Peephole optimizer fuses shift+add/or into single instructions
- **Bit Range Analysis**: Uses OR instead of ADD when operand bit ranges don't overlap
- **LDS Offset Optimization**: Uses ds_read/ds_write offset field (0-8192 bytes) to reduce VALU instructions
- **Optimized Instructions**: Uses the most efficient AMDGCN instructions (shifts over multiplies, masks over divides)
- **Register Efficiency**: CFG-based liveness analysis minimizes register pressure
- **Memory Bandwidth**: Optimizes memory access patterns with base+offset addressing for maximum throughput
- **Precise Hazard Mitigation**: Only inserts s_nop where actually needed (reduces NOPs from ~46 to ~2)
- **Hardware Acceleration**: Leverages MFMA for matrix operations on CDNA architectures with accumulator chaining
- **LDS Staging**: Automatically uses shared memory for improved memory access patterns
- **Loop Efficiency**: SGPR-based loop counters (reserved at s24+) and VGPR accumulators for minimal overhead
- **Dynamic Allocation**: Computes exact register requirements for minimal resource usage
- **Architecture-Specific**: Adapts to hardware granularities for optimal register allocation

Performance vs LLVM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend typically achieves:

- **VALU Instructions**: ~62 vs LLVM's ~33 (gap primarily due to fused 3-operand instructions)
- **VGPRs**: ~20 vs LLVM's ~18
- **SGPRs**: ~40 vs LLVM's ~24
- **Performance**: ~65-70% of LLVM backend throughput

The remaining VALU gap is due to LLVM's use of fused 3-operand instructions like ``v_or3_b32`` (OR three operands) which the ASM backend does not yet emit.

Best Practices
~~~~~~~~~~~~~~

For optimal performance with the ASM backend:

1. **Use Power-of-2 Dimensions**: Enables optimal shift/mask instruction generation
2. **Align Memory Access**: 16-byte aligned access patterns maximize bandwidth
3. **Leverage LDS**: Use shared memory (ADDRESS_SPACE.SHARED_MEMORY) for frequently accessed data
4. **Enable MFMA**: Use matrix operations on CDNA architectures for best performance
5. **Use K-Loops**: Tile the K dimension with ``TilingConstraint`` and ``@tkw.iterate`` for cache efficiency
6. **Chain MFMA Accumulators**: Pass accumulators through loop ``iter_args`` for optimal accumulation
7. **Minimize Register Pressure**: Keep working sets small to maximize occupancy
8. **Multi-Wave Configurations**: Use multiple waves per workgroup to increase parallelism (up to 16 waves)
9. **Profile and Iterate**: Use ROCm profiling tools to identify bottlenecks

Limitations
-----------

The ASM backend has some limitations:

- **AMD GPU Only**: Only supports AMD GPUs with ROCm
- **Power-of-2 Constraints**: Non-power-of-2 modulo and division operations are not supported
- **Expression Complexity**: Some very complex affine expressions may not be supported
- **CDNA for MFMA**: MFMA operations require CDNA2 or CDNA3 architecture (gfx90a, gfx940, gfx941, gfx942)
- **Gather-to-LDS**: Requires gfx95x architecture with buffer_load...lds instruction support
- **Dynamic Shapes**: Requires concrete shape values at compile time
- **Loop Nesting**: While multiple loops are supported, deeply nested loops may increase register pressure
- **No Spilling**: Register allocation fails if registers cannot be allocated (no spill code generation)

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: ValueError: Mod divisor must be power-of-two**

This occurs when using modulo with non-power-of-2 divisors:

.. code-block:: python

   # Bad: modulo by 3 not supported
   idx = tid_x % 3  # ERROR

   # Good: modulo by power-of-2
   idx = tid_x % 16  # OK - uses v_and_b32

**Issue: hipErrorNoBinaryForGpu**

This can occur due to incorrect register metadata. The backend now computes this automatically, but if you encounter this:

- Ensure you're using the correct target architecture (gfx942, gfx90a, etc.)
- Check that register allocations respect granularity constraints
- Verify LDS size doesn't exceed hardware limits

**Issue: NaN Results from MFMA**

This typically indicates missing synchronization:

- The backend automatically inserts ``s_waitcnt lgkmcnt(0)`` before MFMA
- Ensure LDS staging is configured correctly with SHARED_MEMORY address space
- Verify that workgroup size and constraints are properly configured

**Issue: Register Allocation Errors**

The backend uses CFG-based liveness and linear scan allocation:

- Check ``amdhsa_next_free_vgpr`` and ``amdhsa_next_free_sgpr`` in generated assembly
- Ensure allocations are aligned to granularity (VGPR: 4, SGPR: 8 or 16)
- Verify that workgroup size is properly specified in MLIR ``translation_info`` attribute
- For K-loops, ensure BLOCK_K is sized appropriately to avoid excessive temporary registers

**Issue: High Register Pressure with K-Loops**

If K-loops cause high VGPR usage:

- **Increase BLOCK_K**: Larger BLOCK_K means fewer iterations but more work per iteration
- **Reduce Waves per Workgroup**: Fewer waves reduce parallelism but lower register pressure
- **Check Liveness**: Use ``WAVE_DEBUG_LIVENESS=1`` to see computed live ranges

Debugging
~~~~~~~~~

To debug ASM backend issues:

1. **Enable Assembly Output**:

   .. code-block:: python

      options = WaveCompileOptions(
          # ... other options ...
          compile_to_asm=True
      )

2. **Inspect Generated Assembly**: Look for register allocations, instruction sequences, and metadata

3. **Use ROCm Tools**: ``rocgdb``, ``rocprof``, and ``rocm-smi`` for runtime debugging

4. **Check MLIR Output**: Enable ``compile_to_mlir=True`` to see intermediate representation

5. **Debug Environment Variables**:

   - ``WAVE_DEBUG_LIVENESS=1``: Show liveness analysis results
   - ``WAVE_DEBUG_REGALLOC=1``: Show register allocation decisions
   - ``WAVE_LDS_DSREAD_OFFSET_DEBUG=1``: Show ds_read offset optimization decisions
