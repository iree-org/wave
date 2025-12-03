ASM Backend
===========

The ASM backend is Wave's direct AMDGCN assembly code generation backend that compiles Wave kernels to native AMD GPU assembly instructions. This backend provides fine-grained control over GPU execution and enables advanced optimizations not possible with higher-level backends.

Overview
--------

The ASM backend transforms Wave kernels through the following pipeline:

1. **MLIR Generation**: Wave kernels are first compiled to MLIR (Multi-Level Intermediate Representation)
2. **MLIR Analysis**: The MLIR is analyzed to extract kernel information, memory access patterns, and thread organization
3. **Assembly Generation**: AMDGCN assembly instructions are generated directly from the analyzed MLIR
4. **Binary Compilation**: The assembly is compiled to HSACO (Heterogeneous System Architecture Code Object) binaries using AMD's toolchain

Architecture
------------

The ASM backend consists of several key components:

The ASM backend follows a modular architecture with clear separation of concerns between MLIR analysis, assembly generation, and register management.

Key Components
~~~~~~~~~~~~~~

**MLIR Walker** (`mlir_walker.py`)
  Analyzes MLIR operations and extracts kernel information including:

  - Function arguments and their types
  - Memory access patterns (loads/stores)
  - Thread ID operations and bounds
  - Affine expressions and their simplifications
  - Binding subspan operations for memory mapping
  - Unified SSA-to-VGPR tracking for all operations
  - Loop iteration arguments and result mappings

**ASM Emitter** (`asm_emitter.py`)
  Generates AMDGCN assembly instructions from kernel information:

  - Kernel preamble and metadata
  - Register allocation and management
  - Memory access instructions (buffer loads/stores)
  - Thread synchronization and control flow
  - Shader Resource Descriptor (SRD) setup

**Instruction Classes** (`instructions.py`)
  Provides structured representation of AMDGCN instructions:

  - Base instruction classes for different operation types
  - Specific instruction implementations (loads, stores, arithmetic)
  - Loop control flow instructions (s_cmp_lt_u32, s_cbranch_scc1, s_branch, s_add_u32)
  - VGPR-variant MFMA with accumulator support
  - Instruction builders for common patterns
  - Proper assembly formatting and syntax

**Register Allocator** (`register_allocator.py`)
  Manages GPU register allocation:

  - Scalar General Purpose Register (SGPR) allocation
  - Vector General Purpose Register (VGPR) allocation
  - Register conflict detection and resolution
  - Alignment requirements for vector operations
  - Architecture-specific granularities (CDNA2/3: VGPR=4, SGPR=8; RDNA2/3: VGPR=4, SGPR=16)
  - Free list-based register reuse for optimal register pressure
  - Peak VGPR usage tracking for performance analysis
  - Typed operation logging with RegisterOp enum

**Expression Emitter** (`expression_emitter.py`)
  Generic SymPy expression visitor that emits AMDGCN instructions with CSE:

  - Automatic Common Subexpression Elimination (CSE) with memoization
  - Expression canonicalization to maximize CSE hits (flatten, sort, fold constants)
  - Iterative postorder traversal of expression trees
  - Support for constants, symbols, and complex expressions
  - Optimized instruction selection (shifts for power-of-2, masks for modulo)
  - Const marker system for efficient register usage
  - Handles Add, Mul, Mod, floor division, and Pow operations
  - Rational expression support for division in address calculations
  - Structural expression keys for cache lookup
  - SGPR reference handling for loop induction variables
  - Multi-wave thread ID extraction (tid_x, tid_y) from flat ID
  - Error handling for unsupported expression types

**Utils** (`utils.py`)
  Provides utility functions for:

  - MLIR type parsing and analysis
  - Affine expression simplification using SymPy
  - Thread ID bound analysis
  - Expression-to-assembly conversion
  - SymPy expression building from MLIR indices
  - Byte offset calculation for memory addressing

**Handlers** (`handlers.py`)
  Operation-specific handlers for MLIR operations:

  - Memory allocation (memref.alloc) including LDS staging
  - Memory views (memref.view) with offset tracking
  - Load operations (vector.load) from global and LDS memory
  - Store operations (vector.store) to global and LDS memory
  - MFMA operations (amdgpu.mfma) with accumulator chaining
  - LDS read/write operations (ds_read_b64, ds_write_b64)
  - Loop operations (scf.for) with induction variables and accumulators
  - Vector extraction (vector.extract_strided_slice) for register slicing

Features
--------

Direct Assembly Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend generates native AMDGCN assembly instructions, providing:

- **Fine-grained control** over GPU execution
- **Optimized memory access patterns** with proper SRD setup
- **Efficient register usage** through intelligent allocation
- **Thread synchronization** with proper wait instructions

Advanced Optimizations
~~~~~~~~~~~~~~~~~~~~~~

The backend implements several optimization techniques:

- **Common Subexpression Elimination (CSE)**: Automatically detects and reuses identical expressions across operations
- **Expression Canonicalization**: Normalizes expressions (flatten, sort, fold) to maximize CSE effectiveness
- **Affine Expression Simplification**: Uses SymPy to simplify complex index expressions
- **Thread ID Analysis**: Automatically detects and optimizes thread ID usage patterns
- **Memory Access Optimization**: Generates efficient buffer load/store sequences with base+offset addressing
- **Register Lifetime Management**: Frees temporary VGPRs promptly while preserving cached expressions
- **Register Reuse**: Minimizes register pressure through intelligent allocation and CSE
- **Power-of-2 Optimization**: Uses bit shifts instead of multiplication for power-of-2 constants
- **Const Marker System**: Avoids unnecessary register allocation for integer constants

Memory Management
~~~~~~~~~~~~~~~~~

The ASM backend handles memory operations through:

- **Shader Resource Descriptors (SRDs)**: Proper setup for buffer access
- **Vectorized Loads/Stores**: Efficient 16-byte aligned memory operations
- **Address Calculation**: Optimized offset computation using SymPy expressions
- **LDS (Local Data Share) Staging**: Automatic staging through shared memory for improved performance
- **Synchronization**: Proper wait instructions for memory consistency (vmcnt, lgkmcnt)
- **Expression-Based Addressing**: Dynamic address calculation from MLIR affine maps

Hardware Accelerated Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend provides native support for AMD GPU specialized instructions:

- **MFMA (Matrix Multiply-Accumulate)**: Hardware-accelerated matrix operations on CDNA architectures using VGPR-variant instructions with accumulator chaining for K-loops
- **LDS Operations**: Fast shared memory operations (ds_read_b64, ds_write_b64)
- **Multi-Wave Support**: Automatic detection and handling of multi-wave workgroups with proper thread ID extraction
- **Multi-Workgroup Support**: Dynamic detection of workgroup ID usage and conditional SGPR allocation
- **Loop Support (scf.for)**: Native support for structured control flow loops with SGPR induction variables and VGPR accumulators

Architecture Support
~~~~~~~~~~~~~~~~~~~~

The ASM backend supports multiple AMD GPU architectures with architecture-specific optimizations:

- **CDNA3 (gfx942)**: MI300 series with VGPR granularity of 4, SGPR granularity of 8

Dynamic Register Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend features fully dynamic register allocation:

- **Automatic VGPR Allocation**: Computes required VGPRs based on actual usage
- **Automatic SGPR Allocation**: Computes required SGPRs based on actual usage
- **Conditional System Register Allocation**: Dynamically detects workgroup ID and thread ID usage from MLIR
- **Granularity Alignment**: Automatically rounds allocations to architecture-specific granularities
- **VGPR-Variant MFMA**: Uses MFMA instructions that write directly to VGPRs, eliminating accumulator complexity

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

1. **Loop Initialization**: Allocates SGPR for loop counter, step, and upper bound
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

- **SGPR Induction Variables**: Loop counters stored in SGPRs for efficiency
- **VGPR Accumulators**: Loop-carried values (e.g., MFMA results) stay in VGPRs
- **Nested Loops**: Support for multiple active loop contexts via loop_stack
- **Unique Labels**: Each loop gets unique labels (loop_N_header, loop_N_body, etc.)

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

Affine Expression Simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend uses SymPy to simplify complex affine expressions:

.. code-block:: python

   # Complex index expression
   @tkw.wave(constraints)
   def complex_index_kernel(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
       # This expression gets simplified automatically
       idx = tkl.affine.apply(lambda d0, s0: d0 - (d0 // 64) * 64, [tkl.tid.x])
       res = tkw.read(a[idx, 0])
       tkw.write(res, b[idx, 0])

The backend automatically simplifies ``d0 - (d0 // 64) * 64`` to just ``d0`` when ``d0 < 64``.

Thread ID Analysis
~~~~~~~~~~~~~~~~~~

The backend analyzes thread ID usage patterns:

.. code-block:: python

   @tkw.wave(constraints)
   def thread_id_kernel(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
       # Backend automatically detects this is just tid.x
       idx = tkl.affine.apply(lambda d0: d0, [tkl.tid.x])
       res = tkw.read(a[idx, 0])

The backend recognizes that the affine expression simplifies to the thread ID and generates optimal code.

Unified SSA-to-VGPR Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend maintains a unified mapping from MLIR SSA values to their allocated VGPRs through the ``ssa_to_vgpr`` dictionary in ``IRWalker``:

**Design**

- **Single Source of Truth**: All SSA→VGPR mappings are stored in one dictionary
- **Operation Integration**: All handlers (loads, MFMAs, extracts, loops) populate ``ssa_to_vgpr``
- **Type Flexibility**: Values can be single registers, pairs, quads, or lists
- **Lifecycle Management**: Entries persist across operations for proper data flow

**Supported Operations**

- **Global Loads**: ``buffer_load_*`` results mapped to allocated VGPR tuples
- **LDS Reads**: ``ds_read_b64`` results mapped to VGPR pairs
- **MFMA Results**: Matrix multiply outputs mapped to VGPR quads
- **Loop Accumulators**: ``iter_args`` and results mapped to persistent VGPRs
- **Vector Extracts**: ``vector.extract_strided_slice`` maps to register subsets

**Example Flow**

.. code-block:: python

   # Load A matrix → tracked in ssa_to_vgpr
   %a_vec = vector.load %A[...]
   # ssa_to_vgpr["%a_vec"] = (v24, v25)  # VGPR pair

   # Load B matrix → tracked in ssa_to_vgpr
   %b_vec = vector.load %B[...]
   # ssa_to_vgpr["%b_vec"] = (v26, v27)  # VGPR pair

   # MFMA operation uses ssa_to_vgpr to find inputs
   %result = amdgpu.mfma %a_vec, %b_vec, %acc
   # Looks up: ssa_to_vgpr["%a_vec"] → (v24, v25)
   #           ssa_to_vgpr["%b_vec"] → (v26, v27)
   #           ssa_to_vgpr["%acc"] → (v4, v5, v6, v7)
   # Emits: v_mfma_f32_16x16x16_f16 v[4:7], v[24:25], v[26:27], v[4:7]
   # Tracks: ssa_to_vgpr["%result"] = (v4, v5, v6, v7)

   # Vector extraction creates new mapping to subset
   %slice = vector.extract_strided_slice %result [offset=0, size=2]
   # ssa_to_vgpr["%slice"] = (v4, v5)  # First two regs of quad

   # Store uses ssa_to_vgpr to find source registers
   vector.store %result, %C[...]
   # Looks up: ssa_to_vgpr["%result"] → (v4, v5, v6, v7)
   # Emits: buffer_store_dword v4, ...
   #        buffer_store_dword v5, ...

**Benefits**

- **Simplified Code**: Eliminates fragmented tracking variables
- **Better Debugging**: Single place to inspect SSA→register mappings
- **Correct Data Flow**: Ensures operations use the right registers
- **Loop Support**: Natural fit for tracking loop-carried values

Expression Visitor System
~~~~~~~~~~~~~~~~~~~~~~~~~

The ASM backend uses a sophisticated expression visitor (``ExprEmitter``) to convert SymPy expressions to AMDGCN assembly with automatic Common Subexpression Elimination:

**Supported Operations:**

- **Constants and Symbols**: Direct materialization into registers
- **Addition**: Efficient left-to-right accumulation using ``v_add_u32``
- **Multiplication**: Power-of-2 uses ``v_lshlrev_b32`` (shift), others use ``v_mul_lo_u32``
- **Modulo**: Power-of-2 divisors use ``v_and_b32`` (mask)
- **Floor Division**: Power-of-2 divisors use ``v_lshrrev_b32`` (shift)
- **Power of 2**: Constant folding in expressions

**Optimizations:**

- **Common Subexpression Elimination**: Automatically caches and reuses emitted expressions
- **Expression Canonicalization**: Normalizes expressions to maximize cache hits (flatten Add/Mul, sort operands, fold constants)
- **Structural Expression Keys**: Uses structural hashing for efficient cache lookup
- **Const/Dynamic Splitting**: Separates constant offsets into instruction immediates
- **Lifetime Management**: Frees temporary registers while preserving cached values
- **Const Marker System**: Avoids allocating registers for intermediate constants
- **Iterative Postorder Traversal**: Efficient expression tree walking
- **Register Reuse**: Minimizes temporary register allocation through CSE
- **Instruction Selection**: Chooses optimal instructions based on operand types

**Example Expression Handling:**

.. code-block:: python

   # Complex index expression: row*256 + col*4
   # where row = tid_x // 16, col = tid_x % 16
   # Used multiple times in load/store operations

   # The backend automatically:
   # 1. Canonicalizes the expression (flatten, sort, fold)
   # 2. Simplifies floor division by 16 to right shift by 4
   # 3. Simplifies modulo 16 to mask with 15
   # 4. Optimizes multiplication by 256 to left shift by 8
   # 5. Optimizes multiplication by 4 to left shift by 2
   # 6. Caches the result for reuse across multiple operations
   # 7. Accumulates results efficiently

   # Generated assembly (simplified) - computed once, reused multiple times:
   # v_lshrrev_b32 v2, 4, v1      # row = tid_x >> 4 (cached)
   # v_lshlrev_b32 v2, 8, v2       # row * 256 (cached)
   # v_and_b32 v3, 15, v1          # col = tid_x & 15 (cached)
   # v_lshlrev_b32 v3, 2, v3       # col * 4 (cached)
   # v_add_u32 v2, v2, v3          # row*256 + col*4 (cached in v2)
   #
   # # Subsequent uses of the same expression reuse v2:
   # buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen offset:0
   # buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen offset:0
   # # No duplicate shift/mask instructions emitted!

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
6. **Unified Register Tracking**: All SSA values correctly tracked through loop iterations

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
- **Common Subexpression Elimination**: Eliminates redundant computations by caching and reusing expressions
- **Optimized Instructions**: Uses the most efficient AMDGCN instructions (shifts over multiplies, masks over divides)
- **Register Efficiency**: Minimizes register pressure through intelligent allocation, CSE, free lists, and const marker system
- **Memory Bandwidth**: Optimizes memory access patterns with base+offset addressing for maximum throughput
- **Lifetime Management**: Frees temporary registers promptly to reduce pressure while preserving cached values
- **Register Reuse**: Free list-based allocator reuses freed registers before allocating new ones
- **Hardware Acceleration**: Leverages MFMA for matrix operations on CDNA architectures with accumulator chaining
- **LDS Staging**: Automatically uses shared memory for improved memory access patterns
- **Loop Efficiency**: SGPR-based loop counters and VGPR accumulators for minimal overhead
- **Dynamic Allocation**: Computes exact register requirements for minimal resource usage
- **Architecture-Specific**: Adapts to hardware granularities for optimal register allocation
- **Unified Tracking**: Single SSA→VGPR mapping ensures correct data flow and simplifies debugging

VMEM Wait Optimization (Ticket-based vmcnt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To hide vector memory (VMEM) latency and avoid over-synchronization, the backend uses a
ticket-based scheme to place the minimal required ``s_waitcnt vmcnt(N)`` right before
the first use of loaded data.

- Each ``buffer_load_*`` is assigned a monotonically increasing ticket ``T``.
- When consuming data from ticket ``K`` and the last issued ticket is ``T``, the backend emits
  ``s_waitcnt vmcnt(T - K)``. This allows newer loads to remain in flight while ensuring
  the data for ticket ``K`` is ready.
- Waits are placed at the first use (e.g., before an LDS write or a compute op), not after the load.
- The threshold coalesces across nearby uses and is reset when new loads are issued.

Example (conceptual)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: asm

   # Issue two loads back-to-back (both in-flight)
   buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen offset:0   # ticket 0
   buffer_load_dwordx4 v[8:11], v2, s[8:11], 0 offen offset:16  # ticket 1

   # Minimal wait before first use of the data from ticket 0
   s_waitcnt vmcnt(1)   # allow one newer load (ticket 1) to remain outstanding
   buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen offset:0

   # Later when using data from ticket 1
   s_waitcnt vmcnt(0)
   buffer_store_dwordx4 v[8:11], v2, s[12:15], 0 offen offset:16

This placement hides memory latency behind independent address/index computation
and reduces the number of waits (and their strictness) compared to always using
``vmcnt(0)`` immediately after each load.

Latency-Aware Scheduling (Database-driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond VMEM wait placement, the ASM backend employs a database-driven, latency-aware
scheduler to minimize stalls across VMEM, LGKM (LDS/scalar), VALU and MFMA pipelines.

- **Latency Database** (``latency_db/gfx942.json``): Versioned, per-architecture JSON
  containing instruction latencies and throughputs, plus hazard-specific distances
  (e.g. ``mfma_to_agpr_read``). Each entry has a measurement source:
  - ``isa_manual``: From AMD ISA documentation
  - ``llvm_codegen``: From LLVM's proven codegen patterns
  - ``measured`` / ``profiled``: From microbenchmarks or profiling (future-ready)

- **Latency Provider** (``latency_provider.py``): Query interface used by the emitter/handlers
  to retrieve latency/throughput and hazard distances. All values come from the JSON database;
  there are no hardcoded latencies in the code path.

- **Scoreboard** (``scoreboard.py``): Tracks outstanding instructions and their readiness,
  detects RAW/WAW hazards, and recommends minimal waits/NOPs. It also integrates with the
  ticket-based VMEM/LGKM model.

- **Always-On Integration** (``asm_emitter.py``): The emitter unconditionally initializes
  the Latency Provider and Scoreboard. Latency-aware scheduling is always enabled.

MFMA Scheduling Example
^^^^^^^^^^^^^^^^^^^^^^^

The backend uses VGPR-variant MFMA instructions that write results directly to VGPRs,
simplifying the instruction sequence and eliminating the need for accumulator transfers.
The latency-aware scheduler tracks MFMA execution to ensure proper timing before using results:

.. code-block:: asm

   # Wait for LDS data before MFMA
   s_waitcnt lgkmcnt(0)
   # VGPR-variant MFMA writes directly to VGPRs
   v_mfma_f32_16x16x16_f16 v[0:3], v[26:27], v[24:25], 0
   # Results in v[0:3] are ready after MFMA latency (~8 cycles)
   # Scheduler ensures proper timing before using v[0:3]
   buffer_store_dwordx4 v[0:3], v2, s[12:15], 0 offen

Benefits
^^^^^^^^

- Minimal waits and NOPs derived from a single source of truth (the database)
- Architecture-specific values without code changes
- VGPR-variant MFMA simplifies instruction sequences and matches LLVM backend behavior
- Dynamic workgroup/thread ID detection minimizes system register usage
- Ready to adopt measured data in the future

Best Practices
~~~~~~~~~~~~~~

For optimal performance with the ASM backend:

1. **Use Power-of-2 Dimensions**: Enables optimal shift/mask instruction generation
2. **Align Memory Access**: 16-byte aligned access patterns maximize bandwidth
3. **Leverage LDS**: Use shared memory (ADDRESS_SPACE.SHARED_MEMORY) for frequently accessed data
4. **Enable MFMA**: Use matrix operations on CDNA architectures for best performance
5. **Use K-Loops**: Tile the K dimension with ``TilingConstraint`` and ``@tkw.iterate`` for cache efficiency
6. **Chain MFMA Accumulators**: Pass accumulators through loop ``iter_args`` for optimal accumulation
7. **Minimize Register Pressure**: Keep working sets small to maximize occupancy; use free lists naturally via allocator
8. **Multi-Wave Configurations**: Use multiple waves per workgroup to increase parallelism (up to 16 waves)
9. **Profile and Iterate**: Use ROCm profiling tools to identify bottlenecks

Limitations
-----------

The ASM backend has some limitations:

- **AMD GPU Only**: Only supports AMD GPUs with ROCm
- **Power-of-2 Constraints**: Non-power-of-2 modulo and division operations are not supported
- **Expression Complexity**: Some very complex affine expressions may not be supported
- **CDNA for MFMA**: MFMA operations require CDNA2 or CDNA3 architecture (gfx90a, gfx940, gfx941, gfx942)
- **Dynamic Shapes**: Requires concrete shape values at compile time
- **Loop Nesting**: While multiple loops are supported, deeply nested loops may increase register pressure

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

The backend now dynamically computes register requirements:

- Check ``amdhsa_next_free_vgpr`` and ``amdhsa_next_free_sgpr`` in generated assembly
- Ensure allocations are aligned to granularity (VGPR: 4, SGPR: 8 or 16)
- Verify that workgroup size is properly specified in MLIR ``translation_info`` attribute
- For K-loops, ensure BLOCK_K is sized appropriately to avoid excessive temporary registers

**Issue: High Register Pressure with K-Loops**

If K-loops cause high VGPR usage:

- **Increase BLOCK_K**: Larger BLOCK_K means fewer iterations but more work per iteration
- **Reduce Waves per Workgroup**: Fewer waves reduce parallelism but lower register pressure
- **Check Free Lists**: The allocator reuses freed registers; ensure temps are freed properly
- **Profile**: Use ``DEBUG_ASM_ALLOCATOR=1`` to see allocation patterns and peak usage

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
