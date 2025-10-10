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
  - Instruction builders for common patterns
  - Proper assembly formatting and syntax

**Register Allocator** (`register_allocator.py`)
  Manages GPU register allocation:

  - Scalar General Purpose Register (SGPR) allocation
  - Vector General Purpose Register (VGPR) allocation
  - Register conflict detection and resolution
  - Alignment requirements for vector operations

**Utils** (`utils.py`)
  Provides utility functions for:

  - MLIR type parsing and analysis
  - Affine expression simplification using SymPy
  - Thread ID bound analysis
  - Expression-to-assembly conversion

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

- **Affine Expression Simplification**: Uses SymPy to simplify complex index expressions
- **Thread ID Analysis**: Automatically detects and optimizes thread ID usage patterns
- **Memory Access Optimization**: Generates efficient buffer load/store sequences
- **Register Reuse**: Minimizes register pressure through intelligent allocation

Memory Management
~~~~~~~~~~~~~~~~~

The ASM backend handles memory operations through:

- **Shader Resource Descriptors (SRDs)**: Proper setup for buffer access
- **Vectorized Loads/Stores**: Efficient 16-byte aligned memory operations
- **Address Calculation**: Optimized offset computation for memory access
- **Synchronization**: Proper wait instructions for memory consistency

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
4. **Register Allocation**: Intelligently allocates SGPRs and VGPRs to minimize register pressure
5. **Synchronization**: Proper wait instructions ensure memory consistency

Advanced Features
-----------------

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

Error Handling
~~~~~~~~~~~~~~

The ASM backend provides detailed error messages for unsupported operations:

.. code-block:: python

   # This will raise a clear error
   @tkw.wave(constraints)
   def unsupported_kernel(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
       # Complex expression that can't be simplified
       idx = tkl.affine.apply(lambda d0, s0: d0 * s0 + complex_function(d0), [tkl.tid.x, some_symbol])
       res = tkw.read(a[idx, 0])

This will raise: ``ValueError: Cannot emit ASM for expression with unresolved symbols: ['s0']. Expression: d0*s0 + complex_function(d0)``

Performance Considerations
----------------------------

The ASM backend is designed for performance-critical applications:

- **Direct Assembly**: Eliminates intermediate compilation steps
- **Optimized Instructions**: Uses the most efficient AMDGCN instructions
- **Register Efficiency**: Minimizes register pressure through intelligent allocation
- **Memory Bandwidth**: Optimizes memory access patterns for maximum throughput

Limitations
-----------

The ASM backend has some limitations:

- **AMD GPU Only**: Only supports AMD GPUs with ROCm
- **Limited Expression Support**: Some complex affine expressions may not be supported
- **Manual Optimization**: Requires understanding of AMDGCN assembly for advanced optimizations
