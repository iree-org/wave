#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Standalone E2E test for C++ WaveASM backend.

This script tests the C++ waveasm-translate tool by:
1. Defining a copy kernel using wave_lang
2. Capturing the MLIR IR before ASM codegen
3. Compiling to assembly using C++ waveasm-translate
4. Assembling to GPU binary using amdclang++
5. Loading and executing on GPU using wave_runtime
6. Validating results against PyTorch

Run on mi350-4 docker:
    cd /path/to/wave-asm
    python test/e2e/run_cpp_backend_e2e.py

Environment variables:
    WAVEASM_TRANSLATE: Path to waveasm-translate (default: auto-detect in build/)
    WAVE_DEFAULT_ARCH: Target architecture (default: auto-detect from GPU)
    ROCM_PATH: ROCm installation path (default: /opt/rocm)
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

import torch
from torch.testing import assert_close


@dataclass
class CompilationResult:
    """Result of compilation through C++ WaveASM backend."""
    mlir_text: str
    asm_text: str
    binary_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    if "WAVEASM_TRANSLATE" in os.environ:
        return Path(os.environ["WAVEASM_TRANSLATE"])

    # Default: look in wave-asm build directory
    script_dir = Path(__file__).parent
    default_path = script_dir.parent.parent / "build" / "tools" / "waveasm-translate" / "waveasm-translate"

    if default_path.exists():
        return default_path

    # Try to find it in PATH
    try:
        result = subprocess.run(["which", "waveasm-translate"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    raise FileNotFoundError(
        "waveasm-translate not found. Set WAVEASM_TRANSLATE environment variable "
        "or build wave-asm project."
    )


def get_amdclang_path() -> str:
    """Get path to amdclang++ for assembly compilation."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    amdclang = os.path.join(rocm_path, "bin", "amdclang++")

    if os.path.exists(amdclang):
        return amdclang

    try:
        result = subprocess.run(["which", "amdclang++"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    raise FileNotFoundError(
        "amdclang++ not found. Ensure ROCm is installed and in PATH."
    )


def get_target_arch() -> str:
    """Get target architecture from environment or detect."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        return arch.split(":")[0]  # Strip feature flags

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, 'gcnArchName'):
            return props.gcnArchName.split(":")[0]

    return "gfx942"


class WaveASMCompiler:
    """Compiler for MLIR -> Assembly using C++ WaveASM backend."""

    def __init__(
        self,
        target: str = "gfx942",
        codeobj: str = "5",
        keep_temp_files: bool = False,
    ):
        self.target = target
        self.codeobj = codeobj
        self.keep_temp_files = keep_temp_files
        self.waveasm_translate = get_waveasm_translate_path()
        self.amdclang = get_amdclang_path()
        self._temp_dir = None

    def _get_temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="waveasm_cpp_e2e_"))
        return self._temp_dir

    def compile_mlir_to_asm(self, mlir_text: str) -> Tuple[bool, str, str]:
        """
        Compile MLIR to AMDGCN assembly using C++ waveasm-translate.

        Returns:
            Tuple of (success, asm_text_or_error, stderr)
        """
        temp_dir = self._get_temp_dir()
        mlir_file = temp_dir / "input.mlir"
        asm_file = temp_dir / "output.s"

        mlir_file.write_text(mlir_text)

        # Run waveasm-translate with full pipeline
        cmd = [
            str(self.waveasm_translate),
            f"--target={self.target}",
            "--waveasm-scoped-cse",
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
            str(mlir_file),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return False, result.stderr, result.stderr

            asm_text = result.stdout
            asm_file.write_text(asm_text)

            return True, asm_text, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out", ""
        except Exception as e:
            return False, str(e), ""

    def assemble_to_binary(self, asm_text: str) -> Tuple[bool, Optional[Path], str]:
        """
        Assemble AMDGCN assembly to GPU binary using amdclang++.

        This follows the same flow as Python _compile_asm_to_binary in compile.py.

        Returns:
            Tuple of (success, binary_path, error_message)
        """
        temp_dir = self._get_temp_dir()
        asm_file = temp_dir / "kernel.s"
        obj_file = temp_dir / "kernel.o"
        hsaco_file = temp_dir / "kernel.hsaco"

        asm_file.write_text(asm_text)

        # Step 1: Assemble to object file (same as Python backend)
        compile_cmd = [
            self.amdclang,
            "-x", "assembler",
            "-target", "amdgcn-amd-amdhsa",
            f"-mcode-object-version={self.codeobj}",
            f"-mcpu={self.target}",
            "-mwavefrontsize64",
            "-c",
            str(asm_file),
            "-o", str(obj_file),
        ]

        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return False, None, f"Assembly failed: {result.stderr}"

            # Step 2: Link to HSACO (same as Python backend)
            link_cmd = [
                self.amdclang,
                "-target", "amdgcn-amd-amdhsa",
                "-Xlinker", "--build-id=sha1",
                "-o", str(hsaco_file),
                str(obj_file),
            ]

            result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                return False, None, f"Linking failed: {result.stderr}"

            return True, hsaco_file, ""

        except subprocess.TimeoutExpired:
            return False, None, "Assembly/linking timed out"
        except Exception as e:
            return False, None, str(e)

    def compile_full(self, mlir_text: str) -> CompilationResult:
        """Full compilation pipeline: MLIR -> ASM -> Binary."""
        success, asm_or_error, stderr = self.compile_mlir_to_asm(mlir_text)

        if not success:
            return CompilationResult(
                mlir_text=mlir_text,
                asm_text="",
                binary_path=None,
                success=False,
                error_message=f"MLIR->ASM failed: {asm_or_error}",
            )

        asm_text = asm_or_error

        success, binary_path, error = self.assemble_to_binary(asm_text)

        if not success:
            return CompilationResult(
                mlir_text=mlir_text,
                asm_text=asm_text,
                binary_path=None,
                success=False,
                error_message=f"ASM->Binary failed: {error}",
            )

        return CompilationResult(
            mlir_text=mlir_text,
            asm_text=asm_text,
            binary_path=binary_path,
            success=True,
        )

    def cleanup(self):
        if not self.keep_temp_files and self._temp_dir is not None:
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None


def extract_func_from_stream_mlir(mlir_text: str) -> str:
    """
    Extract func.func operations from IREE stream.executable wrapped MLIR.

    The wave_lang compiler wraps kernels in stream.executable, but waveasm-translate
    expects plain func.func or gpu.func operations. This function extracts the
    inner function using the Python MLIR bindings which can handle stream dialect.

    The extracted MLIR is printed in generic op form so it can be parsed by
    waveasm-translate with allowUnregisteredDialects.
    """
    from wave_lang.support.ir_imports import Context, Module, func_d

    def walk_ops_recursively(operation):
        for region in operation.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    yield inner_op
                    yield from walk_ops_recursively(inner_op)

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(mlir_text)

        # Find all func.func operations
        funcs = []
        for op in walk_ops_recursively(module.operation):
            if isinstance(op, func_d.FuncOp):
                # Skip wrapper functions (async, benchmark scaffolding)
                name = op.sym_name.value
                if name.startswith("isolated_benchmark") or name.endswith("$async"):
                    continue
                # Use print_generic_op_form=True so stream ops use generic syntax
                # that can be parsed with allowUnregisteredDialects
                funcs.append(op.get_asm(print_generic_op_form=True))

        if not funcs:
            raise ValueError("No kernel func.func found in MLIR")

        # Wrap extracted functions in a module
        return "module {\n" + "\n".join(funcs) + "\n}\n"


def capture_wave_mlir(options, kernel_func) -> str:
    """
    Capture the MLIR IR from a Wave kernel before ASM codegen.

    This properly sets up the IndexingContext required by wave_lang.
    Returns MLIR with just func.func (extracts from stream.executable wrapper).
    """
    from wave_lang.kernel._support.indexing import IndexingContext
    from wave_lang.kernel.wave.compile import (
        _trace_launchable_and_get_kernel_signature,
    )

    # Must run within IndexingContext like wave_compile does
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)

        # Initialize kernel constraints (same as wave_compile)
        kernel_func.initialize_wave_constraints()
        kernel_func.initialize_symbolic_constraints()
        kernel_func.initialize_workgroup_constraints()

        # Trace and get MLIR
        result = _trace_launchable_and_get_kernel_signature(kernel_func, options)
        # Result is a tuple: (mb, trace, exe, kernel_sig, entrypoint_name, options, ...)
        mb = result[0]

        full_mlir_text = mb.module_op.get_asm(
            enable_debug_info=False,
            use_local_scope=options.use_local_scope,
        )

    # Extract just the func.func from stream.executable wrapper
    mlir_text = extract_func_from_stream_mlir(full_mlir_text)

    return mlir_text


def run_with_wave_runtime(
    binary_path: Path,
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    shared_memory_bytes: int = 0,
    func_name: str = "isolated_benchmark",
):
    """
    Execute a compiled GPU binary using wave_runtime.
    """
    import wave_runtime

    # Initialize HIP functions
    wave_runtime.load_hip_functions()

    # Load GPU function from binary
    gpu_binary, gpu_func = wave_runtime.load_binary(str(binary_path), func_name)

    # Create launch info
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        torch.cuda.current_stream().cuda_stream,
        gpu_func,
        shared_memory_bytes,
        grid[0],
        grid[1],
        grid[2],
        block[0],
        block[1],
        block[2],
        1, 1, 1,  # cluster_dims
    )

    # Prepare kernel arguments
    all_tensors = inputs + outputs
    kern_args = [tensor.data_ptr() for tensor in all_tensors]
    kernel_args = wave_runtime.Int64Vector(kern_args)

    # Launch
    wave_runtime.launch(kernel_launch_info, kernel_args, [], [])

    # Sync
    torch.cuda.synchronize()


def test_copy_kernel_cpp_backend():
    """
    Full end-to-end test for copy kernel using C++ WaveASM backend.

    This is equivalent to test_copy_kernel_asm_backend in asm_backend_test.py,
    but uses the C++ backend instead of Python backend.
    """
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros

    print("=" * 60)
    print("C++ WaveASM Backend E2E Test: Copy Kernel")
    print("=" * 60)

    # Get target architecture
    target = get_target_arch()
    print(f"Target architecture: {target}")

    # Check if GPU is available
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        return False

    # Define copy kernel (same as asm_backend_test.py)
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16
    shape = (16, 16)

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        ),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    @tkw.wave(constraints)
    def copy_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    # Create test data
    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)

    print(f"\nInput tensor shape: {a.shape}")
    print(f"Input tensor dtype: {a.dtype}")
    print(f"Input tensor (first 4 values): {a.flatten()[:4].tolist()}")

    # Create compile options
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Step 1: Capture MLIR
    print("\n[Step 1] Capturing MLIR from kernel...")
    try:
        mlir_text = capture_wave_mlir(options, copy_kernel)
        print(f"  Captured {len(mlir_text)} bytes of MLIR")
    except Exception as e:
        print(f"  ERROR: Failed to capture MLIR: {e}")
        return False

    # Save MLIR for inspection
    temp_dir = Path(tempfile.mkdtemp(prefix="waveasm_cpp_e2e_"))
    mlir_path = temp_dir / "copy_kernel.mlir"
    mlir_path.write_text(mlir_text)
    print(f"  Saved to: {mlir_path}")

    # Step 2: Compile MLIR to assembly using C++ backend
    print("\n[Step 2] Compiling MLIR to assembly (C++ backend)...")
    compiler = WaveASMCompiler(target=target, keep_temp_files=True)

    try:
        success, asm_or_error, stderr = compiler.compile_mlir_to_asm(mlir_text)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False

    if not success:
        print(f"  ERROR: Compilation failed")
        print(f"  {asm_or_error}")
        return False

    asm_text = asm_or_error
    asm_path = temp_dir / "copy_kernel.s"
    asm_path.write_text(asm_text)
    print(f"  Generated {len(asm_text)} bytes of assembly")
    print(f"  Saved to: {asm_path}")

    # Count key instructions
    lines = asm_text.split('\n')
    buffer_loads = sum(1 for l in lines if 'buffer_load' in l)
    buffer_stores = sum(1 for l in lines if 'buffer_store' in l)
    waitcnts = sum(1 for l in lines if 's_waitcnt' in l)
    print(f"  Instructions: {buffer_loads} buffer_load, {buffer_stores} buffer_store, {waitcnts} s_waitcnt")

    # Step 3: Assemble to binary
    print("\n[Step 3] Assembling to GPU binary...")
    try:
        success, binary_path, error = compiler.assemble_to_binary(asm_text)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return False

    if not success:
        print(f"  ERROR: Assembly failed")
        print(f"  {error}")
        return False

    print(f"  Binary path: {binary_path}")
    print(f"  Binary size: {binary_path.stat().st_size} bytes")

    # Step 4: Execute on GPU
    print("\n[Step 4] Executing on GPU...")
    try:
        run_with_wave_runtime(
            binary_path=binary_path,
            inputs=[a],
            outputs=[b],
            grid=(1, 1, 1),
            block=(64, 1, 1),
            shared_memory_bytes=0,
            func_name="copy_kernel",  # Match the kernel name from MLIR
        )
        print("  Kernel executed successfully")
    except Exception as e:
        print(f"  ERROR: Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Validate results
    print("\n[Step 5] Validating results...")
    print(f"  Output tensor (first 4 values): {b.flatten()[:4].tolist()}")

    try:
        assert_close(a, b)
        print("  PASSED: Output matches input")
    except AssertionError as e:
        print(f"  FAILED: Output does not match input")
        print(f"  {e}")
        return False

    print("\n" + "=" * 60)
    print("SUCCESS: C++ WaveASM Backend E2E Test Passed!")
    print("=" * 60)

    # Cleanup
    if not os.environ.get("KEEP_TEMP_FILES"):
        compiler.cleanup()

    return True


def test_compare_cpp_vs_python_backend():
    """
    Compare C++ and Python backend assembly output.
    """
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

    print("=" * 60)
    print("C++ vs Python Backend Comparison")
    print("=" * 60)

    target = get_target_arch()
    print(f"Target architecture: {target}")

    # Define copy kernel
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 16, N: 16},
        ),
        tkw.WorkgroupConstraint(M, 16, 0),
        tkw.WorkgroupConstraint(N, 16, 1),
        tkw.WaveConstraint(M, 16),
        tkw.WaveConstraint(N, 16),
    ]

    @tkw.wave(constraints)
    def copy_kernel(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    options = WaveCompileOptions(
        subs={
            M: 16,
            N: 16,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR
    print("\n[Step 1] Capturing MLIR...")
    mlir_text = capture_wave_mlir(options, copy_kernel)
    print(f"  Captured {len(mlir_text)} bytes")

    # C++ backend
    print("\n[Step 2] Compiling with C++ backend...")
    try:
        cpp_compiler = WaveASMCompiler(target=target)
        cpp_success, cpp_asm, _ = cpp_compiler.compile_mlir_to_asm(mlir_text)
        if not cpp_success:
            cpp_asm = f"C++ compilation failed: {cpp_asm}"
    except Exception as e:
        cpp_asm = f"C++ compilation failed: {e}"

    # Python backend
    print("\n[Step 3] Compiling with Python backend...")
    try:
        from wave_lang.kernel.wave.asm.kernel_module_compiler import KernelModuleCompiler
        python_compiler = KernelModuleCompiler(targetid=target, codeobj="5")
        python_asm = python_compiler.compile_mlir_string(mlir_text)
    except Exception as e:
        python_asm = f"Python compilation failed: {e}"

    # Compare
    print("\n" + "=" * 60)
    print("C++ Backend Output:")
    print("=" * 60)
    print(cpp_asm[:2000] if len(cpp_asm) > 2000 else cpp_asm)

    print("\n" + "=" * 60)
    print("Python Backend Output:")
    print("=" * 60)
    print(python_asm[:2000] if len(python_asm) > 2000 else python_asm)

    # Save for detailed diff
    temp_dir = Path(tempfile.mkdtemp(prefix="waveasm_compare_"))
    (temp_dir / "cpp_backend.s").write_text(cpp_asm if isinstance(cpp_asm, str) else str(cpp_asm))
    (temp_dir / "python_backend.s").write_text(python_asm if isinstance(python_asm, str) else str(python_asm))
    print(f"\nSaved to: {temp_dir}")

    # Basic comparison stats
    def count_instructions(asm):
        if "failed" in asm.lower():
            return {}
        lines = asm.split('\n')
        return {
            'total_lines': len(lines),
            'buffer_load': sum(1 for l in lines if 'buffer_load' in l),
            'buffer_store': sum(1 for l in lines if 'buffer_store' in l),
            's_waitcnt': sum(1 for l in lines if 's_waitcnt' in l),
            's_mov': sum(1 for l in lines if 's_mov' in l),
            's_load': sum(1 for l in lines if 's_load' in l),
        }

    print("\n" + "=" * 60)
    print("Instruction Count Comparison:")
    print("=" * 60)
    cpp_stats = count_instructions(cpp_asm)
    python_stats = count_instructions(python_asm)

    if cpp_stats and python_stats:
        print(f"{'Instruction':<20} {'C++':<10} {'Python':<10}")
        print("-" * 40)
        for key in cpp_stats:
            print(f"{key:<20} {cpp_stats[key]:<10} {python_stats.get(key, 'N/A'):<10}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="C++ WaveASM Backend E2E Test")
    parser.add_argument("--compare", action="store_true", help="Compare C++ vs Python backend")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    if args.keep_temp:
        os.environ["KEEP_TEMP_FILES"] = "1"

    if args.compare:
        test_compare_cpp_vs_python_backend()
    else:
        success = test_copy_kernel_cpp_backend()
        sys.exit(0 if success else 1)
