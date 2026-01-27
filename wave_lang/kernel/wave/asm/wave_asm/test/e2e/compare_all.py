#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compare Python and C++ WaveASM backend outputs for all kernels.

This script runs each kernel through both backends and compares instruction counts.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile, _trace_launchable_and_get_kernel_signature
from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

# Symbols
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

SHARED_ADDRESS_SPACE = tkl.AddressSpace.SHARED_MEMORY.value
GLOBAL_ADDRESS_SPACE = tkl.AddressSpace.GLOBAL_MEMORY.value


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    script_dir = Path(__file__).parent
    default_path = script_dir.parent.parent / "build" / "tools" / "waveasm-translate" / "waveasm-translate"
    if default_path.exists():
        return default_path
    raise FileNotFoundError(f"waveasm-translate not found at {default_path}")


def extract_func_from_stream(mlir_text: str) -> str:
    """Extract func.func from stream.executable wrapper."""
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

        funcs = []
        for op in walk_ops_recursively(module.operation):
            if isinstance(op, func_d.FuncOp):
                name = op.sym_name.value
                if name.startswith("isolated_benchmark") or name.endswith("$async"):
                    continue
                funcs.append(op.get_asm(print_generic_op_form=True))

        if not funcs:
            raise ValueError("No kernel func.func found in MLIR")

        return "module {\n" + "\n".join(funcs) + "\n}\n"


def compile_with_cpp_backend(mlir_text: str, target: str = "gfx942") -> str:
    """Compile MLIR using C++ backend via waveasm-translate."""
    waveasm_translate = get_waveasm_translate_path()

    # Extract func from stream wrapper
    try:
        simplified_mlir = extract_func_from_stream(mlir_text)
    except Exception as e:
        return f"C++ backend error: Failed to extract func: {e}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(simplified_mlir)
        mlir_file = f.name

    try:
        cmd = [
            str(waveasm_translate),
            mlir_file,
            f"--target={target}",
            "--waveasm-scoped-cse",
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            return f"C++ backend error: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "C++ backend error: Compilation timed out"
    except Exception as e:
        return f"C++ backend error: {e}"
    finally:
        Path(mlir_file).unlink(missing_ok=True)


def count_instructions(asm: str) -> dict:
    """Count key instructions in assembly."""
    counts = {
        'buffer_load': 0,
        'buffer_store': 0,
        'ds_read': 0,
        'ds_write': 0,
        'v_mfma': 0,
        's_barrier': 0,
        's_endpgm': 0,
        's_waitcnt': 0,
        'buffer_load_lds': 0,
    }
    for line in asm.split('\n'):
        lower = line.lower().strip()
        if lower.startswith('buffer_load') and 'lds' in lower:
            counts['buffer_load_lds'] += 1
        elif lower.startswith('buffer_load'):
            counts['buffer_load'] += 1
        elif lower.startswith('buffer_store'):
            counts['buffer_store'] += 1
        elif lower.startswith('ds_read'):
            counts['ds_read'] += 1
        elif lower.startswith('ds_write'):
            counts['ds_write'] += 1
        elif 'v_mfma' in lower:
            counts['v_mfma'] += 1
        elif lower.startswith('s_barrier'):
            counts['s_barrier'] += 1
        elif lower.startswith('s_endpgm'):
            counts['s_endpgm'] += 1
        elif lower.startswith('s_waitcnt'):
            counts['s_waitcnt'] += 1
    return counts


def compile_kernel(kernel_func, constraints, subs, kernel_name):
    """Compile a kernel and return Python ASM and MLIR."""
    options = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        kernel_func.initialize_wave_constraints()
        kernel_func.initialize_symbolic_constraints()
        kernel_func.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(kernel_func, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(enable_debug_info=False)

    # Compile with Python backend
    from wave_lang.kernel.wave.asm.kernel_module_compiler import KernelModuleCompiler
    compiler = KernelModuleCompiler(targetid="gfx942", codeobj="5")
    python_asm = compiler.compile_mlir_string(mlir_text)

    return python_asm, mlir_text


def compare_kernel(name, python_asm, cpp_asm):
    """Compare instruction counts between backends."""
    python_counts = count_instructions(python_asm)
    cpp_counts = count_instructions(cpp_asm)

    # Key instructions that must match
    key_instrs = ['buffer_load', 'buffer_store', 'ds_read', 'ds_write', 'v_mfma', 's_barrier', 's_endpgm']

    all_match = True
    results = []
    for instr in key_instrs:
        py_count = python_counts[instr]
        cpp_count = cpp_counts[instr]
        if py_count > 0 or cpp_count > 0:
            match = py_count == cpp_count
            if not match:
                all_match = False
            results.append((instr, py_count, cpp_count, match))

    return all_match, results, python_counts, cpp_counts


# Define all test kernels
def define_read_write():
    constraints = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16}),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
    ]

    @tkw.wave(constraints)
    def read_write(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    subs = {
        M: 16, N: 16, BLOCK_M: 16, BLOCK_N: 16,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
    }
    return read_write, constraints, subs


def define_mma():
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    subs = {
        M: 16, N: 16, K: 16, BLOCK_M: 16, BLOCK_N: 16,
        LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return mma, constraints, subs


def define_gemm_k_loop():
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_k_loop(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    subs = {
        M: 32, N: 32, K: 32,
        BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16,
        LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return gemm_k_loop, constraints, subs


def define_mma_16x16x32():
    """MMA with 16x16x32 variant."""
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x32_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_16x16x32(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    subs = {
        M: 16, N: 16, K: 32, BLOCK_M: 16, BLOCK_N: 16,
        LOAD_ELEMS_PER_THREAD: 8, STORE_ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return mma_16x16x32, constraints, subs


def define_mma_multi_workgroup():
    """MMA with multi-workgroup (1024x1024x16)."""
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_multi(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    subs = {
        M: 1024, N: 1024, K: 16, BLOCK_M: 16, BLOCK_N: 16,
        LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return mma_multi, constraints, subs


def define_gemm_multi_wave():
    """Multi-wave GEMM with K-loop (BLOCK_K=64)."""
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M // 2),  # 2 waves in M
        tkw.WaveConstraint(N, BLOCK_N // 2),  # 2 waves in N
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_multi_wave(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    subs = {
        M: 64, N: 64, K: 128,
        BLOCK_M: 32, BLOCK_N: 32, BLOCK_K: 64,
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
    }
    return gemm_multi_wave, constraints, subs


def main():
    print("=" * 80)
    print("Comparing Python vs C++ WaveASM Backends for All Kernels")
    print("=" * 80)

    # Define all kernels
    # Note: mma_16x16x32 is excluded because it fails in Python backend
    # with "MFMA 16x16x16 requires 2 VGPRs per operand" - needs proper
    # LOAD_ELEMS_PER_THREAD configuration for 16x16x32 variant.
    kernels = [
        ("read_write", define_read_write),
        ("mma", define_mma),
        # ("mma_16x16x32", define_mma_16x16x32),  # Python backend issue
        ("mma_multi_workgroup", define_mma_multi_workgroup),
        ("gemm_k_loop", define_gemm_k_loop),
        ("gemm_multi_wave", define_gemm_multi_wave),
    ]

    results_summary = []

    for kernel_name, define_func in kernels:
        print(f"\n{'='*80}")
        print(f"TEST: {kernel_name}")
        print("=" * 80)

        try:
            kernel_func, constraints, subs = define_func()

            print("Compiling with Python backend...")
            python_asm, mlir_text = compile_kernel(kernel_func, constraints, subs, kernel_name)

            # Save MLIR for debugging
            Path(f"/tmp/{kernel_name}.mlir").write_text(mlir_text)
            Path(f"/tmp/{kernel_name}_python.s").write_text(python_asm)

            print("Compiling with C++ backend...")
            cpp_asm = compile_with_cpp_backend(mlir_text)

            if cpp_asm.startswith("C++ backend error"):
                print(f"  C++ FAILED: {cpp_asm}")
                results_summary.append((kernel_name, False, cpp_asm))
                continue

            Path(f"/tmp/{kernel_name}_cpp.s").write_text(cpp_asm)

            # Compare
            all_match, instr_results, py_counts, cpp_counts = compare_kernel(
                kernel_name, python_asm, cpp_asm
            )

            print("\nInstruction Comparison:")
            print(f"  {'Instruction':<15} {'Python':>8} {'C++':>8} {'Match':>8}")
            print("  " + "-" * 43)
            for instr, py_count, cpp_count, match in instr_results:
                status = "✓" if match else "✗"
                print(f"  {instr:<15} {py_count:>8} {cpp_count:>8} {status:>8}")

            results_summary.append((kernel_name, all_match, None))

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results_summary.append((kernel_name, False, str(e)))

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    all_passed = True
    for kernel_name, passed, error in results_summary:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {kernel_name}: {status}")
        if error:
            print(f"    Error: {error[:60]}...")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("All kernels match! ✓")
    else:
        print("Some kernels differ. ✗")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
