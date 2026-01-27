#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Compare Python and C++ WaveASM backend outputs for all kernels in asm_backend_test.py.

Uses compile_to_asm=True to get Python backend assembly, then compiles the same
MLIR with the C++ backend and compares the outputs.

Run with:
    python test/e2e/compare_all_kernels.py
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
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel._support.indexing import IndexingContext

# Symbols
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
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


def compile_with_cpp_backend(mlir_text: str, target: str = "gfx942") -> str:
    """Compile MLIR using C++ backend via waveasm-translate."""
    waveasm_translate = get_waveasm_translate_path()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(mlir_text)
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


def extract_func_from_stream(mlir_text: str) -> str:
    """Extract func.func from stream.executable wrapper using Python MLIR bindings."""
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


def compare_assembly(python_asm: str, cpp_asm: str, kernel_name: str) -> dict:
    """Compare assembly outputs and return stats."""
    def count_instructions(asm: str) -> dict:
        counts = {}
        patterns = ['buffer_load', 'buffer_store', 's_endpgm', 's_waitcnt',
                    'v_mfma', 'ds_read', 'ds_write', 's_barrier']
        for pattern in patterns:
            counts[pattern] = sum(1 for line in asm.split('\n')
                                  if pattern in line.lower())
        return counts

    python_counts = count_instructions(python_asm)
    cpp_counts = count_instructions(cpp_asm)

    matches = all(python_counts[k] == cpp_counts[k] for k in python_counts)

    return {
        'kernel': kernel_name,
        'matches': matches,
        'python_counts': python_counts,
        'cpp_counts': cpp_counts,
    }


def test_read_write():
    """Test copy kernel (read_write)."""
    print("\n" + "="*80)
    print("TEST: read_write (copy kernel)")
    print("="*80)

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

    options = WaveCompileOptions(
        subs={
            M: 16, N: 16, BLOCK_M: 16, BLOCK_N: 16,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        compile_to_asm=True,
    )

    compiled = wave_compile(options, read_write)
    python_asm = compiled.asm

    # Get MLIR and compile with C++
    mlir_text = compiled.module_str
    simplified_mlir = extract_func_from_stream(mlir_text)
    cpp_asm = compile_with_cpp_backend(simplified_mlir)

    # Save for inspection
    Path("/tmp/read_write_python.s").write_text(python_asm)
    Path("/tmp/read_write_cpp.s").write_text(cpp_asm)
    Path("/tmp/read_write.mlir").write_text(simplified_mlir)

    result = compare_assembly(python_asm, cpp_asm, "read_write")
    print(f"Python counts: {result['python_counts']}")
    print(f"C++ counts:    {result['cpp_counts']}")
    print(f"Match: {'✓' if result['matches'] else '✗'}")

    return result


def test_mma():
    """Test MMA kernel."""
    print("\n" + "="*80)
    print("TEST: mma (16x16x16)")
    print("="*80)

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

    options = WaveCompileOptions(
        subs={
            M: 16, N: 16, K: 16, BLOCK_M: 16, BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        compile_to_asm=True,
    )

    compiled = wave_compile(options, mma)
    python_asm = compiled.asm

    mlir_text = compiled.module_str
    simplified_mlir = extract_func_from_stream(mlir_text)
    cpp_asm = compile_with_cpp_backend(simplified_mlir)

    Path("/tmp/mma_python.s").write_text(python_asm)
    Path("/tmp/mma_cpp.s").write_text(cpp_asm)
    Path("/tmp/mma.mlir").write_text(simplified_mlir)

    result = compare_assembly(python_asm, cpp_asm, "mma")
    print(f"Python counts: {result['python_counts']}")
    print(f"C++ counts:    {result['cpp_counts']}")
    print(f"Match: {'✓' if result['matches'] else '✗'}")

    return result


def test_gemm_k_loop():
    """Test GEMM with K-loop."""
    print("\n" + "="*80)
    print("TEST: gemm_k_loop")
    print("="*80)

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

    options = WaveCompileOptions(
        subs={
            M: 32, N: 32, K: 32,
            BLOCK_M: 16, BLOCK_N: 16, BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4, STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
        compile_to_asm=True,
    )

    compiled = wave_compile(options, gemm_k_loop)
    python_asm = compiled.asm

    mlir_text = compiled.module_str
    simplified_mlir = extract_func_from_stream(mlir_text)
    cpp_asm = compile_with_cpp_backend(simplified_mlir)

    Path("/tmp/gemm_k_loop_python.s").write_text(python_asm)
    Path("/tmp/gemm_k_loop_cpp.s").write_text(cpp_asm)
    Path("/tmp/gemm_k_loop.mlir").write_text(simplified_mlir)

    result = compare_assembly(python_asm, cpp_asm, "gemm_k_loop")
    print(f"Python counts: {result['python_counts']}")
    print(f"C++ counts:    {result['cpp_counts']}")
    print(f"Match: {'✓' if result['matches'] else '✗'}")

    return result


def main():
    print("Comparing Python vs C++ WaveASM backends")
    print("="*80)

    results = []

    try:
        results.append(test_read_write())
    except Exception as e:
        print(f"read_write FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results.append(test_mma())
    except Exception as e:
        print(f"mma FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        results.append(test_gemm_k_loop())
    except Exception as e:
        print(f"gemm_k_loop FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        status = "✓" if r['matches'] else "✗"
        print(f"  {r['kernel']}: {status}")

    all_match = all(r['matches'] for r in results)
    print(f"\nAll kernels match: {'✓' if all_match else '✗'}")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
