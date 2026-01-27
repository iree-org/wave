#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Dump the MLIR IR that feeds into the Python ASM backend.

This script compiles Wave DSL kernels and dumps:
1. The MLIR that goes into the ASM backend
2. The generated assembly output

Usage:
    python dump_mlir.py [--kernel copy|mma|gemm] [--asm]
"""

import argparse
import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.support.location_config import LocationCaptureConfig, LocationCaptureLevel


# ============================================================================
# Kernel Definitions
# ============================================================================

def get_copy_kernel():
    """Simple copy kernel - reads from A, writes to B."""
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints = [
        tkw.HardwareConstraint(threads_per_wave=wave_size, vector_shapes={M: BLOCK_M, N: BLOCK_N}),
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

    subs = {
        M: 16,
        N: 16,
        ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
    }
    return copy_kernel, subs, "copy_kernel"


def get_mma_kernel():
    """MMA kernel - C = A @ B^T."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    wave_size = 64
    BLOCK_M = 16
    BLOCK_N = 16

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def mma_kernel(
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
        M: 16,
        N: 16,
        K: 16,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return mma_kernel, subs, "mma_kernel"


def get_gemm_kernel():
    """GEMM kernel with K-loop - C = A @ B^T."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    wave_size = 64
    WAVE_M = 16
    WAVE_N = 16

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, WAVE_M),
        tkw.WaveConstraint(N, WAVE_N),
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        ),
    ]

    @tkw.wave(constraints)
    def gemm_kernel(
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

        tkw.write(repeat, c)

    subs = {
        M: 64,
        N: 64,
        K: 64,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    return gemm_kernel, subs, "gemm_kernel"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dump MLIR/ASM from Wave kernels")
    parser.add_argument("--kernel", choices=["copy", "mma", "gemm"], default="copy",
                        help="Kernel to compile")
    parser.add_argument("--asm", action="store_true",
                        help="Also generate and print assembly")
    parser.add_argument("--mlir-only", action="store_true",
                        help="Only print MLIR, no assembly")
    args = parser.parse_args()

    # Get kernel
    if args.kernel == "copy":
        kernel, subs, name = get_copy_kernel()
    elif args.kernel == "mma":
        kernel, subs, name = get_mma_kernel()
    elif args.kernel == "gemm":
        kernel, subs, name = get_gemm_kernel()

    print(f"=== {name} ===\n")

    # Capture the MLIR that feeds into the ASM backend by monkeypatching
    import wave_lang.kernel.wave.compile as compile_module

    captured_mlir = []
    original_generate_asm = compile_module._generate_asm_code

    def patched_generate_asm(mb, options):
        mlir_text = mb.module_op.get_asm()
        captured_mlir.append(mlir_text)
        return original_generate_asm(mb, options)

    compile_module._generate_asm_code = patched_generate_asm

    options = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_asm=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options = set_default_run_config(options)

    result = wave_compile(options, kernel)

    # Restore original function
    compile_module._generate_asm_code = original_generate_asm

    if captured_mlir:
        print("=== MLIR IR (input to ASM backend) ===")
        print(captured_mlir[0])
        print()

    if args.asm and not args.mlir_only:
        print("=== Generated Assembly ===")
        print(result.asm)


if __name__ == "__main__":
    main()
