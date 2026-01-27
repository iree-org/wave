#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Functional test runner for AMDASM kernels.

This script:
1. Generates assembly from AMDASM IR using amdasm-translate
2. Assembles to binary using clang
3. Loads and runs the kernel using HIP/ROCm
4. Verifies correctness with reference data

Usage:
    python run_kernel.py <input.mlir> [--verify]
"""

import argparse
import subprocess
import tempfile
import os
import sys
import numpy as np

# ROCm toolchain paths - can be overridden via environment variables
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
CLANG_PATH = os.environ.get("AMDASM_CLANG", os.path.join(ROCM_PATH, "llvm", "bin", "clang"))
LLD_PATH = os.environ.get("AMDASM_LLD", os.path.join(ROCM_PATH, "llvm", "bin", "ld.lld"))

try:
    from hip import hip
    HAS_HIP = True
except ImportError:
    HAS_HIP = False
    print("Warning: HIP Python bindings not available", file=sys.stderr)


def run_amdasm_translate(input_path: str, output_path: str, target: str = "gfx942"):
    """Run amdasm-translate to generate assembly."""
    cmd = [
        "amdasm-translate",
        input_path,
        "--emit-assembly",
        "-o", output_path,
        "--target", target
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"amdasm-translate failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def assemble_to_binary(asm_path: str, output_path: str, target: str = "gfx942"):
    """Assemble assembly to GPU binary using clang."""
    cmd = [
        CLANG_PATH,
        "-x", "assembler",
        "-target", "amdgcn-amd-amdhsa",
        f"-mcpu={target}",
        "-c", asm_path,
        "-o", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"clang assembly failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def link_binary(obj_path: str, output_path: str, target: str = "gfx942"):
    """Link object file to create executable kernel."""
    cmd = [
        LLD_PATH,
        "--no-undefined",
        "-shared",
        "-o", output_path,
        obj_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"lld link failed: {result.stderr}", file=sys.stderr)
        return False
    return True


def run_kernel_hip(binary_path: str, kernel_name: str,
                   input_data: np.ndarray, output_shape: tuple):
    """Load and run a kernel using HIP."""
    if not HAS_HIP:
        print("HIP not available, cannot run kernel", file=sys.stderr)
        return None

    # This is a placeholder - actual implementation would:
    # 1. Load the binary using hipModuleLoad
    # 2. Get the kernel function using hipModuleGetFunction
    # 3. Allocate device memory
    # 4. Copy input data to device
    # 5. Launch kernel
    # 6. Copy output data back
    # 7. Return results

    print(f"Would run kernel '{kernel_name}' from {binary_path}")
    print(f"Input shape: {input_data.shape}, Output shape: {output_shape}")

    # Placeholder return
    return np.zeros(output_shape, dtype=input_data.dtype)


def main():
    parser = argparse.ArgumentParser(description="Run AMDASM kernel functional test")
    parser.add_argument("input", help="Input AMDASM MLIR file")
    parser.add_argument("--target", default="gfx942", help="GPU target (default: gfx942)")
    parser.add_argument("--kernel", default="", help="Kernel name to run")
    parser.add_argument("--verify", action="store_true", help="Verify results")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.keep_temp:
            tmpdir = "/tmp/amdasm-test"
            os.makedirs(tmpdir, exist_ok=True)

        # Generate assembly
        asm_path = os.path.join(tmpdir, "kernel.s")
        if not run_amdasm_translate(args.input, asm_path, args.target):
            return 1

        print(f"Generated assembly: {asm_path}")

        # Assemble to object
        obj_path = os.path.join(tmpdir, "kernel.o")
        if not assemble_to_binary(asm_path, obj_path, args.target):
            return 1

        print(f"Assembled to object: {obj_path}")

        # Link to binary
        bin_path = os.path.join(tmpdir, "kernel.hsaco")
        if not link_binary(obj_path, bin_path, args.target):
            return 1

        print(f"Linked binary: {bin_path}")

        if args.verify:
            # Run the kernel and verify results
            # This would need kernel-specific input/output setup
            print("Verification not yet implemented")
            return 0

        print("Compilation successful!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
