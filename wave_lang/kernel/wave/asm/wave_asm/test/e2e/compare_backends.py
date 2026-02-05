#!/usr/bin/env python3
# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unified tool to compare Python and C++ WaveASM backend assembly outputs.

This script consolidates multiple comparison tools into a single unified interface:
- copy: Simple copy kernel (buffer load/store)
- mma: Single MMA operation
- gemm: GEMM with K-loop
- all: Run all kernel comparisons

Usage:
    python compare_backends.py copy          # Compare copy kernel
    python compare_backends.py mma           # Compare MMA kernel
    python compare_backends.py gemm          # Compare GEMM with K-loop
    python compare_backends.py gemm --g2s    # GEMM with global_to_shared
    python compare_backends.py all           # Run all comparisons
    python compare_backends.py all -v        # Verbose output
"""

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Add wave_lang to path
wave_root = Path(__file__).parent.parent.parent.parent
if str(wave_root) not in sys.path:
    sys.path.insert(0, str(wave_root))

from wave_lang.kernel.wave.asm.utils import extract_func_from_stream_mlir


# =============================================================================
# Common Utilities
# =============================================================================


def get_target_arch() -> str:
    """Get target architecture from environment or detect."""
    if "WAVE_DEFAULT_ARCH" in os.environ:
        arch = os.environ["WAVE_DEFAULT_ARCH"]
        return arch.split(":")[0]

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, "gcnArchName"):
                return props.gcnArchName.split(":")[0]
    except Exception:
        pass

    return "gfx950"


def get_waveasm_translate_path() -> Path:
    """Get path to waveasm-translate executable."""
    script_dir = Path(__file__).parent
    default_path = (
        script_dir.parent.parent
        / "build"
        / "tools"
        / "waveasm-translate"
        / "waveasm-translate"
    )
    if default_path.exists():
        return default_path
    raise FileNotFoundError(f"waveasm-translate not found at {default_path}")


def compile_with_python_backend(mlir_text: str, target: str) -> str:
    """Compile MLIR using Python backend."""
    try:
        from wave_lang.kernel.wave.asm.kernel_module_compiler import (
            KernelModuleCompiler,
        )

        compiler = KernelModuleCompiler(targetid=target, codeobj="5")
        return compiler.compile_mlir_string(mlir_text)
    except Exception as e:
        import traceback

        return f"Python backend error: {e}\n{traceback.format_exc()}"


def compile_with_cpp_backend(
    mlir_text: str, target: str = "gfx942", wg_size: tuple[int, int, int] | None = None
) -> str:
    """Compile MLIR using C++ backend via waveasm-translate."""
    waveasm_translate = get_waveasm_translate_path()

    # Extract func from stream wrapper
    try:
        simplified_mlir = extract_func_from_stream_mlir(mlir_text)
    except Exception as e:
        return f"C++ backend error: Failed to extract func: {e}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(simplified_mlir)
        mlir_file = f.name

    try:
        cmd = [
            str(waveasm_translate),
            mlir_file,
            f"--target={target}",
            "--waveasm-scoped-cse",
            "--waveasm-peephole",
            "--waveasm-linear-scan",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
        ]
        # Add workgroup size if specified
        if wg_size:
            cmd.extend(
                [
                    f"--workgroup-size-x={wg_size[0]}",
                    f"--workgroup-size-y={wg_size[1]}",
                    f"--workgroup-size-z={wg_size[2]}",
                ]
            )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return f"C++ backend error: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "C++ backend error: Compilation timed out"
    except Exception as e:
        return f"C++ backend error: {e}"
    finally:
        Path(mlir_file).unlink(missing_ok=True)


# =============================================================================
# Instruction Counting & Comparison
# =============================================================================


def is_label_line(line: str) -> bool:
    """Check if line is a label (e.g., 'foo:', 'L_loop:')."""
    stripped = line.strip()
    if stripped.endswith(":"):
        label_name = stripped[:-1]
        return label_name.replace("_", "").replace(".", "").isalnum()
    if ":" in stripped:
        before_colon = stripped.split(":")[0]
        if before_colon.replace("_", "").replace(
            ".", ""
        ).isalnum() and not before_colon.startswith(("s[", "v[")):
            return True
    return False


def count_total_instructions(asm: str) -> int:
    """Count total instructions in assembly."""
    count = 0
    in_kernel = False
    for line in asm.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith(".")
            or stripped.startswith("#")
            or stripped.startswith("//")
            or stripped.startswith(";")
        ):
            continue
        if is_label_line(stripped):
            in_kernel = True
            continue
        if in_kernel and stripped and not stripped.startswith("."):
            count += 1
    return count


def extract_instruction_counts(asm: str) -> dict[str, int]:
    """Extract counts of each instruction mnemonic."""
    counts: dict[str, int] = {}
    in_kernel = False
    for line in asm.split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(".")
            or stripped.startswith("#")
            or stripped.startswith("//")
            or stripped.startswith(";")
        ):
            continue
        if is_label_line(stripped):
            in_kernel = True
            continue
        if not in_kernel:
            continue
        parts = stripped.split()
        if parts:
            mnemonic = parts[0]
            counts[mnemonic] = counts.get(mnemonic, 0) + 1
    return counts


def count_key_instructions(asm: str) -> dict[str, int]:
    """Count key instruction patterns."""
    counts = {
        "buffer_load": 0,
        "buffer_store": 0,
        "buffer_load_lds": 0,
        "ds_read": 0,
        "ds_write": 0,
        "v_mfma": 0,
        "s_barrier": 0,
        "s_endpgm": 0,
        "s_waitcnt": 0,
    }
    for line in asm.split("\n"):
        lower = line.lower().strip()
        if lower.startswith("buffer_load") and "lds" in lower:
            counts["buffer_load_lds"] += 1
        elif lower.startswith("buffer_load"):
            counts["buffer_load"] += 1
        elif lower.startswith("buffer_store"):
            counts["buffer_store"] += 1
        elif lower.startswith("ds_read"):
            counts["ds_read"] += 1
        elif lower.startswith("ds_write"):
            counts["ds_write"] += 1
        elif "v_mfma" in lower:
            counts["v_mfma"] += 1
        elif lower.startswith("s_barrier"):
            counts["s_barrier"] += 1
        elif lower.startswith("s_endpgm"):
            counts["s_endpgm"] += 1
        elif lower.startswith("s_waitcnt"):
            counts["s_waitcnt"] += 1
    return counts


@dataclass
class ComparisonResult:
    """Result of comparing two backend outputs."""

    kernel_name: str
    python_asm: str
    cpp_asm: str
    python_total: int = 0
    cpp_total: int = 0
    python_counts: dict[str, int] = field(default_factory=dict)
    cpp_counts: dict[str, int] = field(default_factory=dict)
    python_key_counts: dict[str, int] = field(default_factory=dict)
    cpp_key_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def compute_stats(self):
        """Compute instruction statistics."""
        if "error" in self.python_asm.lower():
            self.error = self.python_asm
            return
        if "error" in self.cpp_asm.lower():
            self.error = self.cpp_asm
            return

        self.python_total = count_total_instructions(self.python_asm)
        self.cpp_total = count_total_instructions(self.cpp_asm)
        self.python_counts = extract_instruction_counts(self.python_asm)
        self.cpp_counts = extract_instruction_counts(self.cpp_asm)
        self.python_key_counts = count_key_instructions(self.python_asm)
        self.cpp_key_counts = count_key_instructions(self.cpp_asm)

    def key_metrics_match(self) -> bool:
        """Check if key metrics (mfma, loads, stores) match."""
        key_patterns = ["v_mfma", "buffer_load", "buffer_store", "ds_read", "ds_write"]
        for pattern in key_patterns:
            if self.python_key_counts.get(pattern, 0) != self.cpp_key_counts.get(
                pattern, 0
            ):
                return False
        return True


def print_comparison(result: ComparisonResult, verbose: bool = False):
    """Print comparison results."""
    print(f"\n{'='*80}")
    print(f"KERNEL: {result.kernel_name}")
    print("=" * 80)

    if result.error:
        print(f"ERROR: {result.error}")
        return

    print(f"\nTotal Instructions: Python={result.python_total}, C++={result.cpp_total}")
    diff = result.cpp_total - result.python_total
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    print(f"Difference: {diff_str} ({diff_str} in C++)")

    # Key metrics
    print(f"\n{'KEY METRICS':-^60}")
    print(f"{'Instruction':<25} {'Python':>10} {'C++':>10} {'Match':>10}")
    print("-" * 55)

    key_patterns = [
        "v_mfma",
        "buffer_load",
        "buffer_store",
        "buffer_load_lds",
        "ds_read",
        "ds_write",
        "s_barrier",
        "s_waitcnt",
        "s_endpgm",
    ]
    for pattern in key_patterns:
        py = result.python_key_counts.get(pattern, 0)
        cpp = result.cpp_key_counts.get(pattern, 0)
        if py > 0 or cpp > 0:
            match = "✓" if py == cpp else "✗"
            print(f"{pattern:<25} {py:>10} {cpp:>10} {match:>10}")

    if verbose:
        # Full instruction breakdown
        print(f"\n{'FULL INSTRUCTION BREAKDOWN':-^60}")
        all_mnemonics = sorted(
            set(result.python_counts.keys()) | set(result.cpp_counts.keys())
        )
        print(f"{'Instruction':<40} {'Python':>10} {'C++':>10}")
        print("-" * 60)
        for mnemonic in all_mnemonics:
            py = result.python_counts.get(mnemonic, 0)
            cpp = result.cpp_counts.get(mnemonic, 0)
            print(f"{mnemonic:<40} {py:>10} {cpp:>10}")

    status = "✓ PASS" if result.key_metrics_match() else "✗ FAIL"
    print(f"\nKey Metrics Match: {status}")


# =============================================================================
# Kernel Definitions
# =============================================================================


def capture_copy_kernel(
    target: str, use_g2s: bool = False
) -> tuple[str, str, tuple[int, int, int] | None]:
    """Define a copy kernel and capture its MLIR."""
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import (
        WaveCompileOptions,
        _trace_launchable_and_get_kernel_signature,
    )
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel._support.indexing import IndexingContext

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints = [
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16}),
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
        use_global_to_shared=use_g2s,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        copy_kernel.initialize_wave_constraints()
        copy_kernel.initialize_symbolic_constraints()
        copy_kernel.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(copy_kernel, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(enable_debug_info=False)

    return mlir_text, "copy", None


def capture_mma_kernel(
    target: str, use_g2s: bool = False
) -> tuple[str, str, tuple[int, int, int] | None]:
    """Define an MMA kernel and capture its MLIR."""
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import (
        WaveCompileOptions,
        _trace_launchable_and_get_kernel_signature,
    )
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel._support.indexing import IndexingContext

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    SHARED = tkl.AddressSpace.SHARED_MEMORY.value
    GLOBAL = tkl.AddressSpace.GLOBAL_MEMORY.value

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.HardwareConstraint(
            threads_per_wave=64, mma_type=tkw.MMAType.F32_16x16x16_F16
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

    options = WaveCompileOptions(
        subs={
            M: 16,
            N: 16,
            K: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED,
            ADDRESS_SPACE_0: GLOBAL,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_g2s,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        mma_kernel.initialize_wave_constraints()
        mma_kernel.initialize_symbolic_constraints()
        mma_kernel.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(mma_kernel, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(enable_debug_info=False)

    return mlir_text, "mma", None


def capture_gemm_kernel(
    target: str, use_g2s: bool = False
) -> tuple[str, str, tuple[int, int, int] | None]:
    """Define a GEMM kernel with K-loop and capture its MLIR."""
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.wave.compile import (
        WaveCompileOptions,
        _trace_launchable_and_get_kernel_signature,
    )
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel._support.indexing import IndexingContext

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

    SHARED = tkl.AddressSpace.SHARED_MEMORY.value
    GLOBAL = tkl.AddressSpace.GLOBAL_MEMORY.value

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M // 2),  # 2 waves in M
        tkw.WaveConstraint(N, BLOCK_N // 2),  # 2 waves in N
        tkw.HardwareConstraint(
            threads_per_wave=64, mma_type=tkw.MMAType.F32_16x16x16_F16
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
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # 64x64x128 problem with 32x32 blocks, BLOCK_K=64 -> 2 K-loop iterations
    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 128,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED,
            ADDRESS_SPACE_0: GLOBAL,
        },
        canonicalize=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=use_g2s,
    )
    options = set_default_run_config(options)

    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        gemm_kernel.initialize_wave_constraints()
        gemm_kernel.initialize_symbolic_constraints()
        gemm_kernel.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(gemm_kernel, options)
        mb = result[0]

        mlir_text = mb.module_op.get_asm(enable_debug_info=False)

        # Extract workgroup size
        hw_constraint = gemm_kernel.hardware_constraints[0]
        threads_per_block = hw_constraint.threads_per_block
        if len(threads_per_block) == 2:
            wg_size = (threads_per_block[0], threads_per_block[1], 1)
        elif len(threads_per_block) == 1:
            wg_size = (threads_per_block[0], 1, 1)
        else:
            wg_size = tuple(threads_per_block[:3])

    return mlir_text, "gemm", wg_size


# =============================================================================
# Main Comparison Logic
# =============================================================================


def run_comparison(
    kernel_name: str,
    capture_func: Callable,
    target: str,
    use_g2s: bool = False,
    save_outputs: bool = True,
) -> ComparisonResult:
    """Run comparison for a single kernel."""
    print(f"Capturing MLIR for {kernel_name}...")
    mlir_text, name, wg_size = capture_func(target, use_g2s)

    print(f"Compiling with Python backend...")
    python_asm = compile_with_python_backend(mlir_text, target)

    print(f"Compiling with C++ backend...")
    cpp_asm = compile_with_cpp_backend(mlir_text, target, wg_size)

    result = ComparisonResult(
        kernel_name=kernel_name, python_asm=python_asm, cpp_asm=cpp_asm
    )
    result.compute_stats()

    if save_outputs:
        Path(f"/tmp/{kernel_name}_python.s").write_text(python_asm)
        Path(f"/tmp/{kernel_name}_cpp.s").write_text(cpp_asm)
        try:
            simplified_mlir = extract_func_from_stream_mlir(mlir_text)
            Path(f"/tmp/{kernel_name}.mlir").write_text(simplified_mlir)
        except Exception:
            pass
        print(f"Outputs saved to /tmp/{kernel_name}_*.s")

    return result


# =============================================================================
# CLI Entry Points
# =============================================================================


def cmd_copy(args):
    """Compare copy kernel."""
    target = get_target_arch()
    print(f"Target: {target}, use_g2s: {args.g2s}")

    result = run_comparison("copy", capture_copy_kernel, target, args.g2s)
    print_comparison(result, args.verbose)
    return 0 if result.key_metrics_match() else 1


def cmd_mma(args):
    """Compare MMA kernel."""
    target = get_target_arch()
    print(f"Target: {target}, use_g2s: {args.g2s}")

    result = run_comparison("mma", capture_mma_kernel, target, args.g2s)
    print_comparison(result, args.verbose)
    return 0 if result.key_metrics_match() else 1


def cmd_gemm(args):
    """Compare GEMM kernel."""
    target = get_target_arch()
    print(f"Target: {target}, use_g2s: {args.g2s}")
    print("GEMM config: 64x64x128, 32x32 blocks, BLOCK_K=64, 4 waves/WG, 2 K-iters")

    result = run_comparison("gemm", capture_gemm_kernel, target, args.g2s)
    print_comparison(result, args.verbose)
    return 0 if result.key_metrics_match() else 1


def cmd_all(args):
    """Run all kernel comparisons."""
    target = get_target_arch()
    print(f"Target: {target}, use_g2s: {args.g2s}")

    kernels = [
        ("copy", capture_copy_kernel),
        ("mma", capture_mma_kernel),
        ("gemm", capture_gemm_kernel),
    ]

    results = []
    for name, capture_func in kernels:
        try:
            result = run_comparison(name, capture_func, target, args.g2s)
            print_comparison(result, args.verbose)
            results.append(result)
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    all_pass = True
    for r in results:
        status = "✓" if r.key_metrics_match() and not r.error else "✗"
        if not r.key_metrics_match() or r.error:
            all_pass = False
        print(
            f"  {r.kernel_name}: {status} (Python={r.python_total}, C++={r.cpp_total})"
        )

    print(f"\nAll kernels pass: {'✓' if all_pass else '✗'}")
    return 0 if all_pass else 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python and C++ WaveASM backend outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_backends.py copy          # Compare copy kernel
  python compare_backends.py mma           # Compare MMA kernel
  python compare_backends.py gemm          # Compare GEMM with K-loop
  python compare_backends.py gemm --g2s    # GEMM with global_to_shared
  python compare_backends.py all -v        # All kernels with verbose output
""",
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full instruction breakdown"
    )
    parser.add_argument(
        "--g2s", action="store_true", help="Enable global_to_shared optimization"
    )

    subparsers = parser.add_subparsers(dest="command", help="Kernel to compare")

    # Subcommands
    subparsers.add_parser("copy", help="Compare copy kernel")
    subparsers.add_parser("mma", help="Compare MMA kernel")
    subparsers.add_parser("gemm", help="Compare GEMM with K-loop")
    subparsers.add_parser("all", help="Run all comparisons")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "copy": cmd_copy,
        "mma": cmd_mma,
        "gemm": cmd_gemm,
        "all": cmd_all,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
