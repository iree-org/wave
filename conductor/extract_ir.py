#!/usr/bin/env python3
"""
Extract WaveASM MLIR IR at the Conductor scheduling stage.

Defines a GEMM kernel inline, runs the Wave compilation pipeline to produce
input MLIR, then runs waveasm-translate with only the pre-scheduling passes
(CSE, peephole, memory-offset-opt). The output is the WaveASM IR that the
Conductor would see — post-optimization, pre-register-allocation.

Usage:
    python -m conductor.extract_ir -o /tmp/conductor_ir.mlir
    python -m conductor.extract_ir --metrics
    python -m conductor.extract_ir --dump-input-mlir

Requires:
    - waveasm-translate binary (auto-detected from build/)
    - WAVE_CACHE_ON=0 recommended
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add wave_lang to path.
wave_root = Path(__file__).parent.parent
sys.path.insert(0, str(wave_root))


def find_waveasm_translate() -> str:
    """Find the waveasm-translate binary."""
    env_path = os.environ.get("WAVEASM_TRANSLATE")
    if env_path and Path(env_path).exists():
        return env_path

    candidates = [
        wave_root
        / "wave_lang"
        / "kernel"
        / "wave"
        / "asm"
        / "wave_asm"
        / "build"
        / "tools"
        / "waveasm-translate"
        / "waveasm-translate",
        wave_root
        / "wave_lang"
        / "kernel"
        / "wave"
        / "asm"
        / "wave_asm"
        / "build"
        / "bin"
        / "waveasm-translate",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "waveasm-translate not found. Set WAVEASM_TRANSLATE env var or build it."
    )


def get_target() -> str:
    """Get the target architecture."""
    return os.environ.get("WAVE_DEFAULT_ARCH", "gfx942")


def capture_mxfp4_kernel_mlir() -> tuple:
    """Capture MLIR from a double-buffered 4-wave MXFP4 GEMM kernel.

    Returns (mlir_text, workgroup_size).
    """
    from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
    from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

    # Same config as test_dbuf_4wave_mxfp4_gemm_cpp_backend.
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape=(1024, 1024, 8192),
        block_shape=(256, 256, 256),
        wave_shape=(1, 4),
    )
    schedule = get_mxfp4_asymmetric_schedule()

    options.backend = "asm"
    options.wave_runtime = True
    options.compile_to_mlir = False
    options = set_default_run_config(options)

    # Reuse the test helper that handles opsel + canonicalize.
    sys.path.insert(
        0,
        str(
            wave_root
            / "wave_lang"
            / "kernel"
            / "wave"
            / "asm"
            / "wave_asm"
            / "test"
            / "e2e"
        ),
    )
    from waveasm_e2e import capture_wave_kernel_info

    info = capture_wave_kernel_info(options, gemm, schedule=schedule)
    return info.mlir_text, info.workgroup_size


def capture_kernel_mlir() -> tuple:
    """Capture MLIR from a manually-scheduled 8-wave GEMM kernel.

    Uses get_tagged_gemm + get_two_pp_cluster_schedule for a 2-stage
    pipelined prefetch with cluster reordering and wave staggering.

    Returns (mlir_text, workgroup_size).
    """
    from wave_lang.kernel.wave.schedules import get_two_pp_cluster_schedule
    from wave_lang.kernel.wave.schedules.gemm_two_pp_cluster import get_tagged_gemm
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

    gemm, options = get_tagged_gemm(
        shape=(4096, 4096, 4096),
        block_shape=(128, 256, 64),
    )
    schedule = get_two_pp_cluster_schedule()

    options.backend = "asm"
    options.wave_runtime = True
    options.compile_to_mlir = False
    options = set_default_run_config(options)

    sys.path.insert(
        0,
        str(
            wave_root
            / "wave_lang"
            / "kernel"
            / "wave"
            / "asm"
            / "wave_asm"
            / "test"
            / "e2e"
        ),
    )
    from waveasm_e2e import capture_wave_kernel_info

    info = capture_wave_kernel_info(options, gemm, schedule=schedule)
    return info.mlir_text, info.workgroup_size


def run_waveasm_translate(
    mlir_text: str, workgroup_size: tuple, extra_flags: list = None
) -> tuple:
    """
    Run waveasm-translate with given flags.

    Returns (stdout, stderr, returncode).
    """
    translate = find_waveasm_translate()
    target = get_target()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(mlir_text)
        input_path = f.name

    try:
        cmd = [
            translate,
            f"--target={target}",
            f"--workgroup-size-x={workgroup_size[0]}",
            f"--workgroup-size-y={workgroup_size[1]}",
            f"--workgroup-size-z={workgroup_size[2]}",
        ]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.append(input_path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(input_path)


def run_pre_scheduling_pipeline(mlir_text: str, workgroup_size: tuple) -> str:
    """
    Run waveasm-translate with only pre-scheduling passes.

    Produces WaveASM IR after:
      TranslateFromMLIR -> ScopedCSE -> Peephole -> MemoryOffsetOpt -> Canonicalizer -> ScopedCSE

    But before:
      LinearScan, InsertWaitcnt, HazardMitigation, EmitAssembly
    """
    flags = [
        "--mlir-cse",
        "--waveasm-scoped-cse",
        "--waveasm-peephole",
        "--waveasm-memory-offset-opt",
        # Stop here — no regalloc, no waitcnt, no hazard, no assembly.
    ]
    stdout, stderr, rc = run_waveasm_translate(mlir_text, workgroup_size, flags)
    if rc != 0:
        print(f"waveasm-translate (pre-scheduling) failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)
    return stdout


def run_full_pipeline(mlir_text: str, workgroup_size: tuple) -> str:
    """Run the full pipeline and return assembly text."""
    flags = [
        "--mlir-cse",
        "--waveasm-scoped-cse",
        "--waveasm-peephole",
        "--waveasm-memory-offset-opt",
        "--waveasm-linear-scan",
        "--max-vgprs=512",
        "--max-agprs=512",
        "--waveasm-insert-waitcnt",
        "--waveasm-hazard-mitigation",
        "--emit-assembly",
    ]
    stdout, stderr, rc = run_waveasm_translate(mlir_text, workgroup_size, flags)
    if rc != 0:
        print(f"Full pipeline failed:\n{stderr}", file=sys.stderr)
        return ""
    return stdout


def count_asm_metrics(asm_text: str) -> dict:
    """Extract basic metrics from assembly text."""
    lines = asm_text.split("\n")
    metrics = {
        "total_instructions": 0,
        "s_waitcnt": 0,
        "s_nop": 0,
        "mfma": 0,
        "buffer_load": 0,
        "ds_read": 0,
        "ds_write": 0,
    }
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", ";", ".")):
            continue
        if stripped.endswith(":"):
            continue
        metrics["total_instructions"] += 1
        if "s_waitcnt" in stripped:
            metrics["s_waitcnt"] += 1
        if "s_nop" in stripped:
            metrics["s_nop"] += 1
        if "mfma" in stripped:
            metrics["mfma"] += 1
        if "buffer_load" in stripped:
            metrics["buffer_load"] += 1
        if "ds_read" in stripped:
            metrics["ds_read"] += 1
        if "ds_write" in stripped:
            metrics["ds_write"] += 1

    # Extract register counts from kernel descriptor.
    for line in lines:
        if ".amdhsa_next_free_vgpr" in line:
            metrics["peak_vgpr"] = int(line.split()[-1])
        if ".amdhsa_next_free_sgpr" in line:
            metrics["peak_sgpr"] = int(line.split()[-1])

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Extract WaveASM IR at the Conductor scheduling stage."
    )
    parser.add_argument(
        "--dump-input-mlir",
        action="store_true",
        help="Also dump the input MLIR (before waveasm-translate).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write WaveASM IR to file instead of stdout.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Also run full pipeline and print baseline metrics.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="gemm",
        choices=["gemm", "mxfp4"],
        help="Kernel to capture (default: gemm).",
    )
    args = parser.parse_args()

    capture_fn = (
        capture_mxfp4_kernel_mlir if args.kernel == "mxfp4" else capture_kernel_mlir
    )
    print(f"Capturing {args.kernel} kernel MLIR...", file=sys.stderr)
    mlir_text, wg_size = capture_fn()
    print(f"  workgroup_size: {wg_size}", file=sys.stderr)
    print(f"  input MLIR: {len(mlir_text)} chars", file=sys.stderr)

    if args.dump_input_mlir:
        print("=== Input MLIR (before waveasm-translate) ===")
        print(mlir_text)
        print("=== End Input MLIR ===\n")

    print("Running pre-scheduling pipeline...", file=sys.stderr)
    waveasm_ir = run_pre_scheduling_pipeline(mlir_text, wg_size)
    print(f"  WaveASM IR: {len(waveasm_ir)} chars", file=sys.stderr)

    op_count = sum(
        1
        for line in waveasm_ir.split("\n")
        if "waveasm." in line and not line.strip().startswith("//")
    )
    print(f"  WaveASM ops: {op_count}", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(waveasm_ir)
        print(f"  Written to: {args.output}", file=sys.stderr)
    else:
        print(waveasm_ir)

    if args.metrics:
        print("\nRunning full pipeline for baseline metrics...", file=sys.stderr)
        asm_text = run_full_pipeline(mlir_text, wg_size)
        if asm_text:
            metrics = count_asm_metrics(asm_text)
            print("\n=== Baseline Metrics ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
