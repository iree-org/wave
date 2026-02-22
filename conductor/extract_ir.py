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


def capture_kernel_mlir() -> tuple:
    """
    Capture MLIR from a multi-wave GEMM kernel.

    Returns (mlir_text, workgroup_size).
    """
    import wave_lang.kernel.lang as tkl
    import wave_lang.kernel.wave as tkw
    from wave_lang.kernel.lang.global_symbols import (
        GLOBAL_ADDRESS_SPACE,
        SHARED_ADDRESS_SPACE,
    )
    from wave_lang.kernel.wave.compile import WaveCompileOptions
    from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
    from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
    from wave_lang.kernel._support.indexing import IndexingContext
    from wave_lang.kernel.wave.compile import _trace_launchable_and_get_kernel_signature
    from wave_lang.support.ir_imports import Context, Module, func_d
    from wave_lang.kernel.wave.asm.mlir_analysis import (
        walk_ops_recursively,
        should_skip_function,
    )
    from wave_lang.kernel.wave.utils.compile_utils import canonicalize_module

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # 4-wave config: 2x2 waves, 32x32 wave tiles.
    block_m, block_n, wave_m, wave_n = 64, 64, 32, 32
    wave_size = 64
    mma_type = tkw.MMAType.F32_16x16x16_F16

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, wave_m),
        tkw.WaveConstraint(N, wave_n),
        tkw.HardwareConstraint(threads_per_wave=wave_size, mma_type=mma_type),
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

    m, n, k = 256, 256, 256
    block_k = 16

    subs = {
        M: m,
        N: n,
        K: k,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
    }
    subs.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_scheduling_barriers=True,
        backend="asm",
        wave_runtime=True,
        compile_to_mlir=False,
        use_global_to_shared=False,
    )
    options = set_default_run_config(options)

    # Capture MLIR via the same path as the e2e tests.
    with IndexingContext() as idxc:
        idxc.set_subs(options.subs)
        gemm_kernel.initialize_wave_constraints()
        gemm_kernel.initialize_symbolic_constraints()
        gemm_kernel.initialize_workgroup_constraints()

        result = _trace_launchable_and_get_kernel_signature(gemm_kernel, options)
        mb = result[0]

        if options.canonicalize:
            canonicalize_module(mb.module_op)

        full_mlir = mb.module_op.get_asm(enable_debug_info=False)

        launch_info = options.kernel_launch_info
        blocks = launch_info.blocks if launch_info.blocks else [64, 1, 1]

    # Extract func.func from stream wrapper.
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(full_mlir)

        for fn in walk_ops_recursively(module.operation):
            if not isinstance(fn, func_d.FuncOp):
                continue
            if should_skip_function(fn):
                continue
            func_text = fn.get_asm(print_generic_op_form=True)
            mlir_text = "module {\n" + func_text + "\n}\n"
            return mlir_text, tuple(blocks)

    raise ValueError("No kernel function found in MLIR")


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
    args = parser.parse_args()

    print("Capturing kernel MLIR...", file=sys.stderr)
    mlir_text, wg_size = capture_kernel_mlir()
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
