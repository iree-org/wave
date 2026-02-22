#!/usr/bin/env python3
"""
Conductor: Python driver for WaveASM instruction scheduling experiments.

Ties together extract_ir → tag → apply moves → post-scheduling pipeline → metrics.

Usage:
    # Baseline (no moves):
    python -m conductor.conductor --metrics

    # Apply specific moves:
    python -m conductor.conductor \
        --moves "swap buffer_load_dwordx4_0 v_mfma_f32_16x16x16_f16_0" --metrics

    # Show tagged IR only:
    python -m conductor.conductor --tag-only

    # Read moves from a file:
    python -m conductor.conductor --moves-file moves.txt --metrics
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

from conductor.extract_ir import (
    run_waveasm_translate,
    count_asm_metrics,
    capture_kernel_mlir,
    run_pre_scheduling_pipeline,
)


def find_waveasm_conductor() -> str:
    """Find the waveasm-conductor binary."""
    env_path = os.environ.get("WAVEASM_CONDUCTOR")
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
        / "waveasm-conductor"
        / "waveasm-conductor",
        wave_root
        / "wave_lang"
        / "kernel"
        / "wave"
        / "asm"
        / "wave_asm"
        / "build"
        / "bin"
        / "waveasm-conductor",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "waveasm-conductor not found. Set WAVEASM_CONDUCTOR env var or build it."
    )


class Conductor:
    """Encapsulates the full round-trip for instruction scheduling experiments."""

    def __init__(self, waveasm_ir: str, workgroup_size: tuple, target: str = "gfx942"):
        self.waveasm_ir = waveasm_ir
        self.workgroup_size = workgroup_size
        self.target = target
        self._baseline_cache = None

    def tag(self) -> str:
        """Run tag-instructions on the IR. Return tagged IR text."""
        flags = [
            "--waveasm-tag-instructions",
            "--print-debug-locs-inline",
        ]
        stdout, stderr, rc = run_waveasm_translate(
            self.waveasm_ir, self.workgroup_size, flags
        )
        if rc != 0:
            raise RuntimeError(f"tag-instructions failed:\n{stderr}")
        return stdout

    def apply_moves(self, tagged_ir: str, commands: list) -> str:
        """Apply CONDUCTOR move commands to tagged IR. Return reordered IR."""
        conductor = find_waveasm_conductor()

        # Prepend CONDUCTOR commands to the tagged IR.
        header = "\n".join(f"// CONDUCTOR: {cmd}" for cmd in commands)
        full_input = header + "\n\n" + tagged_ir

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(full_input)
            input_path = f.name

        try:
            cmd = [conductor, "--print-debug-locs-inline", input_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(
                    f"waveasm-conductor failed (rc={result.returncode}):\n{result.stderr}"
                )
            return result.stdout
        finally:
            os.unlink(input_path)

    def compile_to_asm(self, ir: str) -> str:
        """Run post-scheduling pipeline on already-scheduled WaveASM IR."""
        flags = [
            "--waveasm-linear-scan",
            "--max-vgprs=512",
            "--max-agprs=512",
            "--waveasm-insert-waitcnt",
            "--waveasm-hazard-mitigation",
            "--emit-assembly",
        ]
        stdout, stderr, rc = run_waveasm_translate(ir, self.workgroup_size, flags)
        if rc != 0:
            raise RuntimeError(f"compile_to_asm failed:\n{stderr}")
        return stdout

    def get_metrics(self, asm: str) -> dict:
        """Extract metrics from assembly text."""
        return count_asm_metrics(asm)

    def evaluate(self, commands: list) -> dict:
        """
        Full round-trip: tag → apply_moves → compile_to_asm → get_metrics.

        This is the main entry point for a search algorithm.
        """
        _, metrics = self.evaluate_with_ir(commands)
        return metrics

    def evaluate_with_ir(self, commands: list) -> tuple[str, dict]:
        """Like evaluate, but also returns the reordered tagged IR."""
        tagged = self.tag()
        reordered = self.apply_moves(tagged, commands)
        asm = self.compile_to_asm(reordered)
        return reordered, self.get_metrics(asm)

    def baseline(self) -> dict:
        """Evaluate with no moves (identity schedule). Caches result."""
        if self._baseline_cache is None:
            asm = self.compile_to_asm(self.waveasm_ir)
            self._baseline_cache = self.get_metrics(asm)
        return self._baseline_cache


def main():
    parser = argparse.ArgumentParser(
        description="Conductor: WaveASM instruction scheduling driver."
    )
    parser.add_argument(
        "--moves",
        nargs="*",
        default=None,
        help="Move commands (e.g. 'swap A B', 'move X after Y').",
    )
    parser.add_argument(
        "--moves-file",
        type=str,
        default=None,
        help="Read move commands from a file (one per line).",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print assembly metrics after scheduling.",
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Only show tagged IR, then exit.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Run the LLM-guided scheduling loop.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter model ID for --llm mode.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum LLM scheduling rounds (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        help="Reasoning effort for models that support it (default: high).",
    )
    args = parser.parse_args()

    # Collect commands from both sources.
    commands = []
    if args.moves:
        commands.extend(args.moves)
    if args.moves_file:
        commands.extend(
            line.strip()
            for line in Path(args.moves_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    print("Capturing kernel MLIR...", file=sys.stderr)
    mlir_text, wg_size = capture_kernel_mlir()
    print(f"  workgroup_size: {wg_size}", file=sys.stderr)

    print("Running pre-scheduling pipeline...", file=sys.stderr)
    waveasm_ir = run_pre_scheduling_pipeline(mlir_text, wg_size)
    print(f"  WaveASM IR: {len(waveasm_ir)} chars", file=sys.stderr)

    conductor = Conductor(waveasm_ir, wg_size)

    if args.tag_only:
        print(conductor.tag())
        return

    if args.llm:
        from conductor.llm import run_scheduling_loop, DEFAULT_MODEL

        model = args.model or DEFAULT_MODEL
        print(f"Running LLM scheduling loop (model={model})...", file=sys.stderr)
        result = run_scheduling_loop(
            conductor,
            max_rounds=args.max_rounds,
            model=model,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        print("\n=== LLM Scheduling Result ===")
        print(f"  rounds: {result['rounds']}")
        print(f"  commands: {result['commands']}")
        print("  baseline:")
        for k, v in result["baseline_metrics"].items():
            print(f"    {k}: {v}")
        print("  best:")
        for k, v in result["metrics"].items():
            print(f"    {k}: {v}")
        usage = result.get("usage")
        if usage:
            print(
                f"  tokens: {usage.tokens} (in={usage.input_tokens} out={usage.output_tokens})"
            )
            print(f"  cost: ${usage.cost:.4f}")
        return

    if commands:
        print(f"Applying {len(commands)} move(s)...", file=sys.stderr)
        metrics = conductor.evaluate(commands)
    else:
        print("No moves specified, running baseline...", file=sys.stderr)
        metrics = conductor.baseline()

    if args.metrics:
        print("\n=== Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
