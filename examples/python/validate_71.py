#!/usr/bin/env python3
"""
Validate 7.1_schedule.py across all combinations of macro tiles and shapes.

Runs the inner 7.1_schedule.py test for every (macro-tile, shape) pair and
prints a single PASS/FAIL summary line per combination, suppressing verbose
output from the inner script.

The test function inside 7.1_schedule.py is chosen automatically from
--n-wave, --backend, and --dyn.

Usage:
    python examples/python/validate_71.py \
        --mt macro_tiles.txt --shapes shapes.txt \
        --dyn static --backend waveasm --n-wave 4
"""

import argparse
import os
import re
import subprocess
import sys


# ============================================================
# Per-macro-tile overrides.
#
# Keys are normalized "MxNxK" strings (using 'x' separator).
# Values are dicts with extra CLI flags to pass to the inner script.
#
# Recognized keys in the value dict:
#   "extra_args" : list[str] -- extra CLI arguments appended verbatim
#
# Edit this table to add special handling for new macro tiles.
# ============================================================
MACRO_TILE_OVERRIDES = {
    "128x256x256": {"extra_args": ["--wave_shape", "1,4"]},
    "128x32x256": {"extra_args": ["--wave_shape", "2,2"]},
    "224x160x256": {"extra_args": ["--wave_shape", "2,2"]},
    "256x192x256": {"extra_args": ["--wave_shape", "1,4"]},
    "256x160x256": {"extra_args": ["--wave_shape", "2,2"]},
    "256x224x256": {"extra_args": ["--no-unroll", "--wave_shape", "2,2"]},
}

# ============================================================
# Per-shape overrides (same structure as MACRO_TILE_OVERRIDES).
# ============================================================
SHAPE_OVERRIDES = {
}

# ============================================================
# Global extra arguments appended to every invocation.
# Edit this list to inject flags that should always be present.
# ============================================================
GLOBAL_EXTRA_ARGS: list[str] = []

# ============================================================
# Test-name lookup table.
#
# Maps (n_wave, backend, dyn) to the 7.1_schedule.py test function name.
# Edit this table when new tests are added to 7.1_schedule.py.
# ============================================================
TEST_MAP = {
    (4, "llvm", "static"): "test_dbuf_4wave_mxfp_preshuffle_b_gemm",
    (4, "llvm", "dynamic"): "test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm",
    (4, "waveasm", "static"): "test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp",
    (8, "llvm", "static"): "test_dbuf_8wave_pingpong_mxfp_gemm",
    (8, "llvm", "dynamic"): "test_dbuf_8wave_pingpong_mxfp_gemm",
}

# ============================================================
# Path to the inner script (relative to the repo root).
# ============================================================
INNER_SCRIPT = os.path.join("examples", "python", "7.1_schedule.py")

# ============================================================
# Timeout (seconds) for each inner-script invocation.
# ============================================================
PER_TEST_TIMEOUT = 600


def _parse_triple(s: str) -> tuple[str, str]:
    """Parse 'MxNxK' or 'M,N,K' into (comma_form, display_form)."""
    s = s.strip()
    parts = re.split(r"[x,]+", s)
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Cannot parse triple from '{s}' -- expected MxNxK or M,N,K")
    comma_form = ",".join(parts)
    display_form = "x".join(parts)
    return comma_form, display_form


def _read_triples(path: str) -> list[tuple[str, str]]:
    """Read a file of triples, returning list of (comma_form, display_form)."""
    with open(path) as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return [_parse_triple(line) for line in lines]


def _abbreviated(text: str, n: int = 5) -> str:
    """Return the first and last n lines of text, with '...' in between."""
    lines = text.splitlines()
    if len(lines) <= 2 * n:
        return text
    return "\n".join(lines[:n] + ["..."] + lines[-n:])


def _resolve_test_name(n_wave: int, backend: str, dyn: str) -> str:
    """Look up the 7.1_schedule.py test function from (n_wave, backend, dyn)."""
    key = (n_wave, backend, dyn)
    if key not in TEST_MAP:
        print(
            f"Error: no test configured for n-wave={n_wave}, "
            f"backend={backend}, dyn={dyn}.\n"
            f"Add an entry to TEST_MAP in {__file__}.",
            file=sys.stderr,
        )
        sys.exit(2)
    return TEST_MAP[key]


def _build_cmd(
    test_name: str,
    mt_comma: str,
    mt_display: str,
    shape_comma: str,
    shape_display: str,
) -> list[str]:
    """Assemble the subprocess command list for one (macro-tile, shape) pair."""
    cmd = [
        sys.executable,
        INNER_SCRIPT,
        "--test",
        test_name,
        "--block",
        mt_comma,
        "--shape",
        shape_comma,
    ]
    cmd.extend(GLOBAL_EXTRA_ARGS)
    cmd.extend(MACRO_TILE_OVERRIDES.get(mt_display, {}).get("extra_args", []))
    cmd.extend(SHAPE_OVERRIDES.get(shape_display, {}).get("extra_args", []))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate 7.1_schedule.py across macro-tile x shape combinations.",
    )
    parser.add_argument(
        "--mt",
        required=True,
        help="File with macro tiles, one per line (MxNxK or M,N,K)",
    )
    parser.add_argument(
        "--shapes",
        required=True,
        help="File with shapes, one per line (MxNxK or M,N,K)",
    )
    parser.add_argument(
        "--dyn",
        choices=["static", "dynamic"],
        default=None,
        help="Static or dynamic mode (required)",
    )
    parser.add_argument(
        "--backend",
        choices=["llvm", "waveasm"],
        default=None,
        help="Backend to use (required)",
    )
    parser.add_argument(
        "--n-wave",
        type=int,
        choices=[4, 8],
        default=None,
        help="Number of waves (required)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=PER_TEST_TIMEOUT,
        help=f"Per-test timeout in seconds (default {PER_TEST_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show abbreviated output (first/last 5 lines) for failures",
    )
    args = parser.parse_args()

    errors = []
    if args.dyn is None:
        errors.append("--dyn is required (must be 'static' or 'dynamic')")
    if args.backend is None:
        errors.append("--backend is required (must be 'llvm' or 'waveasm')")
    if args.n_wave is None:
        errors.append("--n-wave is required (must be 4 or 8)")
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    test_name = _resolve_test_name(args.n_wave, args.backend, args.dyn)
    macro_tiles = _read_triples(args.mt)
    shapes = _read_triples(args.shapes)

    any_fail = False
    for mt_comma, mt_display in macro_tiles:
        for shape_comma, shape_display in shapes:
            cmd = _build_cmd(test_name, mt_comma, mt_display, shape_comma, shape_display)

            output = ""
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                status = "PASS" if result.returncode == 0 else "FAIL"
                output = (result.stdout + result.stderr).strip()
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
            except Exception:
                status = "FAIL"

            if status != "PASS":
                any_fail = True

            print(
                f"dyn={args.dyn}, n-wave={args.n_wave}, backend={args.backend}, "
                f"macro-tile={mt_display}, shape={shape_display}, result={status}"
            )
            if args.verbose and status != "PASS" and output:
                for line in _abbreviated(output).splitlines():
                    print(f"  | {line}")
                print()

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
