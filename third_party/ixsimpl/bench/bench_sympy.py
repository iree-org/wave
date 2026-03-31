# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: ixsimpl vs SymPy (simplify + cancel) on the real corpus.

Usage:
    python bench/bench_sympy.py [--top N] [--iters N]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ixsimpl
import sympy

CORPUS_PATH = Path("test/corpus.txt")
ASSUMPTIONS_PATH = Path("test/corpus_assumptions.txt")

N_ITERS_DEFAULT = 3
TOP_DEFAULT = 10

EXPR_RE = re.compile(r"^simplify time:\s*([\d.]+)s:\s*(.+)$")
ASSUMPTION_RE = re.compile(r"^\s*(\S+)\s*(>=|<=|==|!=|>|<)\s*(\S+)\s*$")


def strip_dollar(s: str) -> str:
    return s.replace("$", "_")


def _build_sympy_locals(text: str) -> dict[str, sympy.Symbol]:
    """Extract all identifiers from corpus text and create a Symbol for each,
    overriding any SymPy builtins (N, S, E, I, O) that shadow them."""
    cleaned = strip_dollar(text)
    tokens = set(re.findall(r"[A-Za-z_]\w*", cleaned))
    builtins = {
        "Piecewise",
        "floor",
        "ceiling",
        "Mod",
        "Max",
        "Min",
        "True",
        "False",
        "xor",
        "And",
        "Or",
        "Not",
        "simplify",
        "time",
    }
    return {t: sympy.Symbol(t, integer=True) for t in tokens - builtins}


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------


def load_corpus() -> list[tuple[str, float]]:
    """Return list of (expr_string, recorded_sympy_seconds)."""
    entries: list[tuple[str, float]] = []
    for line in CORPUS_PATH.read_text().splitlines():
        m = EXPR_RE.match(line)
        if m:
            entries.append((m.group(2).strip(), float(m.group(1))))
    return entries


def load_assumptions_ixsimpl(ctx: ixsimpl.Context) -> list[ixsimpl.Expr]:
    """Parse corpus_assumptions.txt into ixsimpl assumption nodes."""
    cmp_map = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: ctx.eq(a, b),
        "!=": lambda a, b: ctx.ne(a, b),
    }
    assumptions: list[ixsimpl.Expr] = []
    for line in ASSUMPTIONS_PATH.read_text().splitlines():
        m = ASSUMPTION_RE.match(line)
        if not m:
            continue
        lhs = ctx.parse(m.group(1))
        rhs = ctx.parse(m.group(3))
        op = m.group(2)
        if lhs.is_error or rhs.is_error:
            continue
        assumptions.append(cmp_map[op](lhs, rhs))  # type: ignore[no-untyped-call]
    return assumptions


# ---------------------------------------------------------------------------
#  Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class ExprResult:
    expr_str: str
    recorded_sympy_s: float
    ixsimpl_us: float = 0.0
    sympy_simplify_us: float = 0.0
    sympy_cancel_us: float = 0.0
    ixsimpl_ok: bool = True
    sympy_ok: bool = True


def time_ixsimpl(
    ctx: ixsimpl.Context,
    expr_strs: list[str],
    assumptions: list[ixsimpl.Expr],
    n_iters: int,
) -> list[float]:
    """Return per-expression best-of-N times in microseconds."""
    n = len(expr_strs)
    best = [float("inf")] * n
    for _ in range(n_iters):
        for i, s in enumerate(expr_strs):
            t0 = time.perf_counter()
            node = ctx.parse(s)
            if not node.is_error:
                node.simplify(assumptions=assumptions)
            elapsed = (time.perf_counter() - t0) * 1e6
            if elapsed < best[i]:
                best[i] = elapsed
    return best


def _time_sympy_fn(
    fn: Any,
    label: str,
    sp_exprs: list[sympy.Basic | None],
    n_iters: int,
    timeout_s: float,
) -> list[float]:
    """Return per-expression best-of-N times (us) for a SymPy function.
    Skips expressions that exceed timeout_s on any iteration."""
    n = len(sp_exprs)
    best = [float("inf")] * n
    skipped: set[int] = set()
    for it in range(n_iters):
        wall_start = time.perf_counter()
        n_timed = 0
        for i, sp in enumerate(sp_exprs):
            if sp is None or i in skipped:
                if best[i] == float("inf"):
                    best[i] = 0.0
                continue
            t0 = time.perf_counter()
            fn(sp)
            elapsed_s = time.perf_counter() - t0
            elapsed_us = elapsed_s * 1e6
            if elapsed_s > timeout_s:
                skipped.add(i)
                best[i] = elapsed_us
            elif elapsed_us < best[i]:
                best[i] = elapsed_us
            n_timed += 1
            if n_timed % 50 == 0:
                wall = time.perf_counter() - wall_start
                print(
                    f"\r  {label} [{it + 1}/{n_iters}]: " f"{n_timed}/{n} ({wall:.0f}s elapsed)   ",
                    end="",
                    flush=True,
                )
        wall = time.perf_counter() - wall_start
        print(
            f"\r  {label} [{it + 1}/{n_iters}]: {n} done in {wall:.1f}s" + " " * 20,
            flush=True,
        )
    if skipped:
        print(f"  ({len(skipped)} exprs exceeded {timeout_s}s timeout)")
    return best


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def fmt_us(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:.2f}s"
    if us >= 1000:
        return f"{us / 1000:.1f}ms"
    return f"{us:.0f}us"


def speedup_str(slow: float, fast: float) -> str:
    if fast <= 0:
        return "inf"
    return f"{slow / fast:.0f}x"


def geo_mean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    log_sum = sum(math.log2(v) for v in positive)
    return 2 ** (log_sum / len(positive))


def _speedup_line(label: str, total: float, ixs_total: float, speedups: list[float]) -> str:
    return (
        f"    {label:20s}  total {speedup_str(total, ixs_total):>6s}"
        f"   median {speedup_str(statistics.median(speedups), 1):>6s}"
        f"   geomean {speedup_str(geo_mean(speedups), 1):>6s}"
    )


def print_report(
    results: list[ExprResult], top_n: int, ran_simplify: bool, ran_cancel: bool
) -> None:
    valid = [r for r in results if r.ixsimpl_ok and r.sympy_ok]
    if not valid:
        print("No valid expressions to compare.")
        return

    ixs_total = sum(r.ixsimpl_us for r in valid)
    recorded_total = sum(r.recorded_sympy_s * 1e6 for r in valid)
    n_valid = len(valid)
    n_skip = len(results) - n_valid

    print(f"\n{'=' * 72}")
    print(f"  ixsimpl vs SymPy — {n_valid} expressions ({n_skip} skipped)")
    print(f"{'=' * 72}\n")

    # --- Aggregate ---
    print("  Aggregate (best of N iterations):\n")
    print(f"    {'':30s} {'Total':>10s}  {'Per-expr':>10s}")
    print(f"    {'ixsimpl':30s} {fmt_us(ixs_total):>10s}  {fmt_us(ixs_total / n_valid):>10s}")

    simp_total = cancel_total = 0.0
    speedups_simplify: list[float] = []
    speedups_cancel: list[float] = []

    if ran_simplify:
        simp_total = sum(r.sympy_simplify_us for r in valid)
        speedups_simplify = [r.sympy_simplify_us / r.ixsimpl_us for r in valid if r.ixsimpl_us > 0]
        print(
            f"    {'sympy.simplify':30s} {fmt_us(simp_total):>10s}  "
            f"{fmt_us(simp_total / n_valid):>10s}"
        )

    if ran_cancel:
        cancel_total = sum(r.sympy_cancel_us for r in valid)
        speedups_cancel = [r.sympy_cancel_us / r.ixsimpl_us for r in valid if r.ixsimpl_us > 0]
        print(
            f"    {'sympy.cancel':30s} {fmt_us(cancel_total):>10s}  "
            f"{fmt_us(cancel_total / n_valid):>10s}"
        )

    print(
        f"    {'recorded sympy (corpus.txt)':30s} {fmt_us(recorded_total):>10s}  "
        f"{fmt_us(recorded_total / n_valid):>10s}"
    )

    print("\n  Speedup vs ixsimpl:\n")
    if ran_simplify and speedups_simplify:
        print(_speedup_line("sympy.simplify", simp_total, ixs_total, speedups_simplify))
    if ran_cancel and speedups_cancel:
        print(_speedup_line("sympy.cancel", cancel_total, ixs_total, speedups_cancel))

    # --- Top N slowest per tool ---
    tops: list[tuple[str, Any]] = [("ixsimpl (slowest)", lambda r: r.ixsimpl_us)]
    if ran_simplify:
        tops.append(("sympy.simplify (slowest)", lambda r: r.sympy_simplify_us))
    if ran_cancel:
        tops.append(("sympy.cancel (slowest)", lambda r: r.sympy_cancel_us))

    for label, key in tops:
        ranked = sorted(valid, key=key, reverse=True)[:top_n]
        print(f"\n  Top {top_n} {label}:\n")
        for r in ranked:
            trunc = r.expr_str[:60] + ("..." if len(r.expr_str) > 60 else "")
            print(f"    {fmt_us(key(r)):>10s}  {trunc}")

    # --- One-liner for CI ---
    parts = [f"{n_valid} exprs", f"ixsimpl {fmt_us(ixs_total)}"]
    if ran_simplify:
        parts.append(f"sympy.simplify {fmt_us(simp_total)} ({speedup_str(simp_total, ixs_total)})")
    if ran_cancel:
        parts.append(
            f"sympy.cancel {fmt_us(cancel_total)} ({speedup_str(cancel_total, ixs_total)})"
        )
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY: {', '.join(parts)}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ixsimpl vs SymPy benchmark")
    parser.add_argument("--top", type=int, default=TOP_DEFAULT, help="show top N slowest")
    parser.add_argument("--iters", type=int, default=N_ITERS_DEFAULT, help="timing iterations")
    parser.add_argument("--timeout", type=float, default=2.0, help="per-expression timeout (s)")
    parser.add_argument(
        "--cancel", action="store_true", help="run only sympy.cancel (skip simplify)"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="run only sympy.simplify (skip cancel)"
    )
    args = parser.parse_args()
    run_both = not args.cancel and not args.simplify
    run_simplify = args.simplify or run_both
    run_cancel = args.cancel or run_both

    corpus = load_corpus()
    if not corpus:
        print(f"No expressions found in {CORPUS_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(corpus)} expressions from {CORPUS_PATH}")

    # --- Parse with SymPy (strip $, override builtins) ---
    raw_text = CORPUS_PATH.read_text()
    sp_locals = _build_sympy_locals(raw_text)
    print("Parsing with SymPy...", end=" ", flush=True)
    sp_exprs: list[sympy.Basic | None] = []
    sp_fail = 0
    for expr_str, _ in corpus:
        try:
            sp_exprs.append(sympy.sympify(strip_dollar(expr_str), locals=sp_locals))
        except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
            sp_exprs.append(None)
            sp_fail += 1
    print(f"done ({sp_fail} failures)")

    # --- Parse with ixsimpl ---
    ctx = ixsimpl.Context()
    assumptions = load_assumptions_ixsimpl(ctx)
    print(f"Loaded {len(assumptions)} assumptions")

    # --- Time ixsimpl ---
    expr_strs = [e for e, _ in corpus]
    print(f"Timing ixsimpl ({args.iters} iters)...", end=" ", flush=True)
    ixs_times = time_ixsimpl(ctx, expr_strs, assumptions, args.iters)
    print("done")

    n = len(corpus)
    simp_times = [0.0] * n
    cancel_times = [0.0] * n

    if run_simplify:
        print(f"Timing sympy.simplify ({args.iters} iters, {args.timeout}s timeout)...")
        simp_times = _time_sympy_fn(sympy.simplify, "simplify", sp_exprs, args.iters, args.timeout)

    if run_cancel:
        print(f"Timing sympy.cancel ({args.iters} iters, {args.timeout}s timeout)...")
        cancel_times = _time_sympy_fn(sympy.cancel, "cancel", sp_exprs, args.iters, args.timeout)

    # --- Build results ---
    results: list[ExprResult] = []
    for i, (expr_str, recorded_s) in enumerate(corpus):
        r = ExprResult(
            expr_str=expr_str,
            recorded_sympy_s=recorded_s,
            ixsimpl_us=ixs_times[i],
            sympy_simplify_us=simp_times[i],
            sympy_cancel_us=cancel_times[i],
            ixsimpl_ok=True,
            sympy_ok=sp_exprs[i] is not None,
        )
        results.append(r)

    print_report(results, args.top, run_simplify, run_cancel)


if __name__ == "__main__":
    main()
