# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for numeric probing in read coalescing.

The pairwise merge uses numeric probing to verify that adjacent reads
have consistent per-dim offset diffs.  With symbolic K, the 2D
decomposition (row = offset floordiv K/2, col = offset mod K/2)
can give inconsistent diffs when probe values don't respect divisibility
constraints.  Applying divisibility forward subs (K -> 256*K') fixes
this.

The expressions here are taken from the MXFP4 preshuffle B-scale
codegen (test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm on gfx950).
"""

import pytest
import sympy

from wave_lang.kernel.wave.analysis.partition_strided_operators import (
    _MERGE_PROBES,
    _eval_expr,
    _find_merge_dim_from_diffs,
)
from wave_lang.kernel.wave.utils.symbol_utils import (
    _numeric_eval_constant,
    safe_subs,
)


# --- Symbols matching the MXFP4 B-scale path ---
# t = thread_id_x (0..63), K = GEMM K dimension (multiple of 256),
# wg_m/wg_n = workgroup tile offsets.
t = sympy.Symbol("t", integer=True, nonneg=True)
K = sympy.Symbol("K", integer=True, positive=True)
wg_m = sympy.Symbol("wg_m", integer=True, nonneg=True)
wg_n = sympy.Symbol("wg_n", integer=True, nonneg=True)
half_K = K / 2

# Preshuffle base offset for thread t (from MLIR #map27/#map28 pattern):
#   base(t) = t*16 + ((t%64)//16)*256 - (t//16)*256
# For t in 0..63 this simplifies to t*16, but sympy doesn't know that.
_base = t * 16 + ((t % 64) // 16) * 256 - (t // 16) * 256


def _make_row_col(offset):
    """Build (row, col) from a linearized preshuffle offset."""
    row = wg_m * 64 + wg_n * 256 + sympy.floor(offset / half_K)
    col = offset % half_K
    return row, col


# The MXFP4 preshuffle MMA layout scatters 16 scale bytes across
# offsets with large gaps.  From the MLIR maps (#map27-#map36), the
# actual offsets within a single thread's 16-byte read are:
#   base, base+1024, base+2, base+3, base+11, base+15,
#   base+16*N, base+16*N+1024, ...
# Here we use the first 4 groups from the MLIR maps: offsets
# base+0, base+1024, base+2, base+3 — which is enough to
# demonstrate the inconsistency.
_PRESHUFFLE_DELTAS = [
    0,
    1,
    1024,
    1025,
    2,
    3,
    11,
    15,
    16,
    17,
    1040,
    1041,
    18,
    19,
    27,
    31,
]

BYTE_READS = []
for delta in _PRESHUFFLE_DELTAS:
    row_i, col_i = _make_row_col(_base + delta)
    flat_i = row_i * half_K + col_i
    BYTE_READS.append({"row": row_i, "col": col_i, "flat": flat_i, "delta": delta})

# Divisibility forward sub: K -> 256 * K'.
K_prime = sympy.Symbol("_K_div_256", integer=True, positive=True)
DIV_FWD = [(K, 256 * K_prime)]


def _build_merge_probe_maps(expressions):
    """Build _MERGE_PROBES probe maps over free symbols in expressions."""
    all_free = set()
    for expr in expressions:
        if hasattr(expr, "free_symbols"):
            all_free |= expr.free_symbols
    free_list = sorted(all_free, key=str)
    return [{s: gen(i) for i, s in enumerate(free_list)} for gen in _MERGE_PROBES]


def _find_reads_by_delta(d0, d1):
    """Return the two BYTE_READS entries with the given deltas."""
    r0 = next(br for br in BYTE_READS if br["delta"] == d0)
    r1 = next(br for br in BYTE_READS if br["delta"] == d1)
    return r0, r1


class TestMergeProbeConsistency:
    """Verify that numeric probing gives consistent per-dim diffs
    for adjacent B-scale reads, with and without divisibility subs."""

    def test_flat_offset_diff_always_correct(self):
        """Flat offset diff between bytes equals their delta difference."""
        all_flats = [br["flat"] for br in BYTE_READS]
        probes = _build_merge_probe_maps(all_flats)
        for probe in probes:
            for i in range(len(BYTE_READS) - 1):
                flat_i = _eval_expr(BYTE_READS[i]["flat"], probe)
                flat_j = _eval_expr(BYTE_READS[i + 1]["flat"], probe)
                expected = BYTE_READS[i + 1]["delta"] - BYTE_READS[i]["delta"]
                assert flat_j - flat_i == expected, (
                    f"flat diff mismatch for deltas "
                    f"{BYTE_READS[i]['delta']},{BYTE_READS[i+1]['delta']}"
                )

    def test_per_dim_diffs_inconsistent_without_div_subs(self):
        """Without divisibility subs, per-dim diffs are inconsistent
        across probe sets for reads with large preshuffle gaps.

        Reads at offsets base+0 and base+1024 (from MXFP4 maps #map27
        and #map29) have row_diff and col_diff that depend on K/2.
        At K/2=68: row_diff=15, col_diff=-3.
        At K/2=125: row_diff=8, col_diff=20. Etc.
        """
        r0, r1 = _find_reads_by_delta(0, 1024)
        all_exprs = list(r0.values()) + list(r1.values())
        probes = _build_merge_probe_maps(all_exprs)

        diffs_per_probe = []
        for probe in probes:
            row_diff = _eval_expr(r1["row"], probe) - _eval_expr(r0["row"], probe)
            col_diff = _eval_expr(r1["col"], probe) - _eval_expr(r0["col"], probe)
            diffs_per_probe.append((row_diff, col_diff))

        all_agree = all(d == diffs_per_probe[0] for d in diffs_per_probe[1:])
        assert not all_agree, (
            "Expected inconsistent per-dim diffs across probes, "
            f"but all agreed: {diffs_per_probe[0]}"
        )

    def test_adjacent_diffs_consistent_with_div_subs(self):
        """With divisibility subs (K -> 256*K'), adjacent byte pairs
        have consistent diffs: row_diff=0, col_diff=delta_diff."""
        div_reads = [
            {k: safe_subs(v, DIV_FWD) for k, v in br.items()} for br in BYTE_READS
        ]
        all_exprs = []
        for br in div_reads:
            all_exprs.extend(v for k, v in br.items() if k != "delta")
        probes = _build_merge_probe_maps(all_exprs)

        # Test all consecutive pairs sorted by delta.
        sorted_reads = sorted(zip(BYTE_READS, div_reads), key=lambda x: x[0]["delta"])
        for probe in probes:
            for idx in range(len(sorted_reads) - 1):
                orig_a, div_a = sorted_reads[idx]
                orig_b, div_b = sorted_reads[idx + 1]
                row_diff = _eval_expr(div_b["row"], probe) - _eval_expr(
                    div_a["row"], probe
                )
                col_diff = _eval_expr(div_b["col"], probe) - _eval_expr(
                    div_a["col"], probe
                )
                expected_flat = orig_b["delta"] - orig_a["delta"]
                assert row_diff == 0, (
                    f"row diff != 0 for deltas " f"{orig_a['delta']},{orig_b['delta']}"
                )
                assert col_diff == expected_flat, (
                    f"col diff {col_diff} != {expected_flat} for deltas "
                    f"{orig_a['delta']},{orig_b['delta']}"
                )

    def test_large_gap_consistent_with_div_subs(self):
        """Reads 1024 apart have consistent diffs after div subs."""
        r0, r1 = _find_reads_by_delta(0, 1024)
        div_r0 = {k: safe_subs(v, DIV_FWD) for k, v in r0.items()}
        div_r1 = {k: safe_subs(v, DIV_FWD) for k, v in r1.items()}

        all_exprs = [v for d in [div_r0, div_r1] for k, v in d.items() if k != "delta"]
        probes = _build_merge_probe_maps(all_exprs)

        for probe in probes:
            row_diff = _eval_expr(div_r1["row"], probe) - _eval_expr(
                div_r0["row"], probe
            )
            col_diff = _eval_expr(div_r1["col"], probe) - _eval_expr(
                div_r0["col"], probe
            )
            assert row_diff == 0, f"row diff {row_diff} != 0"
            assert col_diff == 1024, f"col diff {col_diff} != 1024"

    def test_merge_dim_found_with_div_subs(self):
        """After div subs, _find_merge_dim_from_diffs finds the col dim."""
        r0, r1 = _find_reads_by_delta(0, 1)
        div_r0 = {k: safe_subs(v, DIV_FWD) for k, v in r0.items()}
        div_r1 = {k: safe_subs(v, DIV_FWD) for k, v in r1.items()}
        all_exprs = [v for d in [div_r0, div_r1] for k, v in d.items() if k != "delta"]
        probes = _build_merge_probe_maps(all_exprs)

        dim_row = sympy.Symbol("dim_row")
        dim_col = sympy.Symbol("dim_col")
        dims = [dim_row, dim_col]
        for probe in probes:
            row_diff = _eval_expr(div_r1["row"], probe) - _eval_expr(
                div_r0["row"], probe
            )
            col_diff = _eval_expr(div_r1["col"], probe) - _eval_expr(
                div_r0["col"], probe
            )
            result = _find_merge_dim_from_diffs(
                {dim_row: row_diff, dim_col: col_diff}, 1, dims
            )
            assert result == dim_col


class TestNumericEvalConstant:
    """Test _numeric_eval_constant with preshuffle-like expressions.

    Note: _numeric_eval_constant uses _PROBE_POOL which includes 0,
    causing ZeroDivisionError on floor/Mod expressions with symbolic
    divisors.  It returns None conservatively on any error, so these
    complex expressions are beyond its reach.  This is documented
    behavior, not a bug — the merge probing (_MERGE_PROBES) handles
    these cases instead.
    """

    def test_simple_constant(self):
        """Sanity: detects that a simple constant expression is constant."""
        x = sympy.Symbol("x", integer=True, nonneg=True)
        assert _numeric_eval_constant(3 * x - 3 * x) == 0
        assert _numeric_eval_constant(x - x + 7) == 7

    def test_floor_mod_identity_returns_none(self):
        """floor(x/d)*d + x%d - x is always 0, but _numeric_eval_constant
        can't prove it because _PROBE_POOL includes 0 → ZeroDivisionError."""
        x = sympy.Symbol("x", integer=True, nonneg=True)
        d = sympy.Symbol("d", integer=True, positive=True)
        expr = sympy.floor(x / d) * d + sympy.Mod(x, d) - x
        # Returns None because probing hits d=0.
        assert _numeric_eval_constant(expr) is None

    def test_row_diff_not_constant_without_div_subs(self):
        """Row diff for adjacent bytes is not constant when K is symbolic."""
        row_diff = BYTE_READS[1]["row"] - BYTE_READS[0]["row"]
        assert _numeric_eval_constant(row_diff) is None


class TestFindMergeDim:
    """Test _find_merge_dim_from_diffs helper."""

    def test_single_dim_matches(self):
        dim_a = sympy.Symbol("a")
        dim_b = sympy.Symbol("b")
        dims = [dim_a, dim_b]
        assert _find_merge_dim_from_diffs({dim_a: 0, dim_b: 1}, 1, dims) == dim_b

    def test_both_change_returns_none(self):
        dim_a = sympy.Symbol("a")
        dim_b = sympy.Symbol("b")
        dims = [dim_a, dim_b]
        assert _find_merge_dim_from_diffs({dim_a: 1, dim_b: 1}, 1, dims) is None

    def test_wrong_diff_returns_none(self):
        dim_a = sympy.Symbol("a")
        dim_b = sympy.Symbol("b")
        dims = [dim_a, dim_b]
        assert _find_merge_dim_from_diffs({dim_a: 0, dim_b: 2}, 1, dims) is None

    def test_nonzero_non_ept_returns_none(self):
        """If a dim changes by something other than 0 or ept, reject."""
        dim_a = sympy.Symbol("a")
        dim_b = sympy.Symbol("b")
        dims = [dim_a, dim_b]
        assert _find_merge_dim_from_diffs({dim_a: 3, dim_b: 0}, 1, dims) is None
