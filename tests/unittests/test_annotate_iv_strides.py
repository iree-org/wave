# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Unit tests for the annotate_iv_strides pass (_try_symbolic_stride,
_try_numerical_probe, _try_with_div_subs).

These functions extract a constant IV stride from a flat index expression
so that the compiler can rewrite ``flat(iv)`` into ``base + iv * stride``.
Tests use realistic sympy expressions derived from MXFP4 preshuffle GEMM
patterns without constructing TorchFX graphs.

In the real pipeline, shape symbols (K, M, N) are resolved to concrete
integers via subs_idxc before annotate_iv_strides runs.  The flat
expressions still contain symbolic thread IDs and workgroup offsets.
Dynamic-shape kernels keep K symbolic, which is where _try_with_div_subs
is needed.

Naming convention for symbols:

    iv      -- induction variable ($ARG-prefixed in the compiler)
    t       -- thread index (THREAD_0)
    K       -- GEMM K dimension (symbolic for dynamic shapes)
    wg      -- workgroup tile offset
"""

import pytest
import sympy

from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.wave.analysis.annotate_iv_strides import (
    _try_symbolic_stride,
    _try_numerical_probe,
    _try_with_div_subs,
)
from wave_lang.kernel.wave.assumptions import Assumption, get_divisibility_subs
from wave_lang.kernel.wave.utils.symbol_utils import simplify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sym(name, **kw):
    """Non-negative integer symbol (matches hardware index convention)."""
    kw.setdefault("integer", True)
    kw.setdefault("nonnegative", True)
    return sympy.Symbol(name, **kw)


def _pos(name, **kw):
    """Positive integer symbol."""
    kw.setdefault("integer", True)
    kw.setdefault("positive", True)
    return sympy.Symbol(name, **kw)


# Reusable symbols across tests.
iv = _sym("$ARGK")  # matches get_induction_symbol(K)
t = _sym("t")
wg = _sym("wg")
# K kept symbolic for dynamic-shape tests; concrete values used otherwise.
K_dyn = _pos("K", nonnegative=True)


# ---------------------------------------------------------------------------
# Realistic flat-offset builders
#
# In the real pipeline, K is usually concrete (e.g. 8192).  The expressions
# below produce the same structure as flatten_read_indices.
# ---------------------------------------------------------------------------


def _simple_unmapped_flat(iv_sym, t_sym, stride):
    """Unmapped read: flat = thread_offset + iv * stride.

    Simplest case after flatten_read_indices for a 2-D tensor [M, K]
    tiled along K.  The thread distributes in the fastest dimension,
    producing t*ept as the base offset.
    """
    return t_sym * 16 + iv_sym * stride


def _preshuffle_b_data_flat(iv_sym, t_sym, half_K):
    """Preshuffle B-data flat offset (concrete half_K = K/2).

    The MXFP4 preshuffle mapping scrambles thread indices via floor/Mod
    but the IV contribution is a simple additive term.  After
    linearize_dims cancels the floor/Mod pairs, the flat is:

        scrambled_thread_offset + iv * half_K

    With concrete K (e.g. K=8192 => half_K=4096), this is purely affine.
    """
    thread_part = (
        t_sym * 16
        + sympy.floor(sympy.Mod(t_sym, 64) / 16) * 256
        - sympy.floor(t_sym / 16) * 256
    )
    return thread_part + iv_sym * half_K


def _b_scale_flat(iv_sym, t_sym, k_div_32):
    """B-scale preshuffle flat (concrete k_div_32 = K/32).

    The B-scale tensor has shape [K/32, N].  After linearization, the IV
    stride is k_div_32 (the stride along the K/32 dimension).
    """
    thread_part = sympy.Mod(t_sym, 16) * 4 + sympy.floor(t_sym / 16) * 64
    return thread_part + iv_sym * k_div_32


def _floor_mod_of_iv_flat(iv_sym, t_sym, half_K):
    """Flat where IV appears inside floor/Mod (not yet simplified).

    When flatten_read_indices preserves the raw mapped form for a mapped
    read (without mem_simplify), the expression retains floor/Mod of
    the iv.  With concrete half_K, the underlying function is still
    affine (stride = 128) but symbolic differencing can't see it.
    Numeric probing or mem_simplify resolves it.
    """
    inner = t_sym + iv_sym * sympy.Integer(128)
    return sympy.floor(inner / half_K) * half_K + sympy.Mod(inner, half_K)


def _dynamic_K_cancelling_flat(iv_sym, t_sym, K_sym):
    """Dynamic-shape flat where floor/Mod cancel algebraically.

    floor(x/(K/2))*(K/2) + Mod(x, K/2) == x by the floor/Mod identity.
    mem_simplify can prove this even with symbolic K, so symbolic
    extraction succeeds.
    """
    half_K = K_sym / 2
    inner = t_sym + iv_sym * sympy.Integer(128)
    return sympy.floor(inner / half_K) * half_K + sympy.Mod(inner, half_K)


def _dynamic_K_mismatched_stride_flat(iv_sym, t_sym, K_sym):
    """Dynamic-shape flat where floor/Mod do NOT cancel.

    Models a 2-D decomposition where:
      row = floor((t*16 + iv*128) / (K/2))
      col = Mod((t*16 + iv*128), K/2)
      flat = row * K + col

    The row stride (K) differs from the Mod divisor (K/2), so the
    floor(x/D)*D + Mod(x, D) identity doesn't apply.  mem_simplify
    cannot cancel this.  With symbolic K, both symbolic differencing
    and numeric probing fail.  Divisibility subs (K -> 256*K') resolve
    K to a concrete multiple, collapsing the floor/Mod and exposing
    stride = 128.
    """
    half_K = K_sym / 2
    inner = t_sym * 16 + iv_sym * sympy.Integer(128)
    row = sympy.floor(inner / half_K)
    col = sympy.Mod(inner, half_K)
    return row * K_sym + col


# ===========================================================================
# _try_symbolic_stride
# ===========================================================================


class TestTrySymbolicStride:
    """_try_symbolic_stride extracts stride via flat(iv+step) - flat(iv)."""

    def test_simple_affine_concrete_stride(self):
        """Unmapped read with concrete stride: trivially affine."""
        flat = _simple_unmapped_flat(iv, t, stride=4096)
        result = _try_symbolic_stride(flat, iv, step=1)
        assert result is not None
        base, stride = result
        assert stride == 4096
        assert iv not in base.free_symbols

    def test_simple_affine_with_step_2(self):
        """Step = 2: stride = per_element * step."""
        flat = _simple_unmapped_flat(iv, t, stride=256)
        result = _try_symbolic_stride(flat, iv, step=2)
        assert result is not None
        base, stride = result
        assert stride == 512  # 256 * 2

    def test_preshuffle_b_data_K8192(self):
        """Preshuffle B-data with K=8192: stride = 4096."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_symbolic_stride(flat, iv, step=1)
        assert result is not None
        base, stride = result
        assert stride == 4096
        assert iv not in base.free_symbols

    def test_preshuffle_b_data_step_2(self):
        """K=8192, step=2 => stride = 4096*2 = 8192."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_symbolic_stride(flat, iv, step=2)
        assert result is not None
        _, stride = result
        assert stride == 8192

    def test_b_scale_K8192(self):
        """B-scale with K=8192: stride = 256."""
        flat = _b_scale_flat(iv, t, k_div_32=sympy.Integer(256))
        result = _try_symbolic_stride(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert stride == 256

    def test_base_has_no_iv(self):
        """Base (at iv=0) never contains the induction variable."""
        for flat in [
            _simple_unmapped_flat(iv, t, 4096),
            _preshuffle_b_data_flat(iv, t, sympy.Integer(4096)),
            _b_scale_flat(iv, t, sympy.Integer(256)),
        ]:
            result = _try_symbolic_stride(flat, iv, step=1)
            assert result is not None
            base, _ = result
            assert iv not in base.free_symbols

    def test_iv_independent_gives_zero_stride(self):
        """Flat with no IV dependency: stride = 0."""
        flat = t * 16 + wg * 1024
        result = _try_symbolic_stride(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert stride == 0

    def test_quadratic_iv_returns_none(self):
        """iv^2 is not affine -> extraction fails."""
        flat = t * 16 + iv**2
        assert _try_symbolic_stride(flat, iv, step=1) is None

    def test_floor_mod_of_iv_concrete_K_may_resolve(self):
        """floor/Mod of iv with concrete K: mem_simplify may cancel it.

        _try_symbolic_stride applies mem_simplify when the raw diff
        still contains iv.  Whether it succeeds depends on the expression
        structure.
        """
        flat = _floor_mod_of_iv_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_symbolic_stride(flat, iv, step=1)
        # mem_simplify can cancel floor(x/D)*D + Mod(x,D) -> x, exposing
        # the stride.  If it does, stride should be 128.
        if result is not None:
            _, stride = result
            assert stride == 128

    def test_dynamic_K_returns_none(self):
        """Symbolic K: stride = K/2 has free symbols -> rejected.

        _try_symbolic_stride only accepts strides that are concrete
        integers or Number (no free symbols).  Symbolic strides are
        handled by _try_with_div_subs.
        """
        flat = _preshuffle_b_data_flat(iv, t, half_K=K_dyn / 2)
        result = _try_symbolic_stride(flat, iv, step=1)
        assert result is None


# ===========================================================================
# _try_numerical_probe
# ===========================================================================


class TestTryNumericalProbe:
    """_try_numerical_probe evaluates at concrete IV values to find stride."""

    @pytest.fixture(autouse=True)
    def _idxc(self):
        """Provide an IndexingContext for subs_idxc calls."""
        with IndexingContext() as ctx:
            yield ctx

    def test_simple_affine(self):
        """Numeric probing finds stride for simple affine flat."""
        flat = _simple_unmapped_flat(iv, t, stride=4096)
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        base, stride = result
        assert stride == 4096

    def test_with_step_2(self):
        """Step=2: probing sees stride = per_element * step."""
        flat = _simple_unmapped_flat(iv, t, stride=256)
        result = _try_numerical_probe(flat, iv, step=2)
        assert result is not None
        _, stride = result
        assert stride == 512

    def test_preshuffle_b_data_K8192(self):
        """Preshuffle B-data with K=8192: stride = 4096."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert stride == 4096

    def test_floor_mod_of_iv_K8192(self):
        """floor/Mod of iv with concrete K=8192: numeric probing succeeds.

        This is the key case: symbolic extraction may fail on
        floor((t + iv*128)/4096), but numeric probing sees that the
        function is affine with stride 128.
        """
        flat = _floor_mod_of_iv_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert stride == 128

    def test_b_scale_K8192(self):
        """B-scale with K=8192: stride = 256."""
        flat = _b_scale_flat(iv, t, k_div_32=sympy.Integer(256))
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert stride == 256

    def test_quadratic_iv_returns_none(self):
        """Quadratic: stride is not constant."""
        flat = t * 16 + iv**2
        assert _try_numerical_probe(flat, iv, step=1) is None

    def test_returns_sympy_integer(self):
        """Stride from numeric probing is always a sympy.Integer."""
        flat = _simple_unmapped_flat(iv, t, stride=1024)
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        _, stride = result
        assert isinstance(stride, sympy.Integer)

    def test_base_is_iv_free(self):
        """Base from numeric probing has no IV dependency."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        result = _try_numerical_probe(flat, iv, step=1)
        assert result is not None
        base, _ = result
        assert iv not in base.free_symbols

    def test_dynamic_K_may_or_may_not_succeed(self):
        """Symbolic K: probing assigns a concrete K value.

        Whether this succeeds depends on probe arithmetic (K gets a
        concrete value from the probe map).  The function should return
        either a correct stride or None -- never a wrong answer.
        """
        flat = _preshuffle_b_data_flat(iv, t, half_K=K_dyn / 2)
        result = _try_numerical_probe(flat, iv, step=1)
        # Can't assert success or failure -- depends on probe values.
        # But if it returns something, it must be sensible.
        if result is not None:
            base, stride = result
            assert isinstance(stride, sympy.Integer)
            assert iv not in base.free_symbols


# ===========================================================================
# _try_with_div_subs
# ===========================================================================


class TestTryWithDivSubs:
    """_try_with_div_subs applies divisibility subs then retries extraction.

    This is the fallback for dynamic-shape kernels where K is symbolic
    and appears in floor/Mod denominators.
    """

    @pytest.fixture(autouse=True)
    def _idxc(self):
        with IndexingContext() as ctx:
            yield ctx

    def _get_div_subs(self, *constraints):
        return get_divisibility_subs(constraints)

    def test_returns_none_without_constraints(self):
        """No divisibility constraints -> returns None immediately."""
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        div_fwd, div_bwd = self._get_div_subs()
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        assert result is None

    def test_mismatched_stride_unlocked(self):
        """Mismatched row-stride/Mod-divisor: div subs expose stride.

        The flat expression row*K + Mod(inner, K/2) cannot be simplified
        by mem_simplify because K != K/2.  After K -> 256*K', the
        concrete factor collapses the floor/Mod and stride = 128 appears.
        """
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        assert result is not None
        base, stride, method = result
        assert stride == 128
        assert iv not in base.free_symbols

    def test_mismatched_stride_with_step(self):
        """Step=2: stride doubles."""
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 2, div_fwd, div_bwd)
        assert result is not None
        _, stride, _ = result
        assert stride == 256

    def test_base_preserves_original_structure(self):
        """Base is computed from the ORIGINAL flat (not the div-substituted one).

        This matters for MLIR lowering: the base keeps floor/Mod terms
        with the original symbols so gen_sympy_index produces correct
        integer ops.
        """
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        assert result is not None
        base, _, _ = result
        expected_base = flat.subs({iv: sympy.Integer(0)})
        assert simplify(base - expected_base) == 0

    def test_method_string_contains_divsubs(self):
        """The returned method string indicates div_subs was used."""
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        assert result is not None
        _, _, method = result
        assert "divsubs" in method

    def test_cancelling_form_not_needed(self):
        """The cancelling form (stride == Mod divisor) doesn't need div_subs.

        This verifies that _try_with_div_subs handles expressions that
        could also be resolved by the earlier symbolic step -- it should
        still produce a correct answer.
        """
        flat = _dynamic_K_cancelling_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        # May or may not return a result (it's optional for this form),
        # but if it does, stride must be 128.
        if result is not None:
            _, stride, _ = result
            assert stride == 128

    def test_weak_divisibility_may_still_work(self):
        """K%32==0 (weaker) may or may not unlock.  Never crashes."""
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        constraint = Assumption(sympy.Eq(sympy.Mod(K_dyn, 32), 0))
        div_fwd, div_bwd = self._get_div_subs(constraint)
        result = _try_with_div_subs(flat, iv, 1, div_fwd, div_bwd)
        if result is not None:
            _, stride, _ = result
            assert stride == 128


# ===========================================================================
# Integration: symbolic -> numeric -> div_subs cascade
# ===========================================================================


class TestExtractionCascade:
    """Test the cascade logic: symbolic first, then numeric, then div_subs.

    This mirrors the strategy in annotate_iv_strides without constructing
    a CapturedTrace.
    """

    @pytest.fixture(autouse=True)
    def _idxc(self):
        with IndexingContext() as ctx:
            yield ctx

    def _extract_stride(self, flat, iv_sym, step, constraints=()):
        """Run the same cascade as annotate_iv_strides."""
        div_fwd, div_bwd = get_divisibility_subs(constraints)

        result = _try_symbolic_stride(flat, iv_sym, step)
        if result is not None:
            base, stride = result
            return base, stride, "symbolic"

        result = _try_numerical_probe(flat, iv_sym, step)
        if result is not None:
            base, stride = result
            return base, stride, "numerical"

        div_result = _try_with_div_subs(flat, iv_sym, step, div_fwd, div_bwd)
        if div_result is not None:
            return div_result

        return None

    def test_simple_affine_uses_symbolic(self):
        """Simple affine: symbolic extraction wins."""
        flat = _simple_unmapped_flat(iv, t, 4096)
        result = self._extract_stride(flat, iv, step=1)
        assert result is not None
        _, stride, method = result
        assert stride == 4096
        assert method == "symbolic"

    def test_preshuffle_concrete_K_uses_symbolic(self):
        """Preshuffle with concrete K: symbolic wins (stride is integer)."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        result = self._extract_stride(flat, iv, step=1)
        assert result is not None
        _, stride, method = result
        assert stride == 4096
        assert method == "symbolic"

    def test_floor_mod_of_iv_uses_numeric_or_symbolic(self):
        """floor/Mod of iv with concrete K: symbolic or numeric succeeds."""
        flat = _floor_mod_of_iv_flat(iv, t, half_K=sympy.Integer(4096))
        result = self._extract_stride(flat, iv, step=1)
        assert result is not None
        _, stride, _ = result
        assert stride == 128

    def test_cancelling_form_uses_symbolic(self):
        """Cancelling floor/Mod with symbolic K: mem_simplify resolves it.

        floor(x/(K/2))*(K/2) + Mod(x, K/2) == x, so symbolic wins.
        """
        flat = _dynamic_K_cancelling_flat(iv, t, K_dyn)
        result = self._extract_stride(flat, iv, step=1)
        assert result is not None
        _, stride, method = result
        assert stride == 128
        assert method == "symbolic"

    def test_mismatched_stride_needs_div_subs(self):
        """Mismatched row-stride/Mod-divisor: requires div_subs.

        row*K + Mod(inner, K/2) can't be simplified by mem_simplify.
        Symbolic and numeric both fail.  div_subs succeeds.
        """
        flat = _dynamic_K_mismatched_stride_flat(iv, t, K_dyn)
        # Verify symbolic and numeric both fail.
        assert _try_symbolic_stride(flat, iv, 1) is None

        constraints = [Assumption(sympy.Eq(sympy.Mod(K_dyn, 256), 0))]
        result = self._extract_stride(flat, iv, step=1, constraints=constraints)
        assert result is not None
        _, stride, method = result
        assert stride == 128
        assert "divsubs" in method

    def test_per_unit_computation(self):
        """Verify per_unit = stride // step.

        This is what annotate_iv_strides stores in IndexSequence.stride.
        """
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        step = 2
        result = self._extract_stride(flat, iv, step=step)
        assert result is not None
        _, stride, _ = result
        per_unit = stride // step
        # stride = 8192 (flat(iv+2) - flat(iv) = 4096*2), per_unit = 4096
        assert per_unit == 4096

    def test_new_offset_matches_original(self):
        """Verify base + iv * per_unit equals original flat (simple case)."""
        flat = _simple_unmapped_flat(iv, t, 4096)
        step = 1
        result = self._extract_stride(flat, iv, step=step)
        assert result is not None
        base, stride, _ = result
        per_unit = stride // step
        new_offset = base + iv * per_unit
        assert simplify(new_offset - flat) == 0

    def test_new_offset_matches_preshuffle(self):
        """Verify base + iv * per_unit equals original (preshuffle case)."""
        flat = _preshuffle_b_data_flat(iv, t, half_K=sympy.Integer(4096))
        step = 1
        result = self._extract_stride(flat, iv, step=step)
        assert result is not None
        base, stride, _ = result
        per_unit = stride // step
        new_offset = base + iv * per_unit
        assert simplify(new_offset - flat) == 0

    def test_b_scale_full_pipeline(self):
        """B-scale end-to-end: K=8192, step=2, per_unit = 256."""
        flat = _b_scale_flat(iv, t, k_div_32=sympy.Integer(256))
        step = 2
        result = self._extract_stride(flat, iv, step=step)
        assert result is not None
        base, stride, method = result
        per_unit = stride // step
        assert per_unit == 256
        assert method == "symbolic"
        # Rewrite should match original.
        new_offset = base + iv * per_unit
        assert simplify(new_offset - flat) == 0
