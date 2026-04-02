# Copyright 2026 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Annotate IV strides on flattened Read/GatherToLDS ops.

For each Read/GatherToLDS with a ``LINEAR_INDEX`` inside a loop body, this
pass extracts the IV stride.  It tries symbolic differencing first:

    stride = simplify(flat(IV + step) - flat(IV))

If that fails (e.g. complex preshuffle mappings with floor/Mod), it falls
back to numerical probing via ``_probe_iv_stride_from_flat``.

When the stride is successfully extracted, the flat offset is rewritten
to ``base + IV * stride`` form and the stride is stored in
``IndexSequence.stride``.

This pass runs after ``merge_contiguous_reads`` (which may double the ept
but preserves the LINEAR_INDEX structure) and before codegen.
"""

import sympy

from ..._support.indexing import IndexSequence
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import LINEAR_INDEX
from ...ops.wave_ops import Iterate, Read, GatherToLDS, get_custom
from ..assumptions import get_divisibility_subs
from ..utils.general_utils import is_flattened_index, get_flat_offset
from ..utils.mapping_utils import mem_simplify
from ..utils.symbol_utils import (
    get_induction_symbol,
    safe_subs,
    simplify as sym_simplify,
    subs_idxc,
)


def _try_symbolic_stride(flat, iv_sym, step):
    """Try to extract IV stride via symbolic differencing.

    Returns ``(base, stride)`` if the flat offset is affine in *iv_sym*,
    ``None`` otherwise.
    """
    shifted = safe_subs(flat, {iv_sym: iv_sym + step})
    stride = sym_simplify(shifted - flat)

    if iv_sym in stride.free_symbols:
        stride = mem_simplify(stride)
        if iv_sym in stride.free_symbols:
            return None

    if not (stride.is_Integer or stride.is_Number):
        if stride.free_symbols:
            return None

    base = sym_simplify(safe_subs(flat, {iv_sym: sympy.Integer(0)}))
    return base, stride


def _try_numerical_probe(flat, iv_sym, step):
    """Fall back to numerical probing for IV stride extraction.

    Evaluates the flat address at several concrete IV values and checks
    whether the stride (addr[iv+1] - addr[iv]) is constant.

    Returns ``(base, stride)`` or ``None``.
    """
    iv_flat = flat.subs({iv_sym: step * sympy.Symbol("_iv_probe")})
    iv_flat = subs_idxc(iv_flat)

    free_syms = sorted(iv_flat.free_symbols - {sympy.Symbol("_iv_probe")}, key=str)
    probe_map = {s: 137 + i * 31 for i, s in enumerate(free_syms)}

    def _eval(iv_val):
        expr = flat.subs({iv_sym: step * iv_val})
        expr = subs_idxc(expr)
        expr = expr.subs(probe_map)
        try:
            return int(expr)
        except (TypeError, ValueError):
            return None

    n_probes = 8
    addrs = [_eval(i) for i in range(n_probes + 1)]
    if any(a is None for a in addrs):
        return None

    diffs = [addrs[i + 1] - addrs[i] for i in range(n_probes)]
    if len(set(diffs)) != 1:
        return None

    stride_val = diffs[0]

    probe_map2 = {s: 251 + i * 47 for i, s in enumerate(free_syms)}

    def _eval2(iv_val):
        expr = flat.subs({iv_sym: step * iv_val})
        expr = subs_idxc(expr)
        expr = expr.subs(probe_map2)
        try:
            return int(expr)
        except (TypeError, ValueError):
            return None

    addrs2 = [_eval2(i) for i in range(n_probes + 1)]
    if any(a is None for a in addrs2):
        return None
    diffs2 = [addrs2[i + 1] - addrs2[i] for i in range(n_probes)]
    if len(set(diffs2)) != 1 or diffs2[0] != stride_val:
        return None

    base = sym_simplify(safe_subs(flat, {iv_sym: sympy.Integer(0)}))
    return base, sympy.Integer(stride_val)


def _try_with_div_subs(flat, iv_sym, step, div_fwd, div_bwd):
    """Try stride extraction after applying divisibility substitutions.

    Mapped-read flat expressions contain ``floor(K/d)`` / ``Mod(K, d)``
    terms that block both symbolic and numerical probing.  Divisibility
    constraints (e.g. ``K % 256 == 0``) let us replace ``K`` with
    ``256 * _K_div_256``, collapsing those terms.

    The stride is extracted from the simplified expression but the base
    is computed from the ORIGINAL flat to preserve floor/Mod structure
    needed for correct MLIR integer lowering.
    """
    if not div_fwd:
        return None

    fwd_dict = dict(div_fwd)
    simplified = sympy.sympify(flat).subs(fwd_dict)

    result = _try_symbolic_stride(simplified, iv_sym, step)
    method = "symbolic+divsubs" if result is not None else None

    if result is None:
        result = _try_numerical_probe(simplified, iv_sym, step)
        method = "numerical+divsubs" if result is not None else None

    if result is None:
        return None

    _, stride = result

    if div_bwd:
        bwd_dict = dict(div_bwd)
        stride = mem_simplify(sympy.sympify(stride).subs(bwd_dict))

    base = safe_subs(flat, {iv_sym: sympy.Integer(0)})
    return base, stride, method


def annotate_iv_strides(
    trace: CapturedTrace,
    constraints=(),
):
    """Rewrite flattened Read/GatherToLDS indices to ``base + IV * stride`` form.

    Walks all subgraphs; for those inside Iterate ops, identifies the IV
    and for each flattened read tries to extract the stride.
    """
    div_fwd, div_bwd = get_divisibility_subs(constraints)

    for subgraph in trace.region_graph.subgraphs.values():
        parent_node = getattr(subgraph, "parent_op", None)
        if parent_node is None:
            continue
        parent = get_custom(parent_node)
        if not isinstance(parent, Iterate):
            continue

        iv_sym = get_induction_symbol(parent.axis)
        step = parent.step if parent.step is not None else 1

        for node in subgraph.nodes:
            custom = get_custom(node)
            if not isinstance(custom, (Read, GatherToLDS)):
                continue

            is_g2l = isinstance(custom, GatherToLDS)
            index = custom.src_index if is_g2l else custom.index

            if not is_flattened_index(index):
                continue

            flat = get_flat_offset(index)
            if iv_sym not in flat.free_symbols:
                continue

            method = None
            result = _try_symbolic_stride(flat, iv_sym, step)
            if result is not None:
                method = "symbolic"
                base, stride = result
            else:
                result = _try_numerical_probe(flat, iv_sym, step)
                if result is not None:
                    method = "numerical"
                    base, stride = result

            if result is None:
                div_result = _try_with_div_subs(
                    flat,
                    iv_sym,
                    step,
                    div_fwd,
                    div_bwd,
                )
                if div_result is not None:
                    base, stride, method = div_result

            if result is None and method is None:
                continue

            ept = index[LINEAR_INDEX].size
            per_unit = stride // step
            new_offset = base + iv_sym * per_unit
            new_index = {LINEAR_INDEX: IndexSequence(new_offset, ept, per_unit)}

            if is_g2l:
                custom.src_index = new_index
            else:
                custom.index = new_index
