# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar
from copy import deepcopy

import sympy
import torch.fx as fx

from ..._support.indexing import IndexingContext
from ...lang.wave_types import IndexMapping
from .general_utils import infer_dim, get_fastest_index
from .symbol_utils import IndexExpr, IndexSequence, IndexSymbol, simplify, subs_idxc
from ....support.indexing import piecewise_aware_subs
from ...compiler.utils import strides_from_symbolic_shape

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


def get_dict_with_updated_key(
    original_dict: dict[K, V], old_key: K, new_key: K
) -> dict[K, V]:
    """
    Update a key in a dictionary while preserving the original insertion order of values.

    Creates a new dictionary identical to the original except that the specified old key
    is replaced with the new key. All values and ordering remain unchanged.
    """
    if old_key not in original_dict:
        raise KeyError(f"Old key '{old_key}' not found in dictionary")
    if new_key in original_dict and new_key != old_key:
        raise KeyError(f"New key '{new_key}' already exists in dictionary")

    # Create a new dictionary with the same order but updated key
    new_dict = {}
    for key, value in original_dict.items():
        if key == old_key:
            new_dict[new_key] = value
        else:
            new_dict[key] = value

    return new_dict


def approximate_difference(
    expr: IndexExpr, vars: list[IndexSymbol], elements_per_thread: int
) -> bool:
    """
    During the contiguity check, we take a unit step in the fastest changing
    dimension (j -> j + 1) and we compute f(j + 1) - f(j) to see if it is 1.
    In general, we will end up with expressions of the form
    g(x + eps) - g(x) where x = h(j) and eps is a rational of the form 1/q.
    We can use q to determine if the mapping is contiguous as follows

    if q is divisible by elements_per_thread (dimensions where we have not applied the unit step), or
    if eps is 1 (corresponds to the dimension where we have applied the unit step)
    then the mapping is contiguous.

    The mapping function f(j) will be non-linear in general, and so the difference
    of 1 will be transformed to different constant values based on the function.
    But, if we recover a value of 1, we can assume that the function preserves
    the difference.

    In this function we do a pre-order traversal of the expression to obtain
    the value of the constant eps.
    """
    if expr.is_number:
        return expr
    new_vars, new_exprs = sympy.cse(expr)
    new_expr = new_exprs[0] if new_vars else expr
    new_vars = [x[0] for x in new_vars] if new_vars else vars
    for arg in sympy.preorder_traversal(new_expr):
        if isinstance(arg, sympy.Add):
            if all([x in arg.args for x in new_vars]):
                constant = [x for x in arg.args if x not in new_vars][0]
                if not isinstance(constant, sympy.Rational):
                    return expr
                if constant.p != 1:
                    return expr
                if constant.q == 1:
                    return 1
                return 0 if constant.q % elements_per_thread == 0 else expr
    return expr


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def check_is_dynamic_vals_broadcasted(nodes: list[fx.Node]) -> bool:
    """
    Check if dynamic values in a list of nodes are broadcasted.

    A dynamic value is considered broadcasted if its index has size 1 in all dimensions.
    This function checks all nodes in the list and returns True only if all dynamic values
    are broadcasted (size 1 in all dims).
    """
    for node in nodes:
        index = node.index
        assert index is not None, f"Node {node} has no index"
        if any(subs_idxc(i.size) > 1 for i in index.values()):
            return False
    return True


def check_is_mapping_contiguous(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    array_shape: tuple[IndexExpr, ...],
    index: tuple[IndexExpr, ...],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Check if mapping can be lowered to contiguous vector ops instead of gathers/scatters"""
    elements_per_thread = subs_idxc(elements_per_thread)
    if elements_per_thread == 1:
        return True

    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        symbolic_dims = [infer_dim(dim_size) for dim_size in symbolic_shape]
        index_mapping = mapping.map_input_indices(symbolic_dims)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolic_shape)

    index_mapping = tuple(subs_idxc(i) for i in index_mapping)
    iters = mapping.iters

    subs = {sym: sym + int(i == len(iters) - 1) for i, sym in enumerate(iters)}
    diff = [
        approximate_difference(
            piecewise_aware_subs(index_mapping[i], subs) - index_mapping[i],
            list(iters.keys())[-1:],
            elements_per_thread,
        )
        for i in range(len(index_mapping))
    ]

    expected_diff = [0] * len(index_mapping)
    expected_diff[-1] = 1
    if expected_diff == diff:
        return True

    # If the expected pattern is not found, check if the mapping still produces
    # contiguous memory accesses by computing offsets for each element.
    #
    # The check works by:
    # 1. Computing the linear memory offset symbolically for the first element
    # 2. For each subsequent element:
    #    - Updating the fastest dimension in the index for that element
    #    - Transforming through the mapping
    #    - Computing the new memory offset
    #    - Verifying the offset increased by exactly 1
    # 3. Returns True only if all elements are sequential
    fastest_dim = list(index.keys())[get_fastest_index(index)]

    idxc = IndexingContext.current()
    strides = strides_from_symbolic_shape(idxc, array_shape, allow_mixed_shapes=True)
    new_index = transform_index_on_mapping(
        mapping, symbolic_shape, index, is_read=is_read
    )
    prev_offset = _compute_offset(
        [new_index[infer_dim(d)] for d in symbolic_shape], strides
    )
    for i in range(1, elements_per_thread):
        new_index = deepcopy(index)
        new_index[fastest_dim].start += i
        new_index = transform_index_on_mapping(
            mapping, symbolic_shape, new_index, is_read=is_read
        )
        offset = _compute_offset(
            [new_index[infer_dim(d)] for d in symbolic_shape], strides
        )

        # When the sympy expressions are complex, we fallback to the aligned-base check.
        # Because sympy evaluates to a value, we cannot use the difference check. Hence, the check fails.
        if (offset - prev_offset) != 1:
            return _check_contiguous_with_aligned_base(
                mapping,
                symbolic_shape,
                array_shape,
                index,
                elements_per_thread,
                is_read,
            )

        prev_offset = offset

    return True


def _check_contiguous_with_aligned_base(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    array_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Aligned-base contiguity check for complex floor/Mod mappings.

    Thread indices in the fastest dimension are distributed in chunks of
    elements_per_thread, so the base is always a multiple of that value.
    Encoding this alignment (base = _aligned * elements_per_thread) lets
    sympy resolve sub-expressions such as Mod(16*k + i, 16) -> i and
    floor(k/2 + i/32) → floor(k/2) that the generic check cannot simplify.
    It is independent of the tensor shape and depends on the thread distributuion.
    """
    fastest_dim = list(index.keys())[get_fastest_index(index)]
    _aligned = sympy.Symbol("_aligned", integer=True, nonnegative=True)

    idxc = IndexingContext.current()
    strides = strides_from_symbolic_shape(idxc, array_shape, allow_mixed_shapes=True)

    def _make_aligned_index(offset_val: int) -> dict[IndexExpr, IndexSequence]:
        idx = deepcopy(index)
        idx[fastest_dim].start = _aligned * elements_per_thread + offset_val
        return idx

    new_index = transform_index_on_mapping(
        mapping,
        symbolic_shape,
        _make_aligned_index(0),
        is_read=is_read,
    )
    prev_offset = _compute_offset(
        [new_index[infer_dim(d)] for d in symbolic_shape],
        strides,
    )
    for i in range(1, elements_per_thread):
        new_index = transform_index_on_mapping(
            mapping,
            symbolic_shape,
            _make_aligned_index(i),
            is_read=is_read,
        )
        offset = _compute_offset(
            [new_index[infer_dim(d)] for d in symbolic_shape],
            strides,
        )
        diff_expr = simplify(offset - prev_offset)
        if diff_expr != 1:
            return False
        prev_offset = offset

    return True


def _expand_mod(expr: sympy.Expr) -> sympy.Expr:
    """Rewrite ``Mod(x, d)`` as ``x - d*floor(x/d)``.

    This lets ``floor(x/d)*d + Mod(x, d)`` cancel to ``x`` via normal
    algebraic simplification, which SymPy cannot do when ``Mod`` is
    kept as an opaque node.
    """
    if not expr.has(sympy.Mod):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sympy.Mod),
        lambda e: e.args[0] - e.args[1] * sympy.floor(e.args[0] / e.args[1]),
    )


def _eval_concrete_floor_mod(expr: sympy.Expr) -> sympy.Expr:
    """Evaluate ``floor`` and ``Mod`` nodes whose arguments are concrete.

    SymPy sometimes leaves ``floor(4)`` or ``Mod(0, K)`` un-evaluated
    when they were built symbolically and then substituted.  This pass
    collapses them bottom-up so the result contains no unnecessary
    wrappers around known integers.
    """

    def _try_eval(e):
        if isinstance(e, sympy.floor):
            inner = e.args[0]
            if inner.is_number:
                return sympy.Integer(int(sympy.floor(inner)))
        if isinstance(e, sympy.Mod):
            a, m = e.args
            if a.is_Integer and m.is_Integer and int(m) != 0:
                return sympy.Integer(int(a) % int(m))
            if a.is_Integer and int(a) == 0:
                return sympy.Integer(0)
        return e

    if not expr.has(sympy.floor) and not expr.has(sympy.Mod):
        return expr
    return expr.replace(
        lambda e: isinstance(e, (sympy.floor, sympy.Mod)),
        _try_eval,
    )


def mem_simplify(expr: sympy.Expr) -> sympy.Expr:
    """Simplify an expression representing a tensor memory index.

    Domain-specific alternative to ``sympy.simplify`` that applies only
    the algebraic identities relevant to integer memory addressing:

    1. Rewrite ``Mod(x, d)`` as ``x - d*floor(x/d)`` so that paired
       floor terms cancel algebraically.
    2. ``expand()`` to distribute products and cancel matching terms —
       this collapses the fundamental round-trip
       ``floor(E/D)*D + Mod(E, D) → E``.
    3. Evaluate any ``floor`` or ``Mod`` of concrete integers left over.
    """
    expr = sympy.sympify(expr)
    if expr.is_number:
        return expr
    if expr.has(sympy.Mod):
        expr = _expand_mod(expr)
    expr = sympy.expand(expr)
    expr = _eval_concrete_floor_mod(expr)
    return expr


def linearize_dims(
    dim_exprs: list[sympy.Expr],
    strides: list[sympy.Expr],
) -> sympy.Expr:
    """Compute ``sum(dim_i * stride_i)`` with floor/Mod cancellation.

    Preshuffle mappings produce ``floor(flat/D)`` for one dimension and
    ``Mod(flat, D)`` for another, with the first dimension's stride
    equal to ``D``.  The naive sum keeps SymPy's opaque ``Mod`` node
    and ``simplify`` cannot prove the cancellation.

    By expanding ``Mod`` to its floor definition first and then calling
    ``expand``, the matching ``floor`` terms cancel algebraically::

        floor(E/D)*D + Mod(E,D)
        → floor(E/D)*D + E - D*floor(E/D)
        → E
    """
    total = sum(d * s for d, s in zip(dim_exprs, strides))
    return mem_simplify(total)


def _infer_floor_to_exact(mem_strides: list[IndexExpr]) -> dict:
    """Infer ``floor(sym/n) → sym/n`` substitutions from memory strides.

    When a memory stride has the form ``sym/n`` (symbolic numerator over
    a positive integer denominator), the stride must be a non-negative
    integer for the layout to be physically valid.  That forces the
    numerator to be a multiple of the denominator, so
    ``floor(sym/n) == sym/n``.

    Substituting this into dimension expressions collapses
    ``floor(E / floor(sym/n)) * (sym/n) + Mod(E, floor(sym/n))`` back
    into ``E`` via the standard floor/Mod identity, eliminating the
    symbolic divisor that the probing cannot handle.
    """
    subs_map: dict[sympy.Expr, sympy.Expr] = {}
    for stride in mem_strides:
        stride = sympy.sympify(stride)
        numer, denom = stride.as_numer_denom()
        if denom.is_Integer and int(denom) > 1 and numer.free_symbols:
            exact_quot = numer / denom
            subs_map[sympy.floor(exact_quot)] = exact_quot
    return subs_map


def transform_index_on_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    is_read: bool = True,
) -> dict[IndexExpr, IndexSequence]:
    """Transforms the index according to the specified mapping"""
    symbolic_shape = tuple(infer_dim(d) for d in symbolic_shape)
    if is_read:
        index_mapping = mapping.map_input_indices(symbolic_shape)
    else:
        index_mapping = mapping.map_output_indices(symbolic_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(piecewise_aware_subs(i, idxc.subs) for i in index_mapping)
    iters = mapping.iters
    subs = dict(
        list(zip(iters.keys(), (expr.start for expr in index.values())))
        + list(idxc.subs.items())
    )
    transformed_index = {
        key: piecewise_aware_subs(m, subs)
        for key, m in zip(symbolic_shape, index_mapping)
    }

    return transformed_index
