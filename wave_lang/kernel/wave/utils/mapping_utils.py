# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TypeVar
from copy import deepcopy
from math import gcd, lcm

import sympy
import torch.fx as fx

from collections.abc import Sequence

from ..._support.indexing import IndexingContext
from ...lang.wave_types import IndexMapping
from ..assumptions import get_divisibility_subs
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


_INDUCTION_PREFIX = "$ARG"


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


def compute_iv_stride_through_mapping(
    mapping: IndexMapping,
    symbolic_shape: tuple[IndexExpr, ...],
    index: dict[IndexExpr, IndexSequence],
    is_read: bool = True,
    mem_strides: list[IndexExpr] | None = None,
    constraints: Sequence = (),
) -> dict[sympy.Symbol, IndexExpr | list[IndexExpr]] | None:
    """Compute each IV symbol's linearized stride through a mapping.

    Uses numerical probing: evaluates the linearized address at iv=0,1,...,P,
    takes consecutive differences, and detects constant or cyclic stride
    patterns.  This handles symbolic divisors (e.g. ``floor(K/32)``) that
    defeat the old ``sympy.coeff(_iv)`` approach.

    Parameters
    ----------
    mem_strides : optional physical memory strides.  When provided these are
        used for linearization instead of strides derived from
        ``symbolic_shape``.  Callers with access to the physical memory
        layout should pass these to ensure floor/Mod cancellation.
    constraints : constraint sequence (may include ``Assumption`` objects).
        Divisibility assumptions (e.g. ``Assumption(Eq(K % 256, 0))``)
        are used to simplify floor/Mod expressions before probing.

    Returns ``{iv_sym: stride}`` (constant) or ``{iv_sym: [s0, s1, ...]}``
    (repeating cycle), or ``None`` when no IV is found.
    """
    iters = mapping.iters

    iv_info: dict[sympy.Symbol, tuple[sympy.Symbol, int]] = {}
    for dim_sym, iter_sym in mapping.output_mapping.items():
        seq = index.get(dim_sym)
        if seq is None:
            continue
        start = sympy.sympify(seq.start if isinstance(seq, IndexSequence) else seq)
        for sym in start.free_symbols:
            if not str(sym).startswith(_INDUCTION_PREFIX):
                continue
            coeff = sympy.expand(start).coeff(sym)
            concrete = subs_idxc(coeff)
            if not isinstance(concrete, (int, sympy.Integer)):
                print(
                    f"IV coeff for {sym} is non-concrete: coeff={coeff}"
                    f"  resolved={concrete} (type={type(concrete).__name__})"
                    f" — skipping this IV"
                )
                continue
            iv_info[sym] = (iter_sym, int(concrete))

    if not iv_info:
        return None

    print(f"=== compute_iv_stride_through_mapping  is_read={is_read} ===")
    print(f"  iters: {dict(iters)}")
    for iv_sym, (iv_iter, cc) in iv_info.items():
        print(f"  IV {iv_sym} -> iter={iv_iter}  coeff={cc}")

    map_dims = mapping.input_shape if is_read else mapping.output_shape
    raw_exprs = (
        mapping.map_input_indices(map_dims)
        if is_read
        else mapping.map_output_indices(map_dims)
    )

    idxc = IndexingContext.current()
    dim_exprs = [subs_idxc(e) for e in raw_exprs]

    for i, (raw, resolved) in enumerate(zip(raw_exprs, dim_exprs)):
        changed = str(raw) != str(resolved)
        print(
            f"  dim[{i}]  raw={raw}  ->  resolved={resolved}"
            f"{'  (CHANGED by subs_idxc)' if changed else ''}"
        )

    if mem_strides is None:
        symbolic_shape_resolved = tuple(infer_dim(d) for d in symbolic_shape)
        mem_strides = strides_from_symbolic_shape(
            idxc, symbolic_shape_resolved, allow_mixed_shapes=True
        )

    stride_free = set()
    for s in mem_strides:
        stride_free |= sympy.sympify(s).free_symbols
    print(
        f"  mem_strides={mem_strides}"
        f"  (symbolic={sorted(str(s) for s in stride_free) if stride_free else 'none'})"
    )

    div_fwd, div_bwd = get_divisibility_subs(constraints)
    if div_fwd:
        fwd_dict = dict(div_fwd)
        dim_exprs = [sympy.sympify(e).subs(fwd_dict) for e in dim_exprs]
        mem_strides = [sympy.sympify(s).subs(fwd_dict) for s in mem_strides]
        print(f"  divisibility fwd subs: {fwd_dict}")
        for i, e in enumerate(dim_exprs):
            print(f"  dim_after_div_subs[{i}] = {e}")
        print(f"  mem_strides_after_div_subs={mem_strides}")
    else:
        floor_subs = _infer_floor_to_exact(mem_strides)
        if floor_subs:
            dim_exprs = [sympy.sympify(e).subs(floor_subs) for e in dim_exprs]
            print(f"  floor_to_exact subs (fallback): {floor_subs}")
            for i, e in enumerate(dim_exprs):
                print(f"  dim_after_subs[{i}] = {e}")

    result: dict[sympy.Symbol, IndexExpr | list[IndexExpr]] = {}

    for iv_sym, (iv_iter, concrete_coeff) in iv_info.items():
        stride_or_cycle = _probe_iv_stride(
            dim_exprs, mem_strides, iters, iv_iter, concrete_coeff
        )
        if stride_or_cycle is None:
            print(
                f"  _probe_iv_stride returned None for IV {iv_sym}"
                f" — no pattern detected, returning None for entire mapping"
            )
            return None
        result[iv_sym] = stride_or_cycle

    if div_bwd:
        bwd_dict = dict(div_bwd)

        def _bwd(v):
            if isinstance(v, list):
                return [mem_simplify(sympy.sympify(x).subs(bwd_dict)) for x in v]
            return mem_simplify(sympy.sympify(v).subs(bwd_dict))

        result = {k: _bwd(v) for k, v in result.items()}

    for iv_sym, val in result.items():
        print(f"  RESULT  {iv_sym} -> {val}")

    return result


def _extract_integer_divisors(expr: sympy.Expr) -> set[int]:
    """Collect positive integer divisors from ``floor`` and ``Mod`` nodes."""
    divisors: set[int] = set()
    for sub in sympy.preorder_traversal(expr):
        if isinstance(sub, sympy.Mod):
            d = sub.args[1]
            if d.is_Integer and int(d) > 0:
                divisors.add(int(d))
        elif isinstance(sub, sympy.floor):
            _, denom = sub.args[0].as_numer_denom()
            if denom.is_Integer and int(denom) > 1:
                divisors.add(int(denom))
    return divisors


_MAX_PROBE_DEPTH = 1024


def _compute_probe_depth(flat_expr: sympy.Expr, concrete_coeff: int) -> int:
    """Compute the exact probe depth from divisors in *flat_expr*.

    For each ``floor(expr/D)`` or ``Mod(expr, D)`` with integer D, the IV
    contribution has period ``D / gcd(C, D)`` where ``C`` is the IV step.
    The overall period ``P = LCM(all periods)`` is the exact number of
    diffs needed to capture the full stride pattern.

    Raises ``ValueError`` if the computed depth exceeds ``_MAX_PROBE_DEPTH``,
    which indicates a pathological mapping with too many coprime divisors.
    """
    if concrete_coeff == 0:
        return 1
    divisors = _extract_integer_divisors(flat_expr)
    if not divisors:
        return 1
    C = abs(concrete_coeff)
    periods = [d // gcd(C, d) for d in divisors]
    depth = lcm(*periods)
    if depth > _MAX_PROBE_DEPTH:
        raise ValueError(
            f"Probe depth {depth} exceeds maximum {_MAX_PROBE_DEPTH}"
            f" (divisors={sorted(divisors)}, C={C})."
            f" The mapping has too many coprime floor/Mod divisors"
            f" for exact stride analysis."
        )
    return depth


def _probe_iv_stride(
    dim_exprs: list[IndexExpr],
    mem_strides: list[IndexExpr],
    iters: dict,
    iv_iter: sympy.Symbol,
    concrete_coeff: int,
) -> IndexExpr | list[IndexExpr] | None:
    """Compute the IV stride through the linearized address.

    1. Linearize ``dim_exprs * mem_strides`` symbolically via
       ``linearize_dims`` + ``mem_simplify`` to obtain the flat address
       expression (cancelling floor/Mod round-trip pairs).
    2. Compute the exact probe depth ``P`` from the remaining integer
       divisors in the flat expression: ``P = LCM(D_i / gcd(C, D_i))``.
    3. Evaluate ``P + 1`` concrete addresses and detect constant or
       cyclic stride patterns from the integer diffs.

    If the addresses contain free symbols, this is an error — the
    substitution context has unresolved chained dependencies.

    Returns a single IndexExpr (constant stride), a list of IndexExpr
    (repeating cycle), or ``None`` on failure.
    """

    print(f"_probe_iv_stride  iv_iter={iv_iter}  coeff={concrete_coeff}")

    # Step 1: linearize symbolically, then compute probe depth from the
    # flat expression's divisors.  Apply subs_idxc to iv_flat so the
    # probe-depth computation sees the same symbol resolution as the
    # concrete address evaluations (prevents under-probing when a
    # divisor is symbolic pre-subs but integer post-subs).
    flat_expr = mem_simplify(linearize_dims(dim_exprs, mem_strides))
    iv_flat = flat_expr.subs(
        {
            it: (concrete_coeff * sympy.Symbol("_iv") if it == iv_iter else 0)
            for it in iters.keys()
        }
    )
    iv_flat = subs_idxc(iv_flat)
    probe_depth = _compute_probe_depth(iv_flat, concrete_coeff)

    print(f"  probe_depth={probe_depth}")

    # Step 2: evaluate P+1 concrete addresses.
    def _linearized_addr(iv_val: int) -> IndexExpr:
        subs = {
            it: (concrete_coeff * iv_val if it == iv_iter else 0) for it in iters.keys()
        }
        resolved = [subs_idxc(dim_expr.subs(subs)) for dim_expr in dim_exprs]
        return mem_simplify(subs_idxc(linearize_dims(resolved, mem_strides)))

    addrs: list[int] = []
    for iv in range(probe_depth + 1):
        a = _linearized_addr(iv)
        if getattr(a, "free_symbols", set()):
            print(
                f"  addr[iv={iv}] = {a}  (free={a.free_symbols})"
                f"\n  *** ERROR: address contains unresolved free symbols."
                f"  Fix chained symbolic dependencies upstream."
            )
            return None
        addrs.append(int(a))

    diffs = [addrs[i + 1] - addrs[i] for i in range(probe_depth)]

    for i, a in enumerate(addrs):
        print(f"  addr[iv={i}] = {a}")
    for i, d in enumerate(diffs):
        print(f"  diff[{i}] = {d}")

    if not diffs:
        return None

    # Step 3: detect constant or shortest repeating cycle.
    # The probe depth is the analytically-proven period, so one full
    # repetition is sufficient — use range(1, probe_depth + 1).
    for cycle_len in range(1, probe_depth + 1):
        if all(diffs[i] == diffs[i % cycle_len] for i in range(probe_depth)):
            cycle = [sympy.Integer(diffs[i]) for i in range(cycle_len)]
            if cycle_len == 1:
                print(
                    f"  -> CONSTANT stride = {cycle[0]}"
                    f"  (concrete=True, probe_depth={probe_depth})"
                )
                return cycle[0]
            print(
                f"  -> CYCLIC stride (len={cycle_len}): {diffs[:cycle_len]}"
                f"  (concrete=True, probe_depth={probe_depth})"
            )
            return cycle

    print(
        f"  -> FAILED: no constant or cyclic pattern in {probe_depth}"
        f" diffs.  diffs={diffs}"
    )
    return None


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
