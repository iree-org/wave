# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""
Fuzz tests for ixsimpl using Hypothesis.

Properties tested:
1. Self-consistency: simplification preserves numerical semantics.
2. Cross-check: ixsimpl agrees with SymPy on random expressions.
3. Divisibility: simplification with Mod(sym, d)==0 preserves semantics
   at evaluation points satisfying the assumption.
4. to_sympy semantics: ixsimpl.sympy_conv.to_sympy produces SymPy
   expressions that agree numerically with ixsimpl evaluation.
5. from_sympy semantics: ixsimpl.sympy_conv.from_sympy produces ixsimpl
   expressions that agree numerically with the original tree.
6. Roundtrip: ixsimpl -> to_sympy -> from_sympy -> ixsimpl preserves
   numerical semantics.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from typing import Any

import ixsimpl
import sympy
from hypothesis import assume, example, given
from hypothesis import strategies as st
from ixsimpl.sympy_conv import from_sympy as conv_from_sympy
from ixsimpl.sympy_conv import to_sympy as conv_to_sympy

ExprTree = str | int | tuple[Any, ...]
CondTree = tuple[Any, ...]
Env = dict[str, int]

_VARS = ["x", "y", "z", "w", "a", "b", "c", "d"]


def _env_from_val(val_st: st.SearchStrategy[int]) -> st.SearchStrategy[Env]:
    """Build an env strategy that draws each variable from val_st."""
    return st.fixed_dictionaries({v: val_st for v in _VARS})


def _signed(base: st.SearchStrategy[int]) -> st.SearchStrategy[int]:
    """Draw from base or its negation with equal probability."""
    return st.one_of(base, base.map(lambda x: -x))


def _env_st(lo: int = 1, hi: int = 100) -> st.SearchStrategy[Env]:
    """Env with each variable uniform in [lo, hi]."""
    return _env_from_val(st.integers(lo, hi))


def _wide_env_st() -> st.SearchStrategy[Env]:
    """Env mixing negative, zero, and positive values."""
    return _env_from_val(st.one_of(st.integers(-100, -1), st.just(0), st.integers(1, 100)))


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 127, 251, 509, 1021]
_POW2 = [1 << k for k in range(1, 16)]
_POW2_ADJ = [v + d for v in _POW2 for d in (-1, 1)]
_INTERESTING = sorted(set(_PRIMES + _POW2 + _POW2_ADJ + [0, 1, -1]))


def _spicy_env_st() -> st.SearchStrategy[Env]:
    """Env with primes, powers of 2, pow2 +/- 1, 0, and +/-1."""
    return _env_from_val(_signed(st.sampled_from(_INTERESTING)))


def _prime_env_st() -> st.SearchStrategy[Env]:
    """Env biased toward primes (with optional negation)."""
    return _env_from_val(_signed(st.sampled_from(_PRIMES)))


def _pow2_env_st() -> st.SearchStrategy[Env]:
    """Env biased toward powers of 2 and pow2 +/- 1."""
    return _env_from_val(_signed(st.sampled_from(sorted(set(_POW2 + _POW2_ADJ)))))


def _mixed_env_st() -> st.SearchStrategy[Env]:
    """Env blending uniform [0, 100], wide, and spicy values."""
    return st.one_of(_env_st(0, 100), _wide_env_st(), _spicy_env_st())


# ---------------------------------------------------------------------------
#  Expression tree strategies
# ---------------------------------------------------------------------------

sym_names = st.sampled_from(_VARS)
small_ints = st.integers(min_value=-64, max_value=64)
pos_ints = st.integers(min_value=1, max_value=32)
small_rats = st.tuples(st.integers(min_value=-64, max_value=64), pos_ints).map(
    lambda pq: ("rat", pq[0], pq[1])
)


_OPS_BASE = ["add", "sub", "mul", "neg", "div", "floor", "ceiling", "mod", "max", "min", "xor"]
_OPS_WITH_PW = [*_OPS_BASE, "piecewise"]


@st.composite
def expressions(draw: st.DrawFn, max_depth: int = 6, include_piecewise: bool = True) -> ExprTree:
    # 30% early exit at depth>3, 50% at depth<=3.  Keeps deep trees possible
    # without dominating runtime: expression generation is the bottleneck at
    # depth 6 (65% of wall time at 50/50), and 30/70 cuts it roughly in half.
    if max_depth <= 0 or draw(
        st.sampled_from([True] * 3 + [False] * 7 if max_depth > 3 else [True, False])
    ):
        return draw(st.one_of(sym_names, small_ints, small_rats))
    ops = _OPS_WITH_PW if include_piecewise else _OPS_BASE
    op = draw(st.sampled_from(ops))
    a = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
    if op in ("floor", "ceiling"):
        choice = draw(st.sampled_from(["div", "rat_add", "mul", "sub", "add", "plain"]))
        if choice == "div":
            d = draw(pos_ints)
            return (op, ("div", a, d))
        if choice == "rat_add":
            rat_leaf = draw(small_rats)
            return (op, ("add", rat_leaf, a))
        if choice == "mul":
            b = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
            return (op, ("mul", a, b))
        if choice == "sub":
            b = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
            return (op, ("sub", a, b))
        if choice == "add":
            b = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
            return (op, ("add", a, b))
        return (op, a)
    if op == "neg":
        return (op, a)
    if op == "piecewise":
        # Piecewise tuple layout: ("piecewise", val1, cond1, ..., valN, condN, default)
        # ncases = (len - 2) // 2; default is always tree[-1].
        cond_depth = draw(st.integers(min_value=1, max_value=3))
        cond = draw(conditions(max_depth=cond_depth))
        default = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
        if draw(st.booleans()):
            b = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
            cond2 = draw(conditions(max_depth=cond_depth))
            return (op, a, cond, b, cond2, default)
        return (op, a, cond, default)
    if op == "mod" or op == "div":
        b = draw(pos_ints)
    elif op == "xor":
        if draw(st.integers(min_value=0, max_value=3)) == 0:
            a = draw(st.one_of(sym_names, small_ints))
            b = draw(st.one_of(sym_names, small_ints))
        else:
            a = draw(st.integers(min_value=0, max_value=255))
            b = draw(st.integers(min_value=0, max_value=255))
    else:
        b = draw(expressions(max_depth=max_depth - 1, include_piecewise=include_piecewise))
    return (op, a, b)


@st.composite
def conditions(draw: st.DrawFn, max_depth: int = 2) -> CondTree:
    if max_depth <= 0 or draw(st.booleans()):
        a = draw(expressions(max_depth=4))
        b = draw(expressions(max_depth=4))
        op = draw(st.sampled_from([">=", ">", "<=", "<", "==", "!="]))
        return ("cmp", op, a, b)
    combiner = draw(st.sampled_from(["and", "or", "not"]))
    c1 = draw(conditions(max_depth=max_depth - 1))
    if combiner == "not":
        return ("not", c1)
    c2 = draw(conditions(max_depth=max_depth - 1))
    return (combiner, c1, c2)


# ---------------------------------------------------------------------------
#  Tree -> SymPy
# ---------------------------------------------------------------------------

_sp_syms = {n: sympy.Symbol(n, integer=True) for n in _VARS}


def to_sympy(tree: ExprTree) -> Any:
    if isinstance(tree, str):
        return _sp_syms[tree]
    if isinstance(tree, int):
        return sympy.Integer(tree)
    op = tree[0]
    if op == "rat":
        return sympy.Rational(tree[1], tree[2])
    if op == "add":
        return to_sympy(tree[1]) + to_sympy(tree[2])
    if op == "sub":
        return to_sympy(tree[1]) - to_sympy(tree[2])
    if op == "neg":
        return -to_sympy(tree[1])
    if op == "mul":
        return to_sympy(tree[1]) * to_sympy(tree[2])
    if op == "div":
        a, b = to_sympy(tree[1]), to_sympy(tree[2])
        return sympy.Rational(1, int(b)) * a if isinstance(b, sympy.Integer) else a / b
    if op == "floor":
        # evaluate=False: SymPy incorrectly reports is_integer=True for
        # some rational expressions (e.g. y*(2*x+2*y)/30) and drops floor.
        return sympy.floor(to_sympy(tree[1]), evaluate=False)
    if op == "ceiling":
        return sympy.ceiling(to_sympy(tree[1]), evaluate=False)
    if op == "mod":
        # evaluate=False avoids SymPy Mod bugs (e.g. #28744) that silently
        # produce wrong results for certain inputs.
        return sympy.Mod(to_sympy(tree[1]), to_sympy(tree[2]), evaluate=False)
    if op == "max":
        return sympy.Max(to_sympy(tree[1]), to_sympy(tree[2]))
    if op == "min":
        return sympy.Min(to_sympy(tree[1]), to_sympy(tree[2]))
    if op == "xor":
        raise ValueError("xor not supported in SymPy conversion")
    if op == "piecewise":
        ncases = (len(tree) - 2) // 2
        cases = [(to_sympy(tree[1 + 2 * i]), to_sympy_cond(tree[2 + 2 * i])) for i in range(ncases)]
        cases.append((to_sympy(tree[-1]), True))
        return sympy.Piecewise(*cases)
    raise ValueError(f"unknown op: {op}")


def to_sympy_cond(tree: CondTree) -> Any:
    op = tree[0]
    if op == "cmp":
        _, cmp_op, a, b = tree
        sa, sb = to_sympy(a), to_sympy(b)
        ops = {
            ">=": sympy.Ge,
            ">": sympy.Gt,
            "<=": sympy.Le,
            "<": sympy.Lt,
            "==": sympy.Eq,
            "!=": sympy.Ne,
        }
        return ops[cmp_op](sa, sb)
    if op == "not":
        return ~to_sympy_cond(tree[1])
    if op == "and":
        return to_sympy_cond(tree[1]) & to_sympy_cond(tree[2])
    if op == "or":
        return to_sympy_cond(tree[1]) | to_sympy_cond(tree[2])
    raise ValueError(f"unknown cond op: {op}")


# ---------------------------------------------------------------------------
#  Tree -> ixsimpl
# ---------------------------------------------------------------------------


def to_ixsimpl(ctx: ixsimpl.Context, tree: ExprTree) -> ixsimpl.Expr:
    if isinstance(tree, str):
        return ctx.sym(tree)
    if isinstance(tree, int):
        return ctx.int_(tree)
    op = tree[0]
    if op == "rat":
        return ctx.rat(tree[1], tree[2])
    if op == "add":
        return to_ixsimpl(ctx, tree[1]) + to_ixsimpl(ctx, tree[2])
    if op == "sub":
        return to_ixsimpl(ctx, tree[1]) - to_ixsimpl(ctx, tree[2])
    if op == "neg":
        return -to_ixsimpl(ctx, tree[1])
    if op == "mul":
        return to_ixsimpl(ctx, tree[1]) * to_ixsimpl(ctx, tree[2])
    if op == "div":
        return to_ixsimpl(ctx, tree[1]) / to_ixsimpl(ctx, tree[2])
    if op == "floor":
        return ixsimpl.floor(to_ixsimpl(ctx, tree[1]))
    if op == "ceiling":
        return ixsimpl.ceil(to_ixsimpl(ctx, tree[1]))
    if op == "mod":
        return ixsimpl.mod(to_ixsimpl(ctx, tree[1]), to_ixsimpl(ctx, tree[2]))
    if op == "max":
        return ixsimpl.max_(to_ixsimpl(ctx, tree[1]), to_ixsimpl(ctx, tree[2]))
    if op == "min":
        return ixsimpl.min_(to_ixsimpl(ctx, tree[1]), to_ixsimpl(ctx, tree[2]))
    if op == "xor":
        return ixsimpl.xor_(to_ixsimpl(ctx, tree[1]), to_ixsimpl(ctx, tree[2]))
    if op == "piecewise":
        ncases = (len(tree) - 2) // 2
        cases = [
            (to_ixsimpl(ctx, tree[1 + 2 * i]), to_ixsimpl_cond(ctx, tree[2 + 2 * i]))
            for i in range(ncases)
        ]
        cases.append((to_ixsimpl(ctx, tree[-1]), ctx.true_()))
        return ixsimpl.pw(*cases)
    raise ValueError(f"unknown op: {op}")


def to_ixsimpl_cond(ctx: ixsimpl.Context, tree: CondTree) -> ixsimpl.Expr:
    """Convert condition tree to ixsimpl Expr."""
    op = tree[0]
    if op == "cmp":
        _, cmp_op, a, b = tree
        ia, ib = to_ixsimpl(ctx, a), to_ixsimpl(ctx, b)
        if cmp_op == ">=":
            return ia >= ib
        if cmp_op == ">":
            return ia > ib
        if cmp_op == "<=":
            return ia <= ib
        if cmp_op == "<":
            return ia < ib
        if cmp_op == "==":
            return ctx.eq(ia, ib)
        if cmp_op == "!=":
            return ctx.ne(ia, ib)
        raise ValueError(f"unknown cmp_op: {cmp_op}")
    if op == "not":
        return ixsimpl.not_(to_ixsimpl_cond(ctx, tree[1]))
    if op == "and":
        return ixsimpl.and_(to_ixsimpl_cond(ctx, tree[1]), to_ixsimpl_cond(ctx, tree[2]))
    if op == "or":
        return ixsimpl.or_(to_ixsimpl_cond(ctx, tree[1]), to_ixsimpl_cond(ctx, tree[2]))
    raise ValueError(f"unknown cond op: {op}")


# ---------------------------------------------------------------------------
#  Numerical evaluation
# ---------------------------------------------------------------------------


def _floored_mod(a: Any, b: Any) -> Any:
    """Floored modulo: result has the sign of b (Python's native %).

    Uses Python's built-in % which is exact for integers — an earlier
    version using math.floor(a/b) lost precision for large values."""
    if b == 0:
        raise ZeroDivisionError
    return a % b


def eval_expr(tree: ExprTree, env: Env) -> Any:
    """Evaluate expression tree numerically using Python arithmetic."""
    if isinstance(tree, str):
        return env[tree]
    if isinstance(tree, int):
        return tree
    op = tree[0]
    if op == "rat":
        return Fraction(tree[1], tree[2])
    if op == "add":
        return eval_expr(tree[1], env) + eval_expr(tree[2], env)
    if op == "sub":
        return eval_expr(tree[1], env) - eval_expr(tree[2], env)
    if op == "neg":
        return -eval_expr(tree[1], env)
    if op == "mul":
        return eval_expr(tree[1], env) * eval_expr(tree[2], env)
    if op == "div":
        a, b = eval_expr(tree[1], env), eval_expr(tree[2], env)
        if b == 0:
            raise ZeroDivisionError
        return Fraction(a, b)
    if op == "floor":
        v = eval_expr(tree[1], env)
        return math.floor(v)
    if op == "ceiling":
        v = eval_expr(tree[1], env)
        return math.ceil(v)
    if op == "mod":
        return _floored_mod(eval_expr(tree[1], env), eval_expr(tree[2], env))
    if op == "max":
        return max(eval_expr(tree[1], env), eval_expr(tree[2], env))
    if op == "min":
        return min(eval_expr(tree[1], env), eval_expr(tree[2], env))
    if op == "xor":
        return int(eval_expr(tree[1], env)) ^ int(eval_expr(tree[2], env))
    if op == "piecewise":
        ncases = (len(tree) - 2) // 2
        for i in range(ncases):
            if eval_cond(tree[2 + 2 * i], env):
                return eval_expr(tree[1 + 2 * i], env)
        return eval_expr(tree[-1], env)
    raise ValueError(f"unknown op: {op}")


def eval_cond(tree: CondTree, env: Env) -> Any:
    """Evaluate condition tree to a bool."""
    op = tree[0]
    if op == "cmp":
        _, cmp_op, a, b = tree
        va, vb = eval_expr(a, env), eval_expr(b, env)
        if cmp_op == ">=":
            return va >= vb
        if cmp_op == ">":
            return va > vb
        if cmp_op == "<=":
            return va <= vb
        if cmp_op == "<":
            return va < vb
        if cmp_op == "==":
            return va == vb
        if cmp_op == "!=":
            return va != vb
        raise ValueError(f"unknown cmp_op: {cmp_op}")
    if op == "not":
        return not eval_cond(tree[1], env)
    if op == "and":
        return eval_cond(tree[1], env) and eval_cond(tree[2], env)
    if op == "or":
        return eval_cond(tree[1], env) or eval_cond(tree[2], env)
    raise ValueError(f"unknown cond op: {op}")


def _as_int(val: Any) -> int | None:
    """Coerce eval_expr result to int, or None if non-integer."""
    if isinstance(val, int):
        return val
    if isinstance(val, Fraction):
        return int(val) if val.denominator == 1 else None
    return None


def _subs_all(expr: ixsimpl.Expr, ctx: ixsimpl.Context, env: Env) -> ixsimpl.Expr:
    """Substitute all variables; return the raw ixsimpl expression."""
    result = expr
    for name, val in env.items():
        result = result.subs(name, ctx.int_(val))
    return result


def _assert_sentinel_on_py_error(
    ixs_expr: ixsimpl.Expr,
    ctx: ixsimpl.Context,
    env: Env,
    tree: ExprTree,
) -> None:
    """When Python eval raised an error, verify ixsimpl is consistent.

    Must produce either a sentinel (both agree it's undefined) or a
    concrete integer (simplifier legitimately eliminated the undefined
    subexpression, e.g. 0*(1/0) -> 0).  A non-integer symbolic residue
    would indicate a bug.
    """
    try:
        result = _subs_all(ixs_expr, ctx, env)
    except OverflowError:
        return
    if result.is_error:
        return
    try:
        int(result)
    except (TypeError, ValueError):
        raise AssertionError(
            f"Python eval errored but ixsimpl returned non-integer "
            f"non-sentinel: {result} at {env}, expr={tree}"
        ) from None


def eval_ixs(expr: ixsimpl.Expr, ctx: ixsimpl.Context, env: Env) -> int:
    """Evaluate ixsimpl Expr by substituting all variables."""
    result = _subs_all(expr, ctx, env)
    if result.is_error:
        raise ValueError("sentinel")
    try:
        return int(result)
    except TypeError as e:
        raise ValueError(f"result is not an integer constant: {result}") from e


# ---------------------------------------------------------------------------
#  Fuzz tests
# ---------------------------------------------------------------------------


def test_expand_basic() -> None:
    """expand() distributes MUL over ADD."""
    ctx = ixsimpl.Context()
    e = ctx.parse("2*(a + b)")
    expanded = e.expand()
    s = str(expanded)
    assert "2*a" in s
    assert "2*b" in s
    assert "+" in s

    e2 = ctx.parse("(a + b)*(c + d)")
    s2 = str(e2.expand())
    for term in ("a*c", "a*d", "b*c", "b*d"):
        assert term in s2, f"missing {term} in {s2}"


def _check_simplify_consistency(
    expr: ExprTree,
    envs: list[Env],
    *,
    with_trivial_bounds: bool = False,
) -> None:
    """Simplification preserves semantics: evaluate original and simplified
    at random points, check they agree.  When with_trivial_bounds is True,
    all symbols get wide bounds so that bnds is non-NULL and Piecewise
    branch forking / bounds-gated rules are exercised."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    if with_trivial_bounds:
        assumptions = []
        for v in _VARS:
            s = ctx.sym(v)
            assumptions.append(s >= ctx.int_(-1000000))
            assumptions.append(s < ctx.int_(1000001))
        ixs_simplified = ixs_expr.simplify(assumptions=assumptions)
    else:
        ixs_simplified = ixs_expr.simplify()
    assume(not ixs_simplified.is_error)

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        try:
            simp = eval_ixs(ixs_simplified, ctx, env)
        except (ValueError, TypeError):
            continue
        assert orig == simp, f"Mismatch: {orig} != {simp} at {env}, expr={expr}"
        checked += 1
    assume(checked > 0)


@given(expr=expressions(), envs=st.lists(_env_st(0, 100), min_size=1, max_size=10))
def test_simplify_consistency_uniform(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with uniform env [0, 100]."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_wide_env_st(), min_size=1, max_size=10))
def test_simplify_consistency_wide(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with negative/zero/positive env."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_prime_env_st(), min_size=1, max_size=10))
def test_simplify_consistency_primes(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with prime-biased env."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_pow2_env_st(), min_size=1, max_size=10))
def test_simplify_consistency_pow2(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with pow2-biased env."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_spicy_env_st(), min_size=1, max_size=10))
def test_simplify_consistency_spicy(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with mixed interesting values."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_mixed_env_st(), min_size=1, max_size=10))
def test_simplify_consistency_mixed(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification preserves semantics with blended uniform/wide/spicy env."""
    _check_simplify_consistency(expr, envs)


@given(expr=expressions(), envs=st.lists(_env_st(0, 100), min_size=1, max_size=10))
def test_simplify_bounds_aware_uniform(expr: ExprTree, envs: list[Env]) -> None:
    """Bounds-aware simplification preserves semantics (uniform env).
    Trivial bounds activate Piecewise branch forking, Max/Min collapse,
    and other bounds-gated rules that are dead code without assumptions."""
    _check_simplify_consistency(expr, envs, with_trivial_bounds=True)


@given(expr=expressions(), envs=st.lists(_wide_env_st(), min_size=1, max_size=10))
def test_simplify_bounds_aware_wide(expr: ExprTree, envs: list[Env]) -> None:
    """Bounds-aware simplification with negative/zero/positive env."""
    _check_simplify_consistency(expr, envs, with_trivial_bounds=True)


@given(expr=expressions(), envs=st.lists(_spicy_env_st(), min_size=1, max_size=10))
def test_simplify_bounds_aware_spicy(expr: ExprTree, envs: list[Env]) -> None:
    """Bounds-aware simplification with interesting values."""
    _check_simplify_consistency(expr, envs, with_trivial_bounds=True)


@given(
    expr=expressions(include_piecewise=False),
    envs=st.lists(_env_st(), min_size=1, max_size=10),
)
@example(
    expr=("mod", ("mul", 2, ("mod", "x", 3)), 5),
    envs=[{v: 50 for v in _VARS}],
)
def test_matches_sympy(expr: ExprTree, envs: list[Env]) -> None:
    """Cross-check against SymPy: both should produce numerically
    equivalent results.  Python eval_expr is the ground truth; ixsimpl
    must match it exactly.  SymPy is advisory — disagreements are
    reported as warnings (not assertions) because SymPy 1.14's Mod
    with evaluate=False has known bugs (#28744) that produce wrong
    results for nested Mod expressions."""
    ctx = ixsimpl.Context()
    try:
        ixs_result = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_result.is_error)
    ixs_simplified = ixs_result.simplify()
    assume(not ixs_simplified.is_error)

    try:
        sp_expr = to_sympy(expr)
    except (ValueError, TypeError):
        assume(False)

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        ground_truth = _as_int(raw)
        if ground_truth is None:
            continue
        ixs_val = eval_ixs(ixs_simplified, ctx, env)
        assert ground_truth == ixs_val, (
            f"ixsimpl diverges from ground truth at {env}: "
            f"expected={ground_truth}, got={ixs_val}, expr={expr}"
        )
        try:
            sp_env = {sympy.Symbol(k, integer=True): v for k, v in env.items()}
            sp_val = sp_expr.subs(sp_env)
            if sp_val.is_number and sp_val.is_integer:
                sp_int = int(sp_val)
                # Non-fatal: SymPy 1.14 has Mod evaluate=False bugs (#28744)
                # that cause wrong results for nested Mod expressions.
                if sp_int != ground_truth:
                    warnings.warn(
                        f"SymPy disagrees with ground truth at {env}: "
                        f"sympy={sp_int}, expected={ground_truth}, "
                        f"ixsimpl={ixs_result}, sympy={sp_expr}, "
                        f"ixsimpl_simpl={ixs_simplified}",
                        stacklevel=1,
                    )
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            pass
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    expr=expressions(),
    div_sym=st.sampled_from(_VARS),
    divisor=st.integers(min_value=2, max_value=64),
    env_mults=st.lists(
        st.tuples(_env_st(1, 50), st.integers(-25, 25)),
        min_size=1,
        max_size=10,
    ),
)
def test_simplify_with_divisibility(
    expr: ExprTree,
    div_sym: str,
    divisor: int,
    env_mults: list[tuple[Env, int]],
) -> None:
    """Simplification with a divisibility assumption preserves semantics
    when evaluated at points satisfying the assumption."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    sym_node = ctx.sym(div_sym)
    assumption = ctx.eq(ixsimpl.mod(sym_node, ctx.int_(divisor)), ctx.int_(0))
    ixs_simplified = ixs_expr.simplify(assumptions=[assumption])
    assume(not ixs_simplified.is_error)

    checked = 0
    for base_env, mult in env_mults:
        env = {**base_env, div_sym: mult * divisor}
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, (
            f"Divisibility mismatch: {orig} != {simp} at {env}, "
            f"expr={expr}, assumption=Mod({div_sym},{divisor})==0"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    expr=expressions(include_piecewise=False),
    envs=st.lists(_env_st(), min_size=1, max_size=10),
)
def test_to_sympy_semantics(expr: ExprTree, envs: list[Env]) -> None:
    """ixsimpl.sympy_conv.to_sympy produces a SymPy expression that
    evaluates identically to the ixsimpl expression at random points."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    # SymPy Max/Min reject unevaluated Mod nodes as "not comparable".
    try:
        sp_converted = conv_to_sympy(ixs_expr)
    except (ValueError, TypeError):
        assume(False)

    checked = 0
    for env in envs:
        try:
            ixs_val = eval_ixs(ixs_expr, ctx, env)
        except (ValueError, TypeError):
            continue
        try:
            # xreplace avoids SymPy 1.14 .subs() bug with nested Mod.
            sp_env = {sympy.Symbol(k, integer=True): sympy.Integer(v) for k, v in env.items()}
            sp_val = sp_converted.xreplace(sp_env)
            if not sp_val.is_number:
                continue
            assert int(sp_val) == ixs_val, (
                f"to_sympy mismatch at {env}: "
                f"ixsimpl={ixs_val}, sympy={int(sp_val)}, expr={expr}"
            )
            checked += 1
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            continue
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    expr=expressions(include_piecewise=False),
    envs=st.lists(_env_st(), min_size=1, max_size=10),
)
def test_from_sympy_semantics(expr: ExprTree, envs: list[Env]) -> None:
    """ixsimpl.sympy_conv.from_sympy produces an ixsimpl expression that
    evaluates identically to the original tree at random points."""
    try:
        sp_expr = to_sympy(expr)
    except (ValueError, TypeError):
        assume(False)

    ctx = ixsimpl.Context()
    try:
        ixs_converted = conv_from_sympy(ctx, sp_expr)
    except (ValueError, TypeError):
        assume(False)
    assume(not ixs_converted.is_error)

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            _assert_sentinel_on_py_error(ixs_converted, ctx, env, expr)
            continue
        ground_truth = _as_int(raw)
        if ground_truth is None:
            continue
        try:
            ixs_val = eval_ixs(ixs_converted, ctx, env)
        except (ValueError, TypeError):
            continue
        assert ixs_val == ground_truth, (
            f"from_sympy mismatch at {env}: "
            f"expected={ground_truth}, ixsimpl={ixs_val}, expr={expr}"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    expr=expressions(include_piecewise=False),
    envs=st.lists(_env_st(), min_size=1, max_size=10),
)
def test_sympy_roundtrip_semantics(expr: ExprTree, envs: list[Env]) -> None:
    """ixsimpl -> to_sympy -> from_sympy -> ixsimpl preserves numerical
    semantics at random integer points.

    Structural equality is intentionally not checked: SymPy may simplify
    expressions (e.g. Max(0, x**2) -> x**2 for integer x) and the
    roundtripped form can differ structurally while remaining equivalent."""
    ctx = ixsimpl.Context()
    try:
        original = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not original.is_error)

    # SymPy conversion can fail: Max/Min reject unevaluated Mod nodes
    # as "not comparable", and some ixsimpl constructs have no SymPy
    # equivalent.  Skip rather than fail.
    try:
        sp_expr = conv_to_sympy(original)
    except (ValueError, TypeError):
        assume(False)
    try:
        roundtripped = conv_from_sympy(ctx, sp_expr)
    except (ValueError, TypeError):
        assume(False)
    assume(not roundtripped.is_error)

    checked = 0
    for env in envs:
        try:
            orig_val = eval_ixs(original, ctx, env)
        except (ValueError, TypeError, OverflowError):
            continue
        try:
            rt_val = eval_ixs(roundtripped, ctx, env)
        except (ValueError, TypeError, OverflowError):
            continue
        assert orig_val == rt_val, (
            f"roundtrip mismatch at {env}: "
            f"original={orig_val}, roundtripped={rt_val}, expr={expr}"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    expr=expressions(),
    bound_sym=st.sampled_from(_VARS),
    lo=st.integers(min_value=0, max_value=50),
    hi=st.integers(min_value=51, max_value=200),
    envs=st.lists(
        st.fixed_dictionaries({v: st.integers(0, 200) for v in _VARS}),
        min_size=1,
        max_size=10,
    ),
)
def test_simplify_with_bounds(
    expr: ExprTree,
    bound_sym: str,
    lo: int,
    hi: int,
    envs: list[Env],
) -> None:
    """Simplification with bound assumptions (lo <= sym < hi) preserves
    semantics when evaluated at points satisfying the bounds."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    sym_node = ctx.sym(bound_sym)
    assumptions = [sym_node >= ctx.int_(lo), sym_node < ctx.int_(hi)]
    ixs_simplified = ixs_expr.simplify(assumptions=assumptions)
    assume(not ixs_simplified.is_error)

    checked = 0
    for base_env in envs:
        env = {**base_env, bound_sym: max(lo, min(hi - 1, base_env[bound_sym]))}
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, (
            f"Bounds mismatch: {orig} != {simp} at {env}, "
            f"expr={expr}, bounds={lo} <= {bound_sym} < {hi}"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


_LARGE_VALS_MUL = [-(1 << 30), -(1 << 30) + 1, -1, 0, 1, (1 << 30) - 1, (1 << 30)]
_LARGE_VALS_I64 = [
    -(1 << 62),
    -(1 << 62) + 1,
    -(1 << 31),
    -(1 << 31) + 1,
    -1,
    0,
    1,
    (1 << 31) - 1,
    (1 << 31),
    (1 << 62) - 1,
    (1 << 62),
]


@given(
    expr=expressions(max_depth=3, include_piecewise=False),
    envs=st.lists(
        st.fixed_dictionaries(
            {
                v: st.one_of(
                    st.sampled_from(_LARGE_VALS_MUL),
                    st.sampled_from(_LARGE_VALS_I64),
                    st.integers(-10, 10),
                )
                for v in _VARS
            }
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_simplify_near_overflow(expr: ExprTree, envs: list[Env]) -> None:
    """Simplification with values near int64 overflow boundaries.

    Two tiers: +/-2^30 (safe for x*x), +/-2^62 (near int64 boundary).
    When the result fits in int64, it must match Python arbitrary-precision
    arithmetic.  int64 overflow in ixsimpl is expected and skipped.
    """
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    ixs_simplified = ixs_expr.simplify()
    assume(not ixs_simplified.is_error)

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        if not (-(1 << 62) <= orig <= (1 << 62)):
            continue
        try:
            simp = eval_ixs(ixs_simplified, ctx, env)
        except (ValueError, TypeError):
            # ixsimpl uses int64 internally; simplification may reorganize
            # subexpressions so intermediates overflow even when the final
            # Python result fits.  Skip rather than fail.
            continue
        assert orig == simp, f"Near-overflow mismatch: {orig} != {simp} at {env}, expr={expr}"
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    div_sym=st.sampled_from(_VARS),
    divisor=st.integers(min_value=2, max_value=64),
    other=expressions(max_depth=3),
    pattern=st.sampled_from(["floor_div", "ceil_div", "mod", "compound"]),
    env_mults=st.lists(
        st.tuples(_env_st(1, 50), st.integers(-25, 25)),
        min_size=1,
        max_size=10,
    ),
)
def test_divisibility_targeted(
    div_sym: str,
    divisor: int,
    other: ExprTree,
    pattern: str,
    env_mults: list[tuple[Env, int]],
) -> None:
    """Targeted: divisibility assumption with expressions that exercise it."""
    if pattern == "floor_div":
        expr: ExprTree = ("floor", ("div", div_sym, divisor))
    elif pattern == "ceil_div":
        expr = ("ceiling", ("div", div_sym, divisor))
    elif pattern == "mod":
        expr = ("mod", div_sym, divisor)
    else:
        expr = ("floor", ("div", ("add", div_sym, other), divisor))

    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    sym_node = ctx.sym(div_sym)
    assumption = ctx.eq(ixsimpl.mod(sym_node, ctx.int_(divisor)), ctx.int_(0))
    ixs_simplified = ixs_expr.simplify(assumptions=[assumption])
    assume(not ixs_simplified.is_error)

    checked = 0
    for base_env, mult in env_mults:
        env = {**base_env, div_sym: mult * divisor}
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, (
            f"Targeted divisibility mismatch: {orig} != {simp} at {env}, "
            f"expr={expr}, assumption=Mod({div_sym},{divisor})==0"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


@given(
    bound_sym=st.sampled_from(_VARS),
    lo=st.integers(min_value=0, max_value=50),
    hi=st.integers(min_value=51, max_value=200),
    pattern=st.sampled_from(["max_lo", "min_hi", "floor_div", "mod"]),
    envs=st.lists(
        st.fixed_dictionaries({v: st.integers(0, 200) for v in _VARS}),
        min_size=1,
        max_size=10,
    ),
)
def test_bounds_targeted(
    bound_sym: str,
    lo: int,
    hi: int,
    pattern: str,
    envs: list[Env],
) -> None:
    """Targeted: bound assumptions with expressions that exercise them."""
    if pattern == "max_lo":
        expr: ExprTree = ("max", bound_sym, lo)
    elif pattern == "min_hi":
        expr = ("min", bound_sym, hi - 1)
    elif pattern == "floor_div":
        d = max(1, hi - lo)
        expr = ("floor", ("div", bound_sym, d))
    else:
        m = max(1, hi - lo)
        expr = ("mod", bound_sym, m)

    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    sym_node = ctx.sym(bound_sym)
    assumptions = [sym_node >= ctx.int_(lo), sym_node < ctx.int_(hi)]
    ixs_simplified = ixs_expr.simplify(assumptions=assumptions)
    assume(not ixs_simplified.is_error)

    checked = 0
    for base_env in envs:
        env = {**base_env, bound_sym: max(lo, min(hi - 1, base_env[bound_sym]))}
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, (
            f"Targeted bounds mismatch: {orig} != {simp} at {env}, "
            f"expr={expr}, bounds={lo} <= {bound_sym} < {hi}"
        )
        checked += 1
    assume(checked > 0)  # reject vacuous passes (all envs skipped)


# --- Priority 1: cheap, high-ROI fuzz tests ---


@given(
    expr=expressions(),
    envs=st.lists(_mixed_env_st(), min_size=1, max_size=10),
)
def test_expand_semantics(expr: ExprTree, envs: list[Env]) -> None:
    """expand() preserves numerical semantics."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    expanded = ixs_expr.expand()
    assume(not expanded.is_error)

    checked = 0
    for env in envs:
        try:
            orig = eval_ixs(ixs_expr, ctx, env)
        except (ValueError, TypeError):
            continue
        exp_val = eval_ixs(expanded, ctx, env)
        assert orig == exp_val, f"expand mismatch: {orig} != {exp_val} at {env}, expr={expr}"
        checked += 1
    assume(checked > 0)


@given(
    exprs=st.lists(expressions(max_depth=4), min_size=1, max_size=5),
    envs=st.lists(_env_st(0, 100), min_size=1, max_size=5),
)
def test_simplify_batch_matches_individual(exprs: list[ExprTree], envs: list[Env]) -> None:
    """simplify_batch produces the same results as individual simplify calls."""
    ctx = ixsimpl.Context()
    ixs_exprs = []
    for tree in exprs:
        try:
            ixs_exprs.append(to_ixsimpl(ctx, tree))
        except ValueError:
            assume(False)
    assume(all(not e.is_error for e in ixs_exprs))

    individual = [e.simplify() for e in ixs_exprs]
    batch_copy = list(ixs_exprs)
    ctx.simplify_batch(batch_copy)

    checked = 0
    for env in envs:
        for j in range(len(ixs_exprs)):
            if individual[j].is_error or batch_copy[j].is_error:
                continue
            try:
                ind_val = eval_ixs(individual[j], ctx, env)
            except (ValueError, TypeError):
                continue
            try:
                bat_val = eval_ixs(batch_copy[j], ctx, env)
            except (ValueError, TypeError):
                continue
            assert ind_val == bat_val, (
                f"batch vs individual mismatch: {ind_val} != {bat_val} "
                f"at {env}, expr[{j}]={exprs[j]}"
            )
            checked += 1
    assume(checked > 0)


@given(
    expr=expressions(max_depth=4),
    sub_sym=st.sampled_from(_VARS),
    sub_val=st.integers(min_value=-50, max_value=50),
    envs=st.lists(_env_st(1, 50), min_size=1, max_size=10),
)
def test_subs_correctness(
    expr: ExprTree,
    sub_sym: str,
    sub_val: int,
    envs: list[Env],
) -> None:
    """subs(sym, val) then eval == eval with sym=val in the environment."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    substituted = ixs_expr.subs(sub_sym, ctx.int_(sub_val))
    assume(not substituted.is_error)

    checked = 0
    for base_env in envs:
        full_env = {**base_env, sub_sym: sub_val}
        try:
            expected = eval_expr(expr, full_env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(substituted, ctx, full_env, expr)
            continue
        exp_int = _as_int(expected)
        if exp_int is None:
            continue
        try:
            got = eval_ixs(substituted, ctx, full_env)
        except (ValueError, TypeError):
            continue
        assert exp_int == got, (
            f"subs mismatch: {exp_int} != {got} at {full_env}, "
            f"expr={expr}, subs({sub_sym}={sub_val})"
        )
        checked += 1
    assume(checked > 0)


@given(
    expr=expressions(),
    div_sym=st.sampled_from(_VARS),
    divisor=st.integers(min_value=2, max_value=64),
    bound_sym=st.sampled_from(_VARS),
    lo=st.integers(min_value=0, max_value=50),
    hi=st.integers(min_value=51, max_value=200),
    env_mults=st.lists(
        st.tuples(_env_st(1, 50), st.integers(0, 25)),
        min_size=1,
        max_size=10,
    ),
)
def test_combined_divisibility_and_bounds(
    expr: ExprTree,
    div_sym: str,
    divisor: int,
    bound_sym: str,
    lo: int,
    hi: int,
    env_mults: list[tuple[Env, int]],
) -> None:
    """Simplification with both divisibility and bound assumptions."""
    assume(div_sym != bound_sym)
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)

    sym_d = ctx.sym(div_sym)
    sym_b = ctx.sym(bound_sym)
    assumptions = [
        ctx.eq(ixsimpl.mod(sym_d, ctx.int_(divisor)), ctx.int_(0)),
        sym_b >= ctx.int_(lo),
        sym_b < ctx.int_(hi),
    ]
    ixs_simplified = ixs_expr.simplify(assumptions=assumptions)
    assume(not ixs_simplified.is_error)

    checked = 0
    for base_env, mult in env_mults:
        env = {**base_env, div_sym: mult * divisor}
        env[bound_sym] = max(lo, min(hi - 1, env[bound_sym]))
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(ixs_simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, (
            f"Combined mismatch: {orig} != {simp} at {env}, expr={expr}, "
            f"Mod({div_sym},{divisor})==0, {lo}<={bound_sym}<{hi}"
        )
        checked += 1
    assume(checked > 0)


# --- Priority 2: targeted rule-exercising tests ---


@given(
    sym=st.sampled_from(_VARS),
    N=st.integers(min_value=2, max_value=32),
    envs=st.lists(_env_st(1, 100), min_size=1, max_size=10),
)
def test_recognize_mod_targeted(
    sym: str,
    N: int,
    envs: list[Env],
) -> None:
    """x + (-N)*floor(x/N) should simplify to Mod(x, N)."""
    expr: ExprTree = ("add", sym, ("mul", ("neg", N), ("floor", ("div", sym, N))))

    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    simplified = ixs_expr.simplify()
    assume(not simplified.is_error)

    s = str(simplified)
    assert "floor" not in s, f"recognize_mod should have fired: {s}"

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(simplified, ctx, env)
        assert orig == simp, f"recognize_mod mismatch: {orig} != {simp} at {env}"
        checked += 1
    assume(checked > 0)


@given(
    sym=st.sampled_from(_VARS),
    m=st.integers(min_value=2, max_value=32),
    envs=st.lists(_env_st(1, 100), min_size=1, max_size=10),
)
def test_cancel_floor_mod_pairs_targeted(
    sym: str,
    m: int,
    envs: list[Env],
) -> None:
    """m*floor(x/m) + Mod(x, m) should simplify to x."""
    expr: ExprTree = ("add", ("mul", m, ("floor", ("div", sym, m))), ("mod", sym, m))

    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    simplified = ixs_expr.simplify()
    assume(not simplified.is_error)

    assert str(simplified) == sym, f"Expected {sym}, got {simplified}"

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(simplified, ctx, env)
        assert orig == simp
        checked += 1
    assume(checked > 0)


@given(
    other_sym=st.sampled_from(_VARS),
    c=st.integers(min_value=1, max_value=15),
    K_val=st.integers(min_value=16, max_value=50),
    envs=st.lists(_env_st(0, 100), min_size=1, max_size=10),
)
def test_floor_drop_const_sym_targeted(
    other_sym: str,
    c: int,
    K_val: int,
    envs: list[Env],
) -> None:
    """floor((other + c) / K) with 0 <= c < K: exercises floor_drop_const."""
    assume(c < K_val)
    expr: ExprTree = ("floor", ("div", ("add", other_sym, c), K_val))

    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    simplified = ixs_expr.simplify()
    assume(not simplified.is_error)

    checked = 0
    for env in envs:
        try:
            raw = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            _assert_sentinel_on_py_error(simplified, ctx, env, expr)
            continue
        orig = _as_int(raw)
        if orig is None:
            continue
        simp = eval_ixs(simplified, ctx, env)
        assert orig == simp, f"floor_drop_const_sym mismatch: {orig} != {simp}"
        checked += 1
    assume(checked > 0)


def test_check_entailment_basic() -> None:
    """Mirrors the Wave evaluate_with_assumptions use case."""
    ctx = ixsimpl.Context()
    M = ctx.sym("M")

    assume_lt64 = M < 64

    assert ctx.check(M > 70, assumptions=[assume_lt64]) is False
    assert ctx.check(M < 70, assumptions=[assume_lt64]) is True
    assert ctx.check(M < 32, assumptions=[assume_lt64]) is None

    assert ctx.check(ctx.eq(M, 5), assumptions=[ctx.eq(M, 5)]) is True
    assert ctx.check(ctx.eq(M, 3), assumptions=[ctx.eq(M, 5)]) is False


def test_check_two_sided_bounds() -> None:
    ctx = ixsimpl.Context()
    N = ctx.sym("N")
    assumes = [N >= 0, N <= 10]

    assert ctx.check(ctx.ne(N, 20), assumptions=assumes) is True
    assert ctx.check(N >= 0, assumptions=assumes) is True
    assert ctx.check(N < 0, assumptions=assumes) is False
    assert ctx.check(N > 5, assumptions=assumes) is None


def test_check_no_assumptions() -> None:
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    assert ctx.check(x > 0) is None
    assert ctx.check(x < 0) is None


def test_check_non_cmp_returns_none() -> None:
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    assert ctx.check(x) is None
    assert ctx.check(x + 1) is None


def test_has_basic() -> None:
    ctx = ixsimpl.Context()
    x, y, z = ctx.sym("x"), ctx.sym("y"), ctx.sym("z")
    expr = x + 2 * y
    assert expr.has(x)
    assert expr.has(y)
    assert not expr.has(z)
    assert x.has(x)
    assert not ctx.int_(42).has(x)


def test_abs_simplifies_under_bounds() -> None:
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    a = ixsimpl.abs_(x)
    assert a.tag == ixsimpl.PIECEWISE

    pos = a.simplify(assumptions=[x >= 0])
    assert str(pos) == "x"

    neg = a.simplify(assumptions=[x < 0])
    assert str(neg) == "-x"


def test_abs_constant() -> None:
    ctx = ixsimpl.Context()
    assert str(ixsimpl.abs_(ctx.int_(5))) == "5"
    assert str(ixsimpl.abs_(ctx.int_(-3))) == "3"


# ---------------------------------------------------------------------------
#  Expr.eval and lambdify
# ---------------------------------------------------------------------------


def test_eval_basic() -> None:
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    expr = x + 2 * y
    assert expr.eval({"x": 3, "y": 4}) == 11
    assert expr.eval({"x": 0, "y": 0}) == 0
    assert expr.eval({"x": -1, "y": 5}) == 9


def test_eval_constant() -> None:
    ctx = ixsimpl.Context()
    assert ctx.int_(42).eval({}) == 42


def test_eval_with_expr_keys() -> None:
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    expr = x * y
    assert expr.eval({x: 7, y: 6}) == 42  # type: ignore[dict-item]


def test_eval_raises_on_unbound() -> None:
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    expr = x + y
    import pytest

    with pytest.raises(TypeError):
        expr.eval({"x": 1})


def test_eval_floor_mod() -> None:
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    expr = ixsimpl.floor(x / 3)
    assert expr.eval({"x": 10}) == 3
    assert expr.eval({"x": 9}) == 3
    assert expr.eval({"x": 8}) == 2

    expr2 = ixsimpl.mod(x, 4)
    assert expr2.eval({"x": 10}) == 2
    assert expr2.eval({"x": 8}) == 0


def test_lambdify_single_expr() -> None:
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    f = ixsimpl.lambdify([x, y], x + 2 * y)
    assert f(3, 4) == 11
    assert f(0, 0) == 0
    assert f(-1, 5) == 9


def test_lambdify_scalar_symbol() -> None:
    """Single symbol (not a list) is accepted."""
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    f = ixsimpl.lambdify(x, x * x)
    assert f(5) == 25
    assert f(-3) == 9


def test_lambdify_multi_expr() -> None:
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    f = ixsimpl.lambdify([x, y], [x + y, x - y, x * y])
    assert f(10, 3) == [13, 7, 30]
    assert f(0, 0) == [0, 0, 0]


def test_lambdify_constant() -> None:
    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    f = ixsimpl.lambdify([x], ctx.int_(7))
    assert f(999) == 7


def test_lambdify_string_symbols() -> None:
    """String symbol names work too."""
    ctx = ixsimpl.Context()
    x, y = ctx.sym("x"), ctx.sym("y")
    f = ixsimpl.lambdify(["x", "y"], x * y + 1)
    assert f(6, 7) == 43


@given(
    expr=expressions(max_depth=3),
    envs=st.lists(_env_st(1, 50), min_size=1, max_size=10),
)
def test_eval_matches_subs(expr: ExprTree, envs: list[Env]) -> None:
    """Expr.eval agrees with manual subs for integer results."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    simplified = ixs_expr.simplify()
    assume(not simplified.is_error)

    for env in envs:
        try:
            via_subs = eval_ixs(simplified, ctx, env)
        except (ValueError, TypeError):
            continue
        via_eval = simplified.eval(env)
        assert via_subs == via_eval, f"eval vs subs mismatch: {via_subs} != {via_eval} at {env}"


@given(
    expr=expressions(max_depth=3),
    envs=st.lists(_env_st(1, 50), min_size=1, max_size=10),
)
def test_lambdify_matches_eval(expr: ExprTree, envs: list[Env]) -> None:
    """lambdify callable agrees with Expr.eval."""
    ctx = ixsimpl.Context()
    try:
        ixs_expr = to_ixsimpl(ctx, expr)
    except ValueError:
        assume(False)
    assume(not ixs_expr.is_error)
    simplified = ixs_expr.simplify()
    assume(not simplified.is_error)

    syms = sorted(simplified.free_symbols, key=lambda s: s.sym_name)
    if not syms:
        assume(False)
    f = ixsimpl.lambdify(syms, simplified)

    checked = 0
    for env in envs:
        args = [env[s.sym_name] for s in syms]
        try:
            via_eval = simplified.eval(env)
        except (TypeError, ValueError):
            continue
        via_lam = f(*args)
        assert via_eval == via_lam, f"lambdify vs eval mismatch: {via_eval} != {via_lam} at {env}"
        checked += 1
    assume(checked > 0)
