# Sympy to ixsimpl Migration Plan

Status: **In Progress** (Phase 1 partially complete)

## Current State

**48 files** under `wave_lang/` import sympy directly. ixsimpl currently serves
as the simplification backend via roundtrip conversion in `symbol_utils.py`.
Everything else -- symbol creation, expression construction, type checking,
substitution, codegen, Piecewise logic, numeric probing -- still goes through
sympy despite ixsimpl having native support for most of these operations.

The integration point is `wave_lang/kernel/wave/utils/symbol_utils.py`:
- `simplify(expr)` tries ixsimpl first, falls back to sympy expand/cancel loop.
- `ixs_simplify(expr)` does the roundtrip, returns original on conversion error.
- All 12 call sites go through these two functions. No code calls ixsimpl directly.

## ixsimpl API Surface

Source: `ixsimpl/_ixsimpl.pyi`, `ixsimpl/__init__.py`, `ixsimpl/sympy_conv.py`.

### Expression types (tag-based dispatch)
`INT`, `RAT`, `SYM`, `ADD`, `MUL`, `FLOOR`, `CEIL`, `MOD`, `PIECEWISE`, `MAX`,
`MIN`, `XOR`, `CMP`, `AND`, `OR`, `NOT`, `TRUE`, `FALSE`

### Expr methods
| Method | Purpose |
|--------|---------|
| `.tag` | Node type discriminator (replaces `isinstance(expr, sympy.X)`) |
| `.nchildren` / `.children` / `.child(i)` | Generic structural access (replaces `.args`) |
| `.sym_name` | Symbol name (replaces `str(sym)`) |
| `.rat_num` / `.rat_den` | Rational numerator/denominator (replaces `.p`/`.q`) |
| `.add_coeff` / `.add_nterms` / `.add_term(i)` / `.add_term_coeff(i)` | Add decomposition |
| `.mul_coeff` / `.mul_nfactors` / `.mul_factor_base(i)` / `.mul_factor_exp(i)` | Mul decomposition |
| `.pw_ncases` / `.pw_value(i)` / `.pw_cond(i)` | Piecewise branch access |
| `.cmp_op` | Comparison operator kind |
| `.subs(target, replacement)` / `.subs(mapping)` | Substitution (replaces `.subs()`/`.xreplace()`) |
| `.simplify(assumptions=)` | Simplification with assumption constraints |
| `.expand()` | Expression expansion |
| `.to_c()` | C code generation |
| `.free_symbols` | Free symbol set (cached property on `Expr` subclass) |
| `.has(sym)` | True if `sym` appears in the expression tree (replaces `sympy_expr.has()`) |
| `.eval(env)` | Evaluate with concrete values (replaces `expr.subs(dict)` + `int()`) |
| `.is_error` / `.is_parse_error` / `.is_domain_error` | Error detection |
| Arithmetic operators | `+`, `-`, `*`, `/`, `>=`, `>`, `<=`, `<`, `==`, `!=` |

### Context methods
| Method | Purpose |
|--------|---------|
| `ctx.sym(name)` | Create symbol |
| `ctx.int_(val)` / `ctx.rat(p, q)` | Create constants |
| `ctx.true_()` / `ctx.false_()` | Boolean constants |
| `ctx.eq(a, b)` / `ctx.ne(a, b)` | Equality/inequality nodes |
| `ctx.parse(input)` | Parse expression from string |
| `ctx.check(expr, assumptions=)` | Entailment query: returns `True`/`False`/`None` (replaces some `sympy.solve` uses) |
| `ctx.simplify_batch(exprs, assumptions=)` | Batch simplification |
| `ctx.errors` / `ctx.clear_errors()` | Error reporting |
| `ctx.stats()` / `ctx.stats_reset()` | Performance stats |

### Free functions
`floor`, `ceil`, `mod`, `max_`, `min_`, `abs_`, `xor_`, `and_`, `or_`, `not_`,
`pw`, `same_node`, `lambdify`

### Sympy conversion layer (`ixsimpl.sympy_conv`)
`from_sympy(ctx, expr)`, `to_sympy(expr, symbols=, xor_fn=)`,
`extract_assumptions(ctx, expr)`

Handles: all arithmetic, floor/ceil/mod, min/max, xor, piecewise, all
comparisons, and/or/not, true/false. Custom `xor` Function subclass matched by
name. `Abs` is the only notable missing conversion (emulable via piecewise).

## Sympy Usage Categories

### Already migrated to ixsimpl

| Category | Notes |
|----------|-------|
| **Simplification** | `simplify()` and `ixs_simplify()` delegate to ixsimpl. Fallback to sympy `expand`/`cancel` only on conversion failure. |

### Not yet migrated (gap analysis)

| Category | Sympy API Used | ixsimpl Equivalent | Actual Gap |
|----------|---------------|-------------------|------------|
| **Symbol creation** | `sympy.Symbol(name, integer=True, nonneg=True)` | `ctx.sym(name)` + `extract_assumptions` | **Soft.** ixsimpl symbols carry no assumptions intrinsically; assumptions are passed separately to `simplify()`. Wave would need to track assumptions in a side table or always extract from sympy symbols during conversion. |
| **Expression construction** | `Add`, `Mul`, `Pow`, `Integer`, `Rational`, `Mod(evaluate=False)`, `floor`, `ceiling`, `Min`, `Max`, `Abs`, `Piecewise` | Arithmetic ops, `ctx.int_()`, `ctx.rat()`, `floor`, `ceil`, `mod`, `max_`, `min_`, `abs_`, `pw` | **None.** `Abs` via `abs_()` (piecewise under the hood). `evaluate=False` not needed (ixsimpl does not eagerly evaluate). `Pow` expanded to repeated multiplication (already done in `from_sympy`). |
| **Piecewise / conditionals** | `sympy.Piecewise((val, cond), ...)` | `pw(*branches)`, `.pw_ncases`, `.pw_value(i)`, `.pw_cond(i)` | **None.** Full piecewise support. The `piecewise_aware_subs()` workaround in indexing.py exists only because sympy's `.subs()` triggers expensive boolean simplification on Piecewise -- ixsimpl's `.subs()` does not have this problem. |
| **Structural inspection** | `isinstance(expr, sympy.X)`, `.is_number`, `.is_Atom`, `.args`, `.func` | `.tag` (tag-based dispatch), `.nchildren`, `.children`, `.child(i)`, `.has()`, specialized accessors | **None.** `isinstance` dispatch maps to `expr.tag == ixsimpl.ADD` etc. `.has()` is native. `.is_number` maps to `tag in (INT, RAT)`. Specialized accessors (`.add_nterms`, `.mul_nfactors`) give richer decomposition than sympy's flat `.args`. |
| **Substitution** | `.subs()`, `.xreplace()`, `.replace(pred, fn)` | `.subs(target, repl)`, `.subs(mapping)` | **Soft.** Direct substitution is supported. Missing: `.replace(pred, fn)` (predicate-based bottom-up rewrite). Used in `_custom_simplify_once` for 4 transform passes. Would need a Python-level tree walker. |
| **Free symbol inspection** | `.free_symbols`, `.has(sym)` | `.free_symbols` (cached property), `.has(sym)` | **None.** Both are native on the `Expr` class. |
| **Expression decomposition** | `.as_ordered_terms()`, `.as_numer_denom()` | `.add_nterms`/`.add_term(i)`/`.add_term_coeff(i)`, `.rat_num`/`.rat_den` | **Soft.** Add decomposition is richer in ixsimpl (coefficient + terms). Numer/denom split is only used for Rational nodes (`.rat_num`/`.rat_den`). |
| **Numeric probing** | `sympy.lambdify()` with custom modules | `ixsimpl.lambdify()`, `Expr.eval(env)` | **None.** `lambdify` is a near drop-in replacement (uses `.subs()` + constant folding). `Expr.eval(env)` handles single-point evaluation. No custom `modules` parameter needed -- ixsimpl handles floor/Mod/etc natively. |
| **Affine conversion** | Sympy -> MLIR AffineExpr pipeline | None (but `.tag`-based walk is possible) | **Soft.** The converter does a structural walk over the expression tree. This maps directly to ixsimpl's `.tag` + `.child(i)` API. The walk structure would be nearly identical. |
| **Constraint solving** | `sympy.solve()`, `sympy.Eq()` | `ctx.eq()`, `ctx.check(expr, assumptions=)` | **None.** The sole `sympy.solve()` call is `evaluate_with_assumptions()` in `general_utils.py`, which checks whether an inequality is entailed/contradicted by a set of constraints -- returning `True`/`False`/`None`. This maps directly to `ctx.check(expr, assumptions=)`. |
| **Codegen / printing** | `lambdastr()` for grid dim lambdas | `ixsimpl.lambdify()`, `.to_c()` | **None.** `ixsimpl.lambdify()` replaces the `lambdastr()` + `eval()` pattern directly. `.to_c()` available for C codegen if needed. |
| **Type system** | `sympy.Integer`, `sympy.Rational`, `.is_Integer`, `.is_Rational` | `tag == INT`, `tag == RAT`, `.rat_num`, `.rat_den` | **None.** Tag-based dispatch replaces isinstance checks. |

### Summary of gaps

**Hard gaps** (no ixsimpl equivalent):
1. `.replace(pred, fn)` -- predicate-based bottom-up rewriting (4 transforms in
   `_custom_simplify_once`, but these exist only as fallback when ixsimpl
   conversion fails, so they may become dead code as coverage improves).

**Soft gaps** (ixsimpl has the feature, integration work needed):
2. Symbol assumptions -- tracked separately, not on the symbol node.
3. Affine converter -- structural walk needs porting from sympy isinstance to
   ixsimpl tag dispatch.

**No gap** (ready to use):
5. Piecewise construction and inspection (`pw`, `.pw_*` accessors).
6. Free symbol inspection (`.free_symbols`, `.has()`).
7. Substitution (`.subs()`; predicate-based `.replace` excluded).
8. Structural inspection (`.tag`, `.nchildren`, `.children`, `.child(i)`).
9. Expression decomposition (`.add_*`, `.mul_*`, `.rat_*` accessors).
10. Numeric probing and evaluation (`lambdify()`, `.eval()`).
11. Absolute value (`abs_()`).
12. Entailment queries (`ctx.check()` -- replaces `sympy.solve` in
    `evaluate_with_assumptions`).
13. Codegen (`lambdify()`, `.to_c()`).

## Migration Strategy

ixsimpl covers every sympy feature used in Wave: symbol creation, expression
construction, structural inspection (`.tag` + accessors), substitution, free
symbol queries, piecewise, evaluation (`lambdify`, `.eval()`), entailment
checking (`ctx.check()`), expansion, and C codegen. The sole `sympy.solve()`
call is an entailment check that maps directly to `ctx.check()`. The only
remaining hard gap is `.replace(pred, fn)` (predicate-based rewriting), which
lives in the sympy fallback path and may become dead code.

**Recommended end state: ixsimpl as the sole expression IR. Sympy removed
entirely as a runtime dependency.**

### Incremental phases

## Phase 1: Complete simplification migration (CURRENT)

**Goal:** All simplification goes through ixsimpl. Zero calls to
`sympy.simplify()`, `sympy.expand()`, `sympy.cancel()` outside the fallback path
in `symbol_utils.simplify()`.

**Status:** Mostly done. Remaining direct sympy simplification calls:

| File | Call | Action |
|------|------|--------|
| `symbol_utils.py:498,502` | `sympy.expand()`, `sympy.cancel()` in fallback | Keep -- intentional fallback for unconvertible expressions. |
| `symbol_utils.py:303` | `sympy.cancel(t / divisor)` in `split_sum_by_divisibility` | Route through ixsimpl if possible, else keep. |
| `index_mapping_simplify.py:177` | `sympy.cancel(numer - mod_arg)` | Same -- targeted cancellation. |
| `read_write.py:118` | `sympy.expand(diff)` for non-Piecewise | Replace with `simplify(diff)`. |
| `schedule.py:622` | `expr.simplify()` (sympy native method) | Replace with `simplify(expr)` from symbol_utils. |

**Work items:**
1. Replace `expr.simplify()` call in `schedule.py` with `simplify(expr)`.
2. Replace `sympy.expand(diff)` in `read_write.py:118` with `simplify(diff)`.
3. Audit: ensure no other file calls sympy simplification directly.
4. Track fallback frequency to identify remaining conversion gaps.

## Phase 2: Centralize sympy imports behind Wave wrappers

**Goal:** No production file imports sympy directly except foundation modules.

**Allowed import sites:**
- `wave_lang/support/indexing.py` -- type aliases, symbol creation
- `wave_lang/kernel/wave/utils/symbol_utils.py` -- simplification, bounds, probing
- `wave_lang/kernel/wave/mlir_converter/attr_type_converter.py` -- affine conversion
- `wave_lang/kernel/wave/water_mlir/.../sympy_to_affine_converter.py` -- affine conversion
- `wave_lang/kernel/compiler/wave_codegen/emitter.py` -- codegen pattern matching

**Strategy:**
Re-export needed sympy names from `indexing.py` and `symbol_utils.py`.
Downstream files import from Wave modules, not sympy.

```python
# indexing.py additions
from sympy import (
    Integer, Rational, Mod, Piecewise, Eq,
    floor, ceiling, Min, Max,
)
```

**Work items:**
1. Add re-exports to `indexing.py` for expression constructors.
2. Add re-exports to `symbol_utils.py` for analysis utilities.
3. File-by-file: replace `import sympy` with imports from Wave modules.
4. Add a ruff rule or pre-commit check to flag direct `import sympy`.

**Risk:** Low. Mechanical refactor, no behavior change.

## Phase 3: Dual-IR wrapper layer

**Goal:** Introduce a thin `IndexExpr` wrapper that can hold either a sympy
expression or an ixsimpl `Expr`, exposing a unified API. This allows incremental
migration without big-bang rewrites.

**Key insight:** ixsimpl's `.tag`-based dispatch maps cleanly to sympy's
`isinstance` dispatch. The wrapper translates between them:

```python
# Sketch -- not final API
def expr_tag(expr) -> int:
    """Unified tag for both sympy and ixsimpl expressions."""
    if isinstance(expr, ixsimpl.Expr):
        return expr.tag
    if isinstance(expr, sympy.Add):
        return ixsimpl.ADD
    if isinstance(expr, sympy.Mul):
        return ixsimpl.MUL
    ...

def expr_free_symbols(expr) -> set:
    """Free symbols from either IR."""
    return expr.free_symbols  # both have this

def expr_subs(expr, mapping: dict):
    """Substitution on either IR."""
    if isinstance(expr, ixsimpl.Expr):
        return expr.subs(mapping)
    return piecewise_aware_subs(expr, mapping)
```

**Work items:**
1. Define wrapper functions in a new `expr_api.py` or extend `symbol_utils.py`.
2. Migrate callers one pass at a time (one PR per compiler pass).
3. Each pass can be tested independently.

**Risk:** Medium. Need to ensure sympy/ixsimpl semantic equivalence at each call
site. The conversion layer (`sympy_conv.py`) already validates this for the
simplification path.

## Phase 4: Port affine converter to ixsimpl

**Goal:** `sympy_to_affine_converter.py` works directly on ixsimpl `Expr` nodes
instead of sympy expressions.

This phase has **no ordering dependency** on Phases 3 or 5. The affine converter
is a self-contained structural tree walk -- it can be ported to ixsimpl tags
while the rest of the compiler still uses sympy. During transition, the converter
can accept ixsimpl `Expr` natively and fall back to `from_sympy()` conversion
for callers still passing sympy expressions.

ixsimpl's `.tag` + accessor API maps 1:1 to the current isinstance dispatch:

| Current (sympy) | New (ixsimpl) |
|-----------------|---------------|
| `isinstance(expr, sympy.Integer)` | `expr.tag == INT` |
| `isinstance(expr, sympy.Rational)` | `expr.tag == RAT` |
| `isinstance(expr, sympy.Symbol)` | `expr.tag == SYM` |
| `isinstance(expr, sympy.Add)` | `expr.tag == ADD` |
| `isinstance(expr, sympy.Mul)` | `expr.tag == MUL` |
| `isinstance(expr, sympy.floor)` | `expr.tag == FLOOR` |
| `isinstance(expr, sympy.Mod)` | `expr.tag == MOD` |
| `isinstance(expr, sympy.Piecewise)` | `expr.tag == PIECEWISE` |
| `expr.args[0]` | `expr.child(0)` |
| `int(expr)` | `int(expr)` |
| `expr.p, expr.q` | `expr.rat_num, expr.rat_den` |

**Work items:**
1. Create `ixsimpl_to_affine_converter.py` alongside the existing converter.
2. Port case-by-case, sharing the `AffineFraction` infrastructure.
3. Add a `from_sympy()` shim at the entry point so callers passing sympy
   expressions still work during transition.
4. Test both paths produce identical AffineExpr output.
5. Once validated, swap the default and deprecate the sympy path.

**Risk:** Medium. The converter has subtle Rational/fraction handling that needs
careful porting.

## Phase 5: Port emitter to tag-based dispatch

**Goal:** `emitter.py` pattern-matches on ixsimpl tags instead of sympy types.

The emitter currently does:
```python
match expr:
    case sympy.Add(): ...
    case sympy.Mul(): ...
    case sympy.Mod(): ...
    ...
```

This becomes:
```python
tag = expr.tag
if tag == ADD:
    ...
elif tag == MUL:
    ...
elif tag == MOD:
    ...
```

With ixsimpl's specialized accessors, the emitter can also be cleaner -- e.g.
`.add_nterms` / `.add_term(i)` instead of iterating `.args` and guessing
structure.

**Work items:**
1. Port emitter dispatch to ixsimpl tags.
2. Use `.add_*` / `.mul_*` / `.pw_*` accessors for structured decomposition.
3. Keep sympy conversion as fallback during transition.

**Risk:** Medium. The emitter is well-tested via LIT tests. Changes should be
caught by FileCheck.

## Phase 6: Remove sympy from hot paths

**Goal:** Compiler passes operate on ixsimpl `Expr` natively. Sympy is removed
as a runtime dependency.

**What this means:**
- `IndexExpr` type alias changes from `sympy.Expr` to `ixsimpl.Expr`.
- Symbol creation via `ctx.sym()` instead of `sympy.Symbol()`.
- Expression construction via ixsimpl operators and free functions.
- Substitution via `.subs()`.
- Numeric probing via `ixsimpl.lambdify()` and `Expr.eval()`.
- Entailment queries via `ctx.check()` (replaces `evaluate_with_assumptions`).
- No more `piecewise_aware_subs()` workaround.
- No more `evaluate=False` on Mod/floor (ixsimpl does not eagerly evaluate).
- No more sympy bug workarounds (#28744, floor/ceil evaluation bugs).

**Risk:** High but contained. This is the "flip the switch" phase. Every
preceding phase must be complete and validated.

## What NOT to do

1. **Do not try to eliminate sympy in one shot.** The 6 phases exist for a
   reason. Each phase is independently testable and revertible.

2. **Do not add ixsimpl calls outside the centralized wrappers** (until Phase 6
   flips the default IR). All ixsimpl usage should go through `symbol_utils.py`
   so fallback behavior is consistent.

3. **Do not remove the sympy fallback path prematurely.** Until ixsimpl handles
   100% of expressions the compiler produces, the fallback is a safety net.

4. **Do not port the emitter and affine converter simultaneously.** These are
   independent subsystems; port one at a time with full test coverage between.

## Dependency risks

- **ixsimpl is pinned to a specific git SHA.** API changes require updating the
  pin and potentially the conversion layer.
- **Thread safety:** ixsimpl context is not thread-safe (handled via
  thread-local storage). Works fine for multiprocessing. For async, verify
  context isolation.
- **sympy version sensitivity:** The codebase works around sympy bugs (#28744 Mod
  auto-evaluation, floor/ceil evaluation on Max/Min arguments). Moving to ixsimpl
  as primary IR eliminates these workarounds entirely.
- **ixsimpl `.subs()` semantics:** Need to verify it matches sympy's substitution
  semantics exactly, particularly for Piecewise conditions and nested
  replacements. The `piecewise_aware_subs` workaround exists because sympy's
  `.subs()` triggers boolean simplification -- if ixsimpl's `.subs()` does not
  have this problem, the workaround can be dropped.

## Metrics to track

- Number of files with direct `import sympy` (target: 5 -> 2 -> 0 non-test).
- Fallback rate: how often `simplify()` hits the sympy fallback (indicates
  conversion coverage gaps).
- sympy-to-ixsimpl conversion errors by type (identifies which expression
  patterns still need `from_sympy` support).

## Priority order

| Phase | Effort | Risk | Depends on | Impact |
|-------|--------|------|------------|--------|
| 1. Complete simplification migration | Low | Low | -- | Eliminates stray `sympy.simplify` calls |
| 2. Centralize imports | Low | Low | -- | Creates migration chokepoint |
| 3. Dual-IR wrapper layer | Medium | Medium | 2 | Enables incremental pass migration |
| 4. Port affine converter | Medium | Medium | -- | Removes sympy from MLIR lowering |
| 5. Port emitter | Medium | Medium | 3 | Removes sympy from codegen |
| 6. Remove sympy from hot paths | High | High | 3, 4, 5 | ixsimpl becomes primary IR |

Phases 1, 2, and 4 can run in parallel. Phase 4 uses the existing sympy
roundtrip (`from_sympy`) as a shim, so it does not need to wait for the rest
of the compiler to migrate.
