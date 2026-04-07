# ixsimpl — Index Expression Simplifier

A specialized C library for simplifying integer arithmetic expressions
used in index computation, memory addressing, and loop bound calculation.

## Problem Statement

Compilers for tiled computational kernels generate hundreds of index
expressions to compute memory addresses, loop bounds, and data mappings.
These expressions must be simplified to produce efficient code.

Currently SymPy is used for this task. On a representative workload of 609
expressions from a single kernel compilation:

| Metric | Value |
|---|---|
| Total simplify time | 41.4 s |
| Avg per expression | 68 ms |
| Median | 37 ms |
| P90 | 171 ms |
| P99 | 268 ms |
| Max | 388 ms |
| Expressions >= 100ms | 233 (38%) |

SymPy is a general-purpose Python CAS. It carries enormous overhead for this
narrow domain: polymorphic dispatch through Python objects, general-purpose
pattern matching, and simplification rules for trigonometry, calculus, and
algebra that are never triggered.

**Goal**: A C library that simplifies these expressions 100-1000x faster than
SymPy, targeting < 1ms average and < 5ms worst-case per expression.

## Expression Domain Analysis

### What the expressions look like

A typical expression (929 chars average, 2994 chars max, depth up to 11):

```
128*Piecewise(
  (floor(($WG0 + $WG1*ceiling(_M_div_32/4)
    - 32*ceiling(_M_div_32/4)*floor(ceiling(_N_div_32/8)/32))
    / Max(1, ceiling(_N_div_32/8) - 32*floor(ceiling(_N_div_32/8)/32))),
   (ceiling(_N_div_32/8) - 32*floor(ceiling(_N_div_32/8)/32) > 0)
   & ($WG0 + $WG1*ceiling(_M_div_32/4)
      >= 32*ceiling(_M_div_32/4)*floor(ceiling(_N_div_32/8)/32))),
  (Mod(floor($WG0/32 + $WG1*ceiling(_M_div_32/4)/32),
       ceiling(_M_div_32/4)),
   True))
+ 128*floor($T0/64) + 4*floor((Mod($T0, 64))/16)
```

### Operators and functions (exhaustive list)

| Category | Items | Total occurrences |
|---|---|---|
| Arithmetic | `+`, `-`, `*`, `/` (integer division context) | ~59,000 |
| Rounding | `floor()`, `ceiling()` | 25,601 |
| Modular | `Mod(a, b)` | 6,481 |
| Conditional | `Piecewise((val, cond), ..., (val, True))` | 1,136 |
| Min/Max | `Max(a, b)`, `Min(a, b)` | 1,140 |
| Bitwise | `xor(a, b)` | 116 |
| Boolean | `&` (and), `\|` (or), `~` (not) | ~1,168 |
| Comparison | `>`, `<`, `>=`, `<=`, `==`, `!=` | ~2,236 |

**Not present**: trig, hyperbolic, exp, log, sqrt, abs, derivatives,
integrals, series, polynomials over symbolic variables, matrices, sets,
complex numbers, floating-point constants.

### Variables (complete set, 20 total)

`$T0`, `$T1`, `$T2`, `$WG0`, `$WG1`, `$ARGK`,
`$GPR_NUM`, `$MMA_ACC`, `$MMA_LHS_SCALE`, `$MMA_RHS_SCALE`,
`$MMA_SCALE_FP4`, `$index0`, `$index1`,
`_M_div_32`, `_N_div_32`, `_K_div_256`, `_aligned`,
`M`, `N`, `K`

The `$` and `_` prefixes are external naming conventions with no special
meaning to the library. All are treated uniformly as opaque symbolic names.

### Constants

- All integer constants (no floats). Powers of 2 dominate: 2, 4, 8, 16, 32,
  64, 128, 256, 512, 1024, 2048.
- 127 unique rational constants, all with denominators that are powers of 2
  (mostly /32, /16, /8, /4, /2). One exception: `56/3`.
- Rationals only appear as arguments to `floor()`, e.g.,
  `floor(floor(K/32)/8 + 7/8)`.

### Structural patterns

1. **Template + offset**: Many expressions are identical except for an additive
   constant (`+0`, `+16`, `+32`, `+48`, `+64`, ...). These form families of
   8-16 expressions.

2. **Shared subexpressions**: The Piecewise block for index-to-tile
   mapping appears in ~90% of expressions. Subtrees like
   `ceiling(M/128)*floor(ceiling(N/256)/32)` repeat thousands of times.

3. **Two naming epochs**: Early expressions use `_M_div_32`, `_N_div_32`;
   later ones use raw `M`, `N`, `K` with inline division. Same math, different
   symbolic variable granularity.

4. **Paired differences**: Many expressions come as `f(x+a) - f(x+b)` pairs,
   where the simplification goal is to reduce the difference.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Public C API                      │
│  ixs_ctx_create / ixs_parse / ixs_simplify / ...     │
├──────────────────────────────────────────────────────┤
│          Node Construction + Simplification          │
│  Constructors canonicalize bottom-up via shared      │
│  rule engine; ixs_simplify adds top-down pass with   │
│  assumptions. One rule implementation, not two.      │
├──────────────┬───────────────┬───────────────────────┤
│  Expression  │  Hash-consing │  Rational             │
│  DAG Nodes   │  Table        │  Arithmetic           │
├──────────────┴───────────────┴───────────────────────┤
│                Arena Allocator                       │
└──────────────────────────────────────────────────────┘
```

Node construction and simplification are **one logical layer**: every
constructor (e.g., `ixs_add`, `ixs_floor`) applies canonicalization rules
before hash-consing. This is intentional — it ensures all nodes in the DAG
are always in canonical form. `ixs_simplify()` runs an additional top-down
pass that leverages assumptions for bound-dependent rewrites.

**Implementation dependency split**: Public constructors (in `ctx.c`) call
rule functions (in `simplify.c`), which call internal node allocators (in
`node.c`). `node.c` never calls `simplify.c` — the dependency is one-way:
`ctx.c` → `simplify.c` → `node.c`. This prevents circular dependencies.

**Thread safety**: A single `ixs_ctx` is **not** thread-safe. All operations
on a context must be serialized by the caller. For parallel workloads, use one
`ixs_ctx` per thread. Distinct contexts share no state and can be used
concurrently without synchronization.

**Cross-context contract**: All `ixs_node*` arguments passed to any API
function must belong to the same `ixs_ctx` as the `ctx` parameter. Passing
a node from one context to a different context is **undefined behavior**
(dangling arena pointer, wrong hash table).

**Depth limit**: The parser enforces a recursion depth limit (default 256).
Trees built programmatically via the API have no depth limit. The simplifier,
printer, and `ixs_subs` traverse the DAG recursively. `ixs_subs` uses a
256-slot direct-mapped memo cache (4 KB on the stack) keyed by node pointer
to avoid exponential re-traversal of shared subexpressions; collisions only
cause redundant work, never incorrect results. For expressions built
from the corpus (max depth 11) this is safe. Deliberately constructing
extremely deep trees (depth > ~10,000) via the API may cause stack overflow.
This is considered acceptable for the target domain.

### Layer 0: Memory — Arena Allocator

All expression nodes are allocated from a per-context arena. Nodes are never
individually freed; the entire arena is freed when the context is destroyed.
The `ixs_ctx` struct itself is emplaced into the main arena (built on the
stack, then memcpy'd in), so `ixs_ctx_create` issues zero `calloc` calls.
`ixs_ctx_destroy` snapshots the arena to a local before freeing it.
This eliminates per-node malloc/free overhead and improves cache locality.

```c
typedef struct ixs_arena_chunk {
  char *base;            /* data region (immediately after header) */
  size_t used;
  size_t capacity;
  struct ixs_arena_chunk *next;
} ixs_arena_chunk;

typedef struct {
  ixs_arena_chunk *current;
  size_t min_chunk;      /* default 4096 */
} ixs_arena;
```

Each chunk is a **single `malloc`** — the `ixs_arena_chunk` header is
emplaced at the start of the block, followed by the data region aligned
to 16 bytes. One malloc, one free per chunk. Chunks grow by doubling
(with overflow check — if doubling would exceed `SIZE_MAX`, treat as OOM).
Typical working set for one expression is < 64 KB.

#### Scratch Arena

Smart constructors (`simp_add`, `simp_mul`, `simp_and`, `simp_or`, `simp_pw`)
and the parser flatten variadic children into temporary arrays before building
the final hash-consed node. These temporaries live on the scratch arena —
a second arena (`ixs_arena scratch`) in `ixs_ctx`, used
exclusively for short-lived temporary allocations. A save/restore API lets
callers rewind the scratch arena after the temporary data is consumed:

```c
typedef struct {
  ixs_arena_chunk *chunk;
  size_t used;
} ixs_arena_mark;

ixs_arena_mark ixs_arena_save(ixs_arena *a);
void ixs_arena_restore(ixs_arena *a, ixs_arena_mark m);
```

`ixs_arena_save` snapshots the current allocation position — returns
`{a->current, a->current ? a->current->used : 0}`. For an empty arena
(`a->current == NULL`) the mark is `{NULL, 0}`, and restoring to it frees
all chunks. `ixs_arena_restore` rewinds to that snapshot, logically freeing everything
allocated after the save point. Any chunks that were allocated between save
and restore are freed:

```c
void ixs_arena_restore(ixs_arena *a, ixs_arena_mark m) {
  while (a->current != m.chunk) {
    ixs_arena_chunk *doomed = a->current;
    a->current = doomed->next;
    free(doomed);
  }
  if (a->current)
    a->current->used = m.used;
}
```

In the common case (no new chunks allocated), restore is O(1) — just an
offset reset.

**Precondition**: The mark must have been obtained from the same arena via
`ixs_arena_save`. Passing a mark from a different arena or a destroyed arena
is undefined behavior (the loop walks off the chunk list).

Save/restore pairs nest naturally (LIFO). A function that saves, allocates
scratch space, then calls another function which also saves/allocates/restores
— the inner restore only rewinds to the inner save point, leaving the outer
function's scratch data intact. This is safe to arbitrary depth as long as
every save is paired with a restore in the same function scope, which is
guaranteed by the usage pattern (save at entry, restore before return).

**Scratch usage contract**: Only allocate from the scratch arena inside a
save/restore pair. Always use the wrapper/impl pattern (below) so that
restore is called on every exit path. The scratch arena is *not* for
long-lived data — anything that must survive the current function call
belongs in the main arena.

The scratch arena is separate from the main arena because permanent nodes
(created by `ixs_node_add`, etc.) are allocated from the main arena *during*
the window between scratch allocation and restore. Restoring the main arena
would destroy those nodes. The scratch arena holds only temporaries, so
restoring it is always safe.

**Arena grow**: Growing a scratch array is common enough to warrant a
dedicated operation. `ixs_arena_grow` extends an existing allocation in
place when possible, falling back to alloc + copy. Primarily intended for
the scratch arena, where the slow path wastes the old block and restore
reclaims it. Using grow on the main arena wastes the old block permanently,
which is acceptable when growth is rare and bounded (e.g. the error pointer
array doubles at most log2(n) times, totalling negligible waste).
Shrinking is not supported (`new_size` must be `>= old_size`).

```c
void *ixs_arena_grow(ixs_arena *a, void *ptr,
                      size_t old_size, size_t new_size,
                      size_t align) {
  if (!ptr)
    return ixs_arena_alloc(a, new_size, align);
  if (new_size < old_size)
    return NULL;
  /* Fast path: ptr is at the tip of the current chunk.
     Use offset arithmetic (ptr - base) to avoid pointer overflow UB. */
  if (a->current &&
      (char *)ptr >= a->current->base &&
      (size_t)((char *)ptr - a->current->base) + old_size
          == a->current->used) {
    size_t extra = new_size - old_size;
    if (extra <= a->current->capacity - a->current->used) {
      a->current->used += extra;
      return ptr; /* extended in place, no copy */
    }
  }
  /* Slow path: alloc new, copy, old space wasted. */
  void *p = ixs_arena_alloc(a, new_size, align);
  if (p)
    memcpy(p, ptr, old_size);
  return p;
}
```

The fast path is free — no copy, no waste. The `ptr - base` offset
comparison avoids pointer-addition overflow UB on 32-bit systems. The
`extra <= capacity - used` check avoids wrapping (since `used <= capacity`
is a maintained invariant). The slow path wastes the old block, but this
is harmless since scratch space is reclaimed on restore. Returns NULL on
OOM.

**Cleanup discipline — wrapper/impl split**: C has no RAII, and manually
calling `ixs_arena_restore` at every return site is fragile. The solution is
to split each function that uses scratch space into a thin wrapper (owns the
save/restore) and an inner impl (has unrestricted control flow):

```c
static ixs_node *simp_add_impl(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  size_t cap = 16;
  ixs_addterm *terms = ixs_arena_alloc(&ctx->scratch,
                                        cap * sizeof(*terms),
                                        sizeof(void *));
  if (!terms)
    return NULL; /* OOM */
  /* ... can return early from anywhere: */
  if (ixs_node_is_zero(a))
    return b;
  /* ... flatten children into terms[], growing as needed: */
  if (nterms >= cap) {
    size_t newcap = cap * 2;
    if (newcap <= cap || newcap > (size_t)-1 / sizeof(*terms))
      return NULL; /* overflow */
    terms = ixs_arena_grow(&ctx->scratch, terms,
                            cap * sizeof(*terms),
                            newcap * sizeof(*terms),
                            sizeof(void *));
    if (!terms)
      return NULL; /* OOM */
    cap = newcap;
  }
  /* coeff and nterms are computed during flattening (elided). */
  return ixs_node_add(ctx, coeff, nterms, terms);
}

ixs_node *simp_add(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  ixs_arena_mark m = ixs_arena_save(&ctx->scratch);
  ixs_node *result = simp_add_impl(ctx, a, b);
  ixs_arena_restore(&ctx->scratch, m);
  return result;
}
```

The impl can have any number of early returns — sentinel propagation, NULL
checks, overflow bailouts — without worrying about cleanup. The wrapper is
mechanical and impossible to get wrong. This is essentially manual RAII: the
wrapper is the destructor scope, the impl is the body.

Alternatives and why they're worse here:

- **goto cleanup**: Viable for one or two exit points; painful when every
  other line can bail out (which is the case in our constructors with
  sentinel/NULL propagation checks).
- **Single-exit with result variable**: Forces deep nesting or threading a
  result through the whole function. Obscures the logic.
- **Cleanup macros**: Can't `return` from inside a macro scope without
  skipping the cleanup, so you're back to the same problem.

In practice, the initial allocation (16 elements) covers >99% of cases;
growth is rare, and when it happens the in-place fast path almost always
hits.

**OOM on scratch**: Scratch allocation failure is treated identically to
main-arena OOM — the impl returns NULL, which propagates through the
existing NULL-propagation rules. The wrapper always calls
`ixs_arena_restore`, even when the impl returned NULL, keeping the scratch
arena consistent. No error string is pushed (same as main-arena OOM — we
can't allocate memory for it).

**Error path cleanup**: With scratch arena, the `MAX_TERMS` limit and its
associated error paths (`"too many Piecewise cases"`, the `overflow` goto
labels that conflated term-count overflow with rational overflow) are
removed entirely. The only remaining failure mode for term accumulation is
OOM (NULL return).

**Scratch arena in the parser**: `parse_piecewise` collects `values[]` and
`conds[]` arrays of unknown length. Same wrapper/impl split:
`parse_piecewise` saves, calls `parse_piecewise_impl` (which has all the
early `return parse_error(...)` exits), then restores.

**Recursive calls**: A wrapper (e.g. `simp_add`) may call another wrapper
(`simp_mul`) which saves/restores its own scratch region. This is safe —
save/restore pairs nest in LIFO order, each wrapper restores only its own
mark. The scratch arena never leaks across wrapper boundaries.

**Lifecycle**: Initialized in `ixs_ctx_create`, destroyed in
`ixs_ctx_destroy`. No other management needed.

**Status**: Fully implemented. All smart constructors, the parser, `subs`,
and `rewrite_impl` use the scratch arena with wrapper/impl split.
`MAX_TERMS` and `ixs_limits.h` have been removed. OOM injection for
arena testing is tracked separately (bead 4-96o).

### Layer 1: Expression Representation — Hash-Consed DAG

Expressions are represented as a directed acyclic graph (DAG) with
hash-consing. Structurally identical subexpressions share a single node.
This is critical because the input expressions have massive subexpression
sharing.

```c
typedef enum {
    IXS_INT,         // 64-bit integer literal
    IXS_RAT,         // p/q rational (both int64_t, q > 0, gcd(p,q) = 1)
    IXS_SYM,         // named variable
    IXS_ADD,         // n-ary sum: coeff + c1*t1 + c2*t2 + ...
    IXS_MUL,         // n-ary product: coeff * t1^e1 * t2^e2 * ...
    IXS_FLOOR,       // floor(x)
    IXS_CEIL,        // ceiling(x)
    IXS_MOD,         // Mod(a, b) = a - b*floor(a/b), b > 0
    IXS_PIECEWISE,   // Piecewise((v1,c1), ..., (vn, True))
    IXS_MAX,         // Max(a, b)
    IXS_MIN,         // Min(a, b)
    IXS_XOR,         // bitwise xor(a, b) (integer domain)
    IXS_CMP,         // comparison: a op b (see ixs_cmp_op below)
    IXS_AND,         // boolean and
    IXS_OR,          // boolean or
    IXS_NOT,         // boolean not
    IXS_TRUE,        // boolean constant
    IXS_FALSE,       // boolean constant
    IXS_ERROR,       // sentinel: domain error (div/0, overflow, etc.)
    IXS_PARSE_ERROR, // sentinel: syntax error from ixs_parse
} ixs_tag;

typedef enum {
    IXS_CMP_GT,      // >
    IXS_CMP_GE,      // >=
    IXS_CMP_LT,      // <
    IXS_CMP_LE,      // <=
    IXS_CMP_EQ,      // ==
    IXS_CMP_NE,      // !=
} ixs_cmp_op;
```

Each node is a small struct:

```c
typedef struct ixs_node {
    ixs_tag tag;
    uint32_t hash;        // precomputed, used for hash-consing
    union ixs_node_data {
        int64_t ival;                     // IXS_INT
        struct { int64_t p, q; } rat;     // IXS_RAT
        const char *name;                 // IXS_SYM (interned in arena)
        struct {                          // IXS_ADD
            struct ixs_node *coeff;       //   rational constant term
            uint32_t nterms;
            struct ixs_addterm *terms;    //   sorted array (see below)
        } add;
        struct {                          // IXS_MUL
            struct ixs_node *coeff;       //   rational constant factor
            uint32_t nfactors;
            struct ixs_mulfactor *factors; //  sorted array (see below)
        } mul;
        struct {                          // IXS_FLOOR, IXS_CEIL
            struct ixs_node *arg;
        } unary;
        struct {                          // IXS_MOD, IXS_MAX, IXS_MIN, IXS_XOR, IXS_CMP
            struct ixs_node *lhs;
            struct ixs_node *rhs;
            ixs_cmp_op cmp_op;           // used only for IXS_CMP; value ignored for other binary types
        } binary;
        struct {                          // IXS_PIECEWISE
            uint32_t ncases;
            struct ixs_pwcase *cases;     //  array of {value, condition}
        } pw;
        struct {                          // IXS_AND, IXS_OR (n-ary, flattened)
            uint32_t nargs;
            struct ixs_node **args;       // sorted by canonical order
        } logic;
        struct {                          // IXS_NOT
            struct ixs_node *arg;
        } unary_bool;
    } u;
} ixs_node;
```

Helper structs for compound nodes:

```c
typedef struct ixs_addterm {
    struct ixs_node *term;    // the non-constant subexpression
    struct ixs_node *coeff;   // rational coefficient (IXS_INT or IXS_RAT, nonzero)
} ixs_addterm;

typedef struct ixs_mulfactor {
    struct ixs_node *base;    // the non-constant base
    int32_t exp;              // nonzero integer exponent
} ixs_mulfactor;

typedef struct ixs_pwcase {
    struct ixs_node *value;   // branch value
    struct ixs_node *cond;    // branch condition (boolean expression)
} ixs_pwcase;
```

**Symbol interning**: Symbol names are copied into the arena and deduplicated.
The `name` pointer in `IXS_SYM` always points to arena-owned memory, so it
remains valid for the lifetime of the context regardless of whether the caller
frees the original input string.

**Hash-consing**: A global (per-context) hash table maps
`(tag, hash_of_children)` → `ixs_node*`. The table uses open addressing
with linear probing and rehashes at 70% load.

**Probe-before-allocate**: Each node constructor builds a stack-local
`ixs_node tmp`, populates its fields and hash, and probes the hash table
via `htab_lookup`. On hit (full structural comparison confirms match), the
existing pointer is returned with zero arena allocation. On miss, the
constructor arena-allocates the node, copies `tmp` into it, and inserts it
into the table. This avoids wasting arena memory on duplicate nodes.

**Canonical ordering**: Children of `IXS_ADD` and `IXS_MUL` are sorted by a
total order on nodes (by tag, then by content). This ensures `a + b` and
`b + a` produce the same hash and the same canonical node.

### Layer 2: Rational Arithmetic

All constants are exact rationals `p/q` with `int64_t` numerator and
denominator. For this domain, 64-bit is sufficient (the largest observed
constant is 335/32; intermediates from multiplication of tile sizes stay within
int64 range).

Operations: `add(a, b)`, `sub(a, b)`, `mul(a, b)`, `div(a, b)`,
`neg(a)`, `gcd(a, b)`, `is_zero(a)`, `is_one(a)`, `is_negative(a)`,
`cmp(a, b)`, `floor_rat(a)`, `ceil_rat(a)`, `mod_rat(a, b)`.

**Division and Mod use floored (Python/SymPy) semantics, not C truncated
semantics.** This is critical for correctness when comparing against SymPy:

- `floor_div(a, b)` = floor(a / b), e.g., `floor_div(-7, 2) = -4` (not -3)
- `floor_mod(a, b)` = `a - b * floor_div(a, b)`, always non-negative when
  `b > 0`, e.g., `floor_mod(-7, 2) = 1` (not -1)

All results are reduced to lowest terms.

**Overflow**: All intermediate arithmetic is checked for overflow. On overflow
the enclosing constructor returns the sentinel node and appends an error (see
Error Model). Not UB, not assert. `INT64_MIN` is explicitly handled throughout:

- `neg(INT64_MIN)` → sentinel + error.
- `floor_div(INT64_MIN, -1)` → sentinel + error.
- `ixs_rat(ctx, INT64_MIN, q)` for any `q < 0` → sentinel + error (negating
  `p` overflows). Includes `q = -1, -2, ...`.
- `ixs_rat(ctx, p, INT64_MIN)` → sentinel + error (`-q` overflows).
- `gcd(|p|, |q|)` where `p == INT64_MIN` or `q == INT64_MIN` → the GCD
  implementation must handle this without computing `abs(INT64_MIN)`. Use
  binary GCD or special-case: `gcd(INT64_MIN, q)` treats `INT64_MIN` as
  `2^63` (its unsigned magnitude) for the purpose of reduction.

In practice the constants in this domain are small (< 2^20), so overflow
should never occur; the checks are a safety net. No 128-bit fallback in the
initial implementation.

**Division by zero**: `ixs_rat(ctx, p, 0)` returns sentinel. `Mod(x, 0)` and
any `x / 0` during construction or parsing returns sentinel. `ixs_rat` with
`q < 0` normalizes to `(-p, -q)`.

**Mod with negative divisor**: `Mod(x, b)` requires `b > 0`. If `b` is a
known negative constant, the constructor returns `IXS_ERROR`. If `b` is
symbolic, it is assumed positive (the caller's responsibility via
assumptions). This matches the corpus where all Mod divisors are positive
constants or expressions provably positive under assumptions.

### Layer 3: Parser

Recursive descent parser for the SymPy output format. A configurable
recursion depth limit (default 256) prevents stack overflow on maliciously
deep inputs. The grammar is small:

```
expr     = term (('+' | '-') term)*
term     = unary (('*' | '/') unary)*
unary    = '-' unary | atom
atom     = INT | SYMBOL
         | 'floor' '(' expr ')'
         | 'ceiling' '(' expr ')'
         | 'Mod' '(' expr ',' expr ')'
         | 'Max' '(' expr ',' expr ')'
         | 'Min' '(' expr ',' expr ')'
         | 'xor' '(' expr ',' expr ')'
         | 'Piecewise' '(' pw_cases ')'
         | '(' expr ')'
pw_cases = '(' expr ',' cond ')' (',' '(' expr ',' cond ')')*
cond     = cmp_expr (('&' | '|') cmp_expr)*
cmp_expr = '~' cmp_expr | expr cmp_op expr | 'True' | 'False'
           | '(' cond ')' | expr
cmp_op   = '>' | '<' | '>=' | '<=' | '==' | '!='
```

**Bare expressions in conditions**: The grammar allows a bare `expr` in
condition context (the `| expr` alternative in `cmp_expr`). This handles
corpus patterns like `$MMA_LHS_SCALE | $MMA_RHS_SCALE | $MMA_SCALE_FP4` in
Piecewise conditions, where integer-valued flag variables are used as boolean
tests. A bare expression `e` in condition context is desugared to `e != 0`
(i.e., `ixs_cmp(ctx, e, IXS_CMP_NE, ixs_int(ctx, 0))`). The `|` and `&`
operators in conditions are always boolean (`IXS_OR`, `IXS_AND`), never
bitwise. For the 0/1 flag variables in the corpus, boolean and bitwise OR
produce identical truth values. True multi-bit integer bitwise OR is a
non-goal (not present in the corpus).

Symbols: any identifier matching `[A-Za-z_$][A-Za-z0-9_$]*`. All parsed as
`IXS_SYM`. The `$` and `_` prefixes carry no special semantics.

The parser accepts SymPy's `ceiling`; the C API uses `ixs_ceil` for brevity.
Similarly, the parser accepts `True`/`False`; the API uses `ixs_true`/`ixs_false`.

Integer literals: sequences of digits. Rationals are not parsed directly —
they arise from `3/8` being parsed as `IXS_INT(3) / IXS_INT(8)` and
immediately folded to `IXS_RAT(3, 8)`.

The parser builds the DAG directly via the hash-consing table.

### Layer 4: Simplification Engine

Simplification is **table-driven**. Each node type has an ordered array of
`ixs_rule` entries (function pointer, name, needs-bounds flag). The
`try_rules()` dispatch walks the array, tries each rule, records an
`IXS_STATS` hit when one fires, and returns. Rules that require bounds
are skipped when `bnds == NULL`.

Each `simp_*` constructor (e.g., `simp_floor`, `simp_mod`) applies the
fast pre-checks (sentinel propagation, const fold, identity) directly,
constructs the node, then calls `try_rules(ctx, NULL, node, *_rules)`.
For the top-down rewrite pass, each `simp_*_bnds` variant takes an
optional `ixs_bounds *` and calls `try_rules(ctx, bnds, ...)` so
bounds-dependent rules also fire. `rewrite_impl` is just:
```c
case IXS_FLOOR:
    return simp_floor_bnds(ctx, bnds, rewrite(ctx, arg, ...));
```
There is one rule table per type, used by both construction and rewriting.

**Termination guarantee**: The top-down rewrite pass in `ixs_simplify()` runs
a fixed-point loop with a configurable iteration limit (default 64). Each
iteration applies rules bottom-up over the DAG. Most rules are
size-reducing: they either reduce the number of nodes or replace a complex
node with a simpler one (e.g., `floor(3/2)` → `1`, `Mod(x, m)` where
`0 <= x < m` → `x`). A few normalization rules (e.g., comparison
`a > b` → `(a - b) > 0`) may temporarily increase size but enable subsequent
reductions. The iteration limit is the safety net that guarantees termination
regardless. If the limit is reached without convergence, the current best
result is returned and an error is appended to the error list (not sentinel —
the result is still valid, just possibly not fully simplified).

#### 4.1 Constant Folding

Any operation on constants is immediately evaluated:

- `floor(7/2)` → `3`
- `Mod(17, 5)` → `2`
- `Max(3, 7)` → `7`
- `ceiling(4)` → `4` (floor/ceil of integer = identity)
- `xor(5, 3)` → `6`
- `3 > 2` → `True`

#### 4.2 Canonical Form — Add

`IXS_ADD` stores `coeff + Σ ci * ti` where:

- `coeff` is a rational constant (possibly 0)
- Each `ti` is a non-ADD, non-constant node
- Each `ci` is a nonzero rational coefficient
- Terms sorted by canonical order on `ti`

Construction rules:

- `ADD(... + ADD(c + Σ di*ui) ...)` → flatten: merge `c` into coeff, merge
  all `di*ui` terms
- `ADD(... + k * ADD(c + Σ di*ui) ...)` → distribute `k` and flatten
  (enables `(x+y) - (x+y) → 0` by flattening `MUL(-1, ADD(x, y))`)
- Collect like terms: if `ti == tj`, merge `ci + cj`
- Drop terms with `ci == 0`
- If all terms vanish, return `coeff`
- If one term remains with `coeff == 0` and `ci == 1`, return `ti`

#### 4.3 Canonical Form — Mul

`IXS_MUL` stores `coeff * Π bi^ei` where:

- `coeff` is a nonzero rational constant (default 1)
- Each `bi` is a non-MUL, non-constant node
- Each `ei` is a nonzero integer exponent
- Factors sorted by canonical order on `bi`

Construction rules:

- `MUL(... * MUL(c * Π dj^fj) ...)` → flatten
- Collect like bases: if `bi == bj`, merge `ei + ej` (checked for `int32_t`
  overflow; on overflow → sentinel + error)
- Drop factors with `ei == 0` (they contribute 1)
- If `coeff == 0`, return `0`
- `expr * 1` → `expr`
- Pull constant factors out of ADD: `2 * (a + b)` is kept as-is (don't
  distribute). Distribution is not performed by the simplifier.
  Use `ixs_expand` to distribute on demand.

#### 4.4 Floor / Ceiling Rules

Both `floor` and `ceiling` are kept as first-class nodes. They appear in
roughly equal frequency (12,198 and 13,403 occurrences) and normalizing one
to the other (e.g., `ceiling(x) → -floor(-x)`) would introduce negations
that obscure the structure and make output harder to compare against SymPy.

```
floor(integer)                     → identity
floor(p/q)                         → ⌊p/q⌋  (constant fold)
floor(floor(x))                    → floor(x)
floor(ceiling(x))                  → ceiling(x)
floor(integer_valued)              → identity
floor(n + intval_terms + rest)     → n + intval_terms + floor(rest)
    where each intval_term is provably integer-valued (structurally,
    or via congruence: (p/q)*sym when q divides sym's known modulus)
floor(outer * (a + b + ...))       → distribute, extract integer-valued products
    e.g. floor((6*K*floor(x/3) + y) / (2*K)) → 3*floor(x/3) + floor(y/(2*K))
    MUL bases in outer are decomposed for symbolic cancellation

ceiling(integer)                   → identity
ceiling(p/q)                       → ⌈p/q⌉  (constant fold)
ceiling(ceiling(x))                → ceiling(x)
ceiling(floor(x))                  → floor(x)
ceiling(integer_valued)            → identity
ceiling(n + intval_terms + rest)   → n + intval_terms + ceiling(rest)
    (same congruence-aware integrality check as floor)
ceiling(outer * (a + b + ...))     → distribute, extract integer-valued products
```

The ADD and MUL-over-ADD extraction rules share implementation via
`round_extract_add` and `round_extract_mul_add`.  `round_extract_add`
uses a `round_fn` callback for its recursive tail; `round_extract_mul_add`
calls `simp_floor_bnds` / `simp_ceil_bnds` directly (selected by a
`bool is_floor` flag) so the expanded sum inherits bounds context.

```
floor(c + sum(ci*bi))  → floor(sum(ci*bi))
    when every bi is integer-valued and 0 < c < 1/lcm(qi)
    (the sum lies on a 1/L grid; c < 1/L can't cross a grid point)
floor(Mod(X, M) / K)  → 0   when K >= M > 0
round(round(A) / D)   → round(A / D)   when D is a positive integer
                                        (round = floor or ceiling)
```

The `round_unwrap_inner` helper applies the identity `round(round(a)/D) = round(a/D)`
for positive integer D.  Proof (floor): let a = D*q + r + f where q = floor(a/D),
0 <= r < D integer, 0 <= f < 1.  Then r + f < D, so both sides equal q.
The ceiling proof is symmetric.  D must be positive; the identity fails for
negative D (counterexample: `floor(floor(-5.5)/(-3)) = 2`, `floor(-5.5/(-3)) = 1`).
The helper detects MUL nodes where one factor is FLOOR/CEIL^1 and computes
D by inverting exponents and reciprocating the coefficient.  Registered as
`floor_unwrap_inner` in `floor_rules[]` and `ceil_unwrap_inner` in `ceil_rules[]`.

The constant-drop rule is implemented in `floor_drop_const` and registered
in `floor_rules[]`.  When bounds are available, `floor_drop_const` refines
the effective denominator per-term: if `Mod(K, M) == 0` is known for a
symbol `K`, then `|coeff|*M` replaces `|coeff|` in the LCM calculation,
widening the grid (e.g. `floor(7/8 + K/256)` -> `K/256` under `256|K`
because the effective grid is 1, not 1/256).  For non-structurally-integer
terms, `is_known_divisible` checks whether the denominator divides out.

`round_extract_add` splits rational constants into integer + fractional
parts (e.g. `65/32 → 2 + 1/32`) before testing the drop condition,
so `floor(65/32 + 1/2*floor(x/16))` reduces to `2 + floor(1/2*floor(x/16))`.
When bounds are available, `round_extract_add` uses `is_integer_with_divinfo`
and `is_known_divisible` to extract addends with rational coefficients that
are provably integer via congruence (e.g. `floor(x/3 + K/32)` ->
`K/32 + floor(x/3)` when `Mod(K, M) == 0` for any `M` with `32 | M`,
since `32 | K` makes `K/32` integer).  The helper `addterm_is_integer_valued`
encapsulates this check for both the detection and extraction passes.

Both `is_integer_with_divinfo` and `is_known_divisible` handle multi-factor
MUL nodes: for `p/q * f1^e1 * ... * fn^en`, each positive-exponent factor
is checked for integer-valued-ness, and any single factor whose congruence
absorbs the remaining denominator suffices to prove the whole product
integer-valued (or divisible by a given modulus).  This is a sufficient
OR-of-factors test, not full product factorization: `6 | (2*3)` cannot
be proved when neither factor alone is divisible by 6.

```
floor(C/D + sum(ci * ti / D))  →  floor(C'/D + sum(ci * ti / D))
    when every ti is integer-valued, D is symbolic, all terms share D^{-1},
    and C' = C - (C mod gcd(g, L)) where g = gcd of base numerator
    coefficients and L = lcm of rational coefficient denominators.
    (Proof: N = sum(ni*ti) is always a multiple of g; adding
    r < gcd(g, L) to N cannot push past the next floor boundary.)
```

The symbolic-denominator constant-drop rule is implemented in
`floor_drop_const_sym`, also registered in `floor_rules[]`.

`round_extract_mul_add` also distributes `floor(outer * (const + terms))`
when `outer` is non-integer and the ADD has a nonzero constant, even if no
distributed term is integer-valued.  This converts factored forms into the
distributed form expected by `floor_drop_const_sym`.

`ixs_node_is_integer_valued` recognises `IXS_PIECEWISE` nodes as
integer-valued when all value branches are integer-valued.

More advanced rules (applied when domain info is available):

```
floor(x / n) where x = n*q + r, 0 <= r < n
  → q    (when r's bounds are provable)

floor(floor(x/a) / b)    → floor(x / (a*b))    when a,b > 0 integer
ceiling(ceiling(x/a) / b) → ceiling(x / (a*b))  when a,b > 0 integer
Mod(a*floor(x/a), a)     → 0
Mod(x, n) where 0 <= x < n is provable → x
```

#### 4.5 Mod Rules

`Mod(a, b)` is canonically `a - b * floor(a / b)`. The simplifier can either:

(a) Keep `Mod` as a first-class node when no simplification applies, or
(b) Expand to `a - b * floor(a / b)` and let floor rules do the work.

Strategy: keep `Mod` as a node (it reads better and avoids expression blowup),
but apply these rules:

```
Mod(c, m)           where c,m constant   → c mod m
Mod(x, 1)                                → 0
Mod(x + k*m, m)     where k is integer   → Mod(x, m)
Mod(x, m)           where 0 <= x < m     → x
Mod(Mod(x, m), m)                        → Mod(x, m)
Mod(a*m + b, m)     where a contains no IXS_MOD node → Mod(b, m)
Mod(g*x + r, g*m)   where g > 1, 0 <= r < g,
                     all terms integer   → g*Mod(x, m) + r
Mod(c*x, c*m)       where c > 1         → c*Mod(x, m)

(reverse direction, in simp_add — recognize_mod)
c*E - c*N*floor(E/N)                    → c*Mod(E, N)
c*N*ceil(E/N) - c*E                     → c*Mod(-E, N)
c*E - c*D*floor(E/D)                    → c*Mod(E, D)     (D symbolic)
c*D*ceil(E/D) - c*E                     → c*Mod(-E, D)    (D symbolic)

(forward direction, in simp_add — cancel_floor_mod_pairs)
ci*m*floor(E/m) + ci*Mod(E, m)          → ci*E
```

The forward cancellation uses two verification strategies:

1. **floor(A/m) == floor_node** — reconstructs the expected floor via
   `simp_floor(simp_div(A, m))` and checks hash-consed pointer equality.
   Robust against eager floor rewrites (e.g. `round_pull_in_denom`
   collapsing `floor(floor(x/3)/2)` into `floor(x/6)`).

2. **m * floor_arg == A** — distributes `m` over the floor argument using
   `distribute_mul_decompose`, which decomposes compound inverse MUL bases
   (e.g. `K * (K/2)^{-1} → 2`) to enable symbolic cancellation.

#### 4.5.1 Opposite-Coefficient MUL-ADD Cancellation

In `simp_add`, after coalescing like terms, `reduce_opposite_mul_add`
scans for pairs of addterms `c*M_i` and `-c*M_j` where both bases are
MUL nodes with the same number of factors.  If the two MUL nodes share
all factors (the "outer product") except for exactly one ADD^1 factor
each, the pair collapses:

```
c*K*(PW + A) - c*K*(PW + B)  ->  c*K*(A - B)
```

This eliminates shared Piecewise sub-expressions that arise when
index computations branch identically across a stride decomposition.
Factor ordering in MUL is hash-based, so the rule compares reconstructed
outer products (via `apply_pow`) rather than relying on positional
matching.  The pass runs to a fixpoint (repeated until no pair reduces).

#### 4.6 Piecewise Rules

Piecewise requires `n >= 1`. The last case should have condition `True`
(catch-all default). If after eliminating `False` branches no cases remain,
the result is `IXS_ERROR` (no defined value).

```
Piecewise((v, True))                → v
Piecewise((v, False), rest...)      → Piecewise(rest...)
Piecewise((v, c), (v, d), rest...)  → Piecewise((v, c | d), rest...)
                                      (same value, merge conditions)
Piecewise((a, c), (b, True))       where c evaluates to True → a
                                    where c evaluates to False → b

// Propagation through arithmetic:
k * Piecewise((v1, c1), ..., (vn, cn)) → Piecewise((k*v1, c1), ..., (k*vn, cn))
Piecewise(...) + expr                 → Piecewise((v1+expr, c1), ..., (vn+expr, cn))
floor(Piecewise((v1, c1), ...))       → Piecewise((floor(v1), c1), ...)
Mod(Piecewise(...), m)                → Piecewise((Mod(v1, m), c1), ...)
```

**Propagation strategy**: Push Piecewise inward (into branches) when the
enclosing operation is a simple linear function of the Piecewise result
(multiply by constant, add a term, apply floor/Mod). This lets each branch
simplify independently. Do NOT lift Piecewise outward (e.g., wrap an entire
Add in Piecewise) as that duplicates the non-Piecewise terms and causes
expression blowup.

**Branch-aware bounds**: During bounds-aware rewriting, each non-default
Piecewise branch gets a forked copy of the current bounds augmented with
the branch condition.  Conditions like `E > 0` tighten the lower bound of
`E` to 1, letting `Max(1, E)` collapse inside that branch.  Expression-level
bounds are stored alongside per-symbol bounds and intersected during
interval queries.  All four comparison directions (GT, GE, LT, LE) are
stored directly as expression bounds.  For LT/LE conditions (`E < 0`,
`E <= 0`), the negated expression `-E` is also stored with a flipped GT/GE
bound; this is necessary because `Max(-E, c)` sees `-E` as a distinct
hash-consed node from `E`, and the expression-bound lookup requires pointer
equality.

**Product-zero decomposition**: When a guard pins a product `A*B` to zero
(the LE bound and propagated GE bound intersect to `[0,0]`) and all factors
except one are provably nonzero, the remaining factor gets an explicit
`[0,0]` bound.  For example, `floor(C/32)*ceil(M/256) <= 0` with `M >= 1`
forces `ceil(M/256) >= 1`, so `floor(C/32)` must be zero.  This lets the
branch value `floor(-32*floor(C/32)*...) = floor(0) = 0` collapse,
eliminating the entire Piecewise when the branch matches the default.

**Sentinel handling**: Sentinels do not eagerly propagate through Piecewise
(see Error Model). A sentinel value in a branch whose condition folds to
`False` is silently dropped. A sentinel condition in an unreachable branch
(preceded by a `True` condition) is silently dropped. Only when the sentinel
is in the "live" path does the Piecewise become sentinel.

#### 4.7 Max / Min Rules

```
Max(a, a)       → a
Max(a, b)       where a >= b provable → a
Max(1, x)       where x > 0 provable → Max(1, x) (keep)
                where x >= 1 provable → x

Min(a, a)       → a
Min(a, b)       where a <= b provable → a
Min(a, b)       where a,b constant   → min(a, b)
```

#### 4.8 XOR Rules

```
xor(a, a)       → 0
xor(a, 0)       → a
xor(0, b)       → b
xor(c1, c2)     → c1 ^ c2  (constant fold)
```

XOR appears only 116 times in the corpus and only in specific patterns
(`xor(Mod($T0, 8), floor(Mod($T0, 64)/16) + offset)`). These are bit
manipulation patterns and may not need deep algebraic simplification.

#### 4.9 Boolean Simplification

```
True & x        → x
False & x       → False
True | x        → True
False | x       → x
~True           → False
~False          → True
~(~x)           → x
~(a > b)        → a <= b
(a > b) & (a >= b) → a > b
```

#### 4.10 Comparison Simplification

```
a > b   → (a - b) > 0   (normalize to compare against 0)
```

Then apply constant folding when `a - b` reduces to a constant, or bound
analysis when the sign of `a - b` is provable.

### Layer 5: Bound Analysis (Phase 4)

Many simplification rules require knowing whether a subexpression is
non-negative, positive, or bounded. A lightweight interval analysis pass:

- **Variable storage**: Per-variable bounds live in a growable array on the
  scratch arena (starts at 16 slots, doubles on overflow).  Lookup is O(n)
  linear scan by interned name pointer.  If profiling shows this is hot,
  the array can be swapped for an open-addressing hash map keyed on the
  same pointer — the `ixs_bounds` interface is designed to make that a
  drop-in replacement.
- **Interval bounds**: `$T0 >= 0`, `$T0 < 256`, etc. — the simplifier
  extracts interval bounds from comparison assumptions automatically
- **Modular congruence**: `Mod(K, 32) == R` — the simplifier tracks
  `K ≡ R (mod 32)`.  Multiple assumptions on the same symbol merge via CRT
  (Chinese Remainder Theorem).  Pure divisibility (`R == 0`) is the common
  special case; `Mod(K, 256) == 0` implies `Mod(K, 32) == 0`.
- `floor(x)`: if `lo <= x <= hi`, then `floor(lo) <= floor(x) <= floor(hi)`
- `Mod(x, m)`: result in `[0, m-1]` when `m > 0` and `x` is integer-valued
- `ceiling(x/m)`: result >= 0 when `x >= 0` and `m > 0`

**Conflicting assumptions**: Detecting contradictory assumptions is
**best-effort**. When a contradiction is detected (e.g., interval
intersection yields `lo > hi` for a variable), the simplifier returns
`IXS_ERROR` and appends a diagnostic (e.g., `"contradictory assumptions:
$T0 >= 5 and $T0 < 3"`). However, not all contradictions are detectable
(cross-variable constraints like `x >= y, y >= x + 1` require constraint
solving, which is out of scope). When a contradiction goes undetected, the
expression may be simplified incorrectly or returned unchanged. Passing
consistent assumptions is the caller's responsibility.

This enables rules like:

- **Equality substitution**: when a symbol's bounds collapse to a point
  interval `[c, c]` (from `sym == c` assumption, or derived via
  `sym >= c` ∧ `sym <= c`), the rewriter replaces the symbol with integer
  `c` throughout the expression tree. This cascades through constant
  folding, collapsing `ceiling(M/256)` to `1` when `M == 256`, etc.
- `Mod(x, 32)` where `0 <= x < 32` → `x`
- `floor(x/64)` where `0 <= x < 64` → `0`
- `floor(x)` → constant when `floor(lo) == floor(hi)` (same for ceiling)
- `Mod(x, m)` bounds tightened to dividend's bounds when `0 <= x < m`
- `Max(1, expr)` where `expr >= 1` → `expr`

**Congruence-gated rewrites** (requires `Mod(sym, M) == R` assumption):

- `Mod(sym, m)` → `R % m` when `M % m == 0` (generalizes divisibility to
  nonzero remainders; for `R == 0` this gives `→ 0`)
- `floor(sym / m)` → `sym / m` when `M % m == 0` and `R % m == 0`
  (divisibility: the known congruence absorbs `m`)
- `Mod(c*sym, m)` → `0` when `m` divides `c * divisor(sym)`
- `floor(expr)` / `ceiling(expr)` → `expr` when `expr` is provably
  integer-valued using congruence (e.g., `floor(K/32)` → `K/32`
  when `Mod(K, 32) == 0`)
- `floor(a + (p/q)*sym + rest)` → `(p/q)*sym + floor(a + rest)` when
  `q/gcd(|p|,q)` divides sym's known modulus (the rational addend is
  integer per congruence and can be extracted from the floor)

**Algebraic rewrites** (no bounds needed):

- `Mod(c1*t1 + ... + c, q)` → `Mod(terms, q) + c` when each `|ci|`
  divides `q`, each `ti` is integer-valued, and `0 < c < gcd(|ci|)`.
  (Ported from IREE Wave's `symbol_utils.py`, corrected: use `gcd` not
  `min` for the multi-term case.)

**Bounds-gated algebraic rewrites**:

- `floor(c1*t1 + ... + r)` → `floor(c1*t1 + ...)` when each `ti` is a
  non-negative integer (verified via bounds), `0 < r < 1/lcm(denoms)`,
  so `r` is too small to shift the floor past an integer boundary.

**Interval propagation through symbolic MUL/DIV**:

The bounds engine propagates intervals through `IXS_MUL` nodes with
arbitrary numbers of factors and negative exponents (division by symbolic
expressions).  This enables proving `floor(A/D) = 0` when `A` has a
concrete upper bound and `D` has a symbolic lower bound that guarantees
`A/D < 1`.

- **Interval multiplication** (`iv_mul`): computes the tightest enclosing
  interval for `[a,b] * [c,d]` by evaluating all four corner products
  `{a*c, a*d, b*c, b*d}` and selecting the min/max.  Handles mixed-sign
  intervals correctly.
- **Interval reciprocal** (`iv_recip`): for a strictly positive interval
  `[a,b]` with `a > 0`, returns `[1/b, 1/a]`.  When the upper bound
  is effectively infinite (`INT64_MAX`), the lower bound of the reciprocal
  is `0` (safe over-approximation: `1/+inf → 0`).
- **Overflow widening** (`iv_endpoint_widen`): when `ixs_rat_mul` overflows
  during interval arithmetic, the endpoint is widened to `INT64_MIN` or
  `INT64_MAX` (representing −∞ or +∞) based on the sign of the factors.
  This trades precision for soundness: the interval is always a correct
  over-approximation.

The `IXS_MUL` propagation rule in `bounds_get_propagated`:

```
MUL(coeff, f1^e1, f2^e2, ...) where each ei ∈ {-1, +1}:
  result = interval(coeff)
  for each factor fi:
    if ei == +1:  result = iv_mul(result, bounds(fi))
    if ei == -1:  result = iv_mul(result, iv_recip(bounds(fi)))
```

Example: `floor(x / (128*K))` with `x ∈ [0,127], K ≥ 1`.
Internal representation is `floor(MUL(1/128, x^1, K^-1))`.
Propagation: `[1/128, 1/128] * [0, 127] * iv_recip([1, +inf))`
= `[0, 127/128] * [0, 1]` = `[0, 127/128]`.
Then `floor([0, 127/128]) = [0, 0]`, collapsing to constant `0`.

**Limitation**: independent interval propagation cannot handle relational
constraints between variables.  For `floor(x/K)` with `x < K-1, K >= 2`,
the bounds engine sees `x ∈ [0, INT64_MAX]` and `K ∈ [2, INT64_MAX]`
independently, so `x/K` has an unbounded interval.  Proving `floor(x/K) = 0`
from `x < K` requires tracking cross-variable relationships, which is a
non-goal for the current bounds engine.  The planned alternative is
divisibility-aware floor reasoning (see beads tracker).

## Error Model

The library uses a three-tier error model: NULL, and two distinct sentinel
node types.

### Tier 1: NULL — Out of Memory (catastrophic)

All constructors and `ixs_parse` return `NULL` when the arena cannot grow.
This is unrecoverable. No error string is set (we can't allocate memory for
it). NULL propagates silently: any constructor that receives a NULL argument
returns NULL immediately without side effects.

### Tier 2: Parse Error Sentinel (`IXS_PARSE_ERROR`)

Returned by `ixs_parse` when the input is syntactically malformed (unexpected
token, unmatched parentheses, unknown function name, recursion depth limit
exceeded). There is one parse-error sentinel per context (singleton). Error
details are appended to the context's error list.

Only `ixs_parse` produces this sentinel. Constructors never produce it.

### Tier 3: Domain Error Sentinel (`IXS_ERROR`)

Returned by constructors when an operation is mathematically undefined:
division by zero, `Mod(x, 0)`, rational overflow, `ixs_rat(ctx, p, 0)`,
integer literal overflow during parsing of a syntactically valid number.
There is one domain-error sentinel per context (singleton). Error details are
appended to the context's error list.

`ixs_parse` can return this sentinel when the input is syntactically valid
but contains a domain error (e.g., `"1/0 + x"`).

### Propagation rules

Both sentinels propagate identically through constructors:

| Input | Behavior |
|---|---|
| Any constructor receives a sentinel arg | Returns that sentinel silently (no new error) |
| Any constructor receives NULL arg | Returns NULL silently |
| Two sentinel args (different kinds) | Returns `IXS_PARSE_ERROR` (highest sentinel priority) |
| NULL + any sentinel | Returns NULL (NULL always wins) |
| Operation causes domain error | Returns `IXS_ERROR`, appends error to list |

**Priority**: `NULL` > `IXS_PARSE_ERROR` > `IXS_ERROR`. When multiple error
tiers are present in the arguments, the highest-priority one wins. This
ensures that parse errors are never masked by domain errors, and OOM is
never masked by anything.

Only the operation that **originates** the error appends to the error list.
Propagation through downstream constructors is silent.

**Piecewise exception** — sentinels do NOT eagerly propagate through
`ixs_pw`. A Piecewise branch may contain a sentinel value or sentinel
condition without poisoning the entire expression, similar to LLVM's poison
semantics in `select`:

- If a condition folds to `False`, the branch is dropped — sentinel in its
  value disappears harmlessly.
- If a condition folds to `True`, the branch value (sentinel or not) becomes
  the result.
- If the first non-eliminated condition is a sentinel, the Piecewise cannot
  determine which branch to take and becomes that sentinel.
- `Piecewise((sentinel, x > 0), (42, True))` with `x = -1` simplifies to
  `42`, not sentinel.

This enables batch processing: one expression with a div/0 in a dead
Piecewise branch doesn't invalidate the other 608 expressions.

### Error List API

```c
// Query errors accumulated since last clear.
// ixs_ctx_error returns NULL if index >= ixs_ctx_nerrors(ctx).
size_t      ixs_ctx_nerrors(ixs_ctx *ctx);
const char *ixs_ctx_error(ixs_ctx *ctx, size_t index);
void        ixs_ctx_clear_errors(ixs_ctx *ctx);

// Check sentinel kind
bool        ixs_is_error(ixs_node *node);        // true for either sentinel
bool        ixs_is_parse_error(ixs_node *node);   // true only for IXS_PARSE_ERROR
bool        ixs_is_domain_error(ixs_node *node);  // true only for IXS_ERROR
```

Error strings are arena-allocated (so they live until `ixs_ctx_destroy`).
Each string includes the error kind and location, e.g.:
- `"parse error at offset 7: unexpected token 'bar'"`
- `"parse error: recursion depth limit (256) exceeded"`
- `"division by zero"`
- `"Mod: divisor is zero"`
- `"rational overflow in multiply"`
- `"integer literal overflow at offset 42"`

### Summary Table

| Condition | Return | Error list | Example |
|---|---|---|---|
| OOM | `NULL` | unchanged | arena exhausted |
| Syntax error | `IXS_PARSE_ERROR` | appended | `"foo bar +"` |
| Domain error | `IXS_ERROR` | appended | `1/0`, `Mod(x,0)`, overflow |
| Valid parse with domain error | `IXS_ERROR` | appended | `"1/0 + x"` |
| NULL input | `NULL` | unchanged | propagation |
| Sentinel input | same sentinel | unchanged | propagation |
| Sentinel in Piecewise value | Piecewise kept | unchanged | dead branch |
| `ixs_print(sentinel)` | writes `"<error>"` | unchanged | round-trip safe |

### ixs_parse Return Values

| Input | Return |
|---|---|
| Valid expression | `ixs_node*` (valid) |
| Syntax error | `IXS_PARSE_ERROR` sentinel |
| Syntactically valid but contains domain error | `IXS_ERROR` sentinel |
| OOM | `NULL` |

## Public API

```c
// Context: owns the arena, hash-consing table, and symbol table
typedef struct ixs_ctx ixs_ctx;
typedef struct ixs_node ixs_node;

// Lifecycle. Returns NULL on OOM.
ixs_ctx   *ixs_ctx_create(void);
void       ixs_ctx_destroy(ixs_ctx *ctx);  // NULL-safe

// Error list and sentinel checks (see Error Model section above).
// ctx must be non-NULL for these (only ixs_ctx_destroy is NULL-safe).
size_t      ixs_ctx_nerrors(ixs_ctx *ctx);
const char *ixs_ctx_error(ixs_ctx *ctx, size_t index);
void        ixs_ctx_clear_errors(ixs_ctx *ctx);
bool        ixs_is_error(ixs_node *node);         // true for either sentinel
bool        ixs_is_parse_error(ixs_node *node);    // true only for IXS_PARSE_ERROR
bool        ixs_is_domain_error(ixs_node *node);   // true only for IXS_ERROR

// Parse a SymPy-format expression string.
// Returns: valid node on success, IXS_PARSE_ERROR sentinel on syntax error,
// IXS_ERROR sentinel if syntactically valid but contains domain error
// (e.g. 1/0), NULL on OOM. Error details appended to context error list.
// Precondition: input must be a non-NULL pointer to at least len bytes of
// readable memory. NULL input is undefined behavior.
ixs_node *ixs_parse(ixs_ctx *ctx, const char *input, size_t len);

// Construct expressions programmatically
ixs_node *ixs_int(ixs_ctx *ctx, int64_t val);
ixs_node *ixs_rat(ixs_ctx *ctx, int64_t p, int64_t q);
ixs_node *ixs_sym(ixs_ctx *ctx, const char *name);  // name must be non-NULL and non-empty
ixs_node *ixs_add(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_sub(ixs_ctx *ctx, ixs_node *a, ixs_node *b);  // a + (-1)*b
ixs_node *ixs_neg(ixs_ctx *ctx, ixs_node *a);                // (-1)*a
ixs_node *ixs_mul(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_div(ixs_ctx *ctx, ixs_node *a, ixs_node *b);  // a*b^(-1); ERROR on b==0
ixs_node *ixs_floor(ixs_ctx *ctx, ixs_node *x);
ixs_node *ixs_ceil(ixs_ctx *ctx, ixs_node *x);
ixs_node *ixs_mod(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_max(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_min(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_xor(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_pw(ixs_ctx *ctx, uint32_t n, ixs_node **values, ixs_node **conds);
                    // n >= 1; last cond should be ixs_true
                    // values and conds must be non-NULL arrays of at least n elements
                    // NULL/sentinel entries: NULL element → entire result is NULL;
                    // sentinel element → propagates per priority rules
ixs_node *ixs_cmp(ixs_ctx *ctx, ixs_node *a, ixs_cmp_op op, ixs_node *b);
ixs_node *ixs_and(ixs_ctx *ctx, ixs_node *a, ixs_node *b);  // flattens into n-ary
ixs_node *ixs_or(ixs_ctx *ctx, ixs_node *a, ixs_node *b);   // flattens into n-ary
ixs_node *ixs_not(ixs_ctx *ctx, ixs_node *a);
ixs_node *ixs_true(ixs_ctx *ctx);
ixs_node *ixs_false(ixs_ctx *ctx);

// Simplify with assumptions. Assumptions are boolean expression nodes
// (e.g., comparisons like `M > 0`, `$T0 >= 0`, `$T0 < 256`, or equalities
// like `Mod(M, 128) == 0`). The simplifier extracts variable bounds from
// comparison assumptions automatically. Assumptions are separate from the
// expression tree so that hash-consing is preserved. Pass NULL/0 if none.
// NULL/sentinel propagation: if expr is NULL returns NULL, if expr is
// sentinel returns that sentinel. NULL/sentinel entries in assumptions
// array are silently skipped.
// Precondition: assumptions must be non-NULL when n_assumptions > 0.
// NOTE: if the fixed-point iteration limit is reached without convergence,
// the current best result is returned and an error is appended to the
// error list. Always check ixs_ctx_nerrors() after simplification if you
// need to detect incomplete simplification.
ixs_node *ixs_simplify(ixs_ctx *ctx, ixs_node *expr,
                       ixs_node *const *assumptions, size_t n_assumptions);

// Introspection — node must not be NULL:
ixs_tag ixs_node_tag(ixs_node *node);       // returns the node's type tag
int64_t ixs_node_int_val(ixs_node *node);   // IXS_INT value; UB if tag != IXS_INT
uint32_t ixs_node_hash(ixs_node *node);     // precomputed content hash

// Pointer equality (O(1) — hash-consing guarantees that structurally
// identical expressions within the same context share the same pointer).
// Only valid for nodes from the same ixs_ctx. NULL arguments are allowed:
// ixs_same_node(NULL, NULL) returns true, ixs_same_node(NULL, x) returns false.
bool ixs_same_node(ixs_node *a, ixs_node *b);

// Substitution — single-pass: replaces all occurrences of `target` in
// `expr` with `replacement`. target can be any node (symbol, constant,
// subexpression). Matching uses pointer equality (O(1) per node thanks
// to hash-consing). Does NOT re-substitute into the replacement itself
// (no recursive expansion). NULL/sentinel propagation applies to expr,
// target, and replacement as with constructors.
ixs_node *ixs_subs(ixs_ctx *ctx, ixs_node *expr,
                   ixs_node *target, ixs_node *replacement);

// Simultaneous multi-target substitution.  Replaces targets[i] with
// replacements[i] in a single walk.  No replacement is recursed into,
// so {A->B, B->C} on A+B yields B+C, not C+C.  Duplicate targets:
// first matching entry wins.  Thin wrapper around the same engine as
// ixs_subs (nsubs=1 delegates here).
ixs_node *ixs_subs_multi(ixs_ctx *ctx, ixs_node *expr, uint32_t nsubs,
                         ixs_node *const *targets,
                         ixs_node *const *replacements);

// Output — snprintf-like: returns the number of chars that would be written
// (excluding '\0'). If buf is NULL or bufsize is 0, returns the required
// length without writing. Output is always null-terminated when bufsize > 0.
// Sentinel prints as "<error>". NULL expr returns 0 and does not modify buf.
size_t ixs_print(ixs_node *expr, char *buf, size_t bufsize);  // SymPy format
size_t ixs_print_c(ixs_node *expr, char *buf, size_t bufsize); // C code

// Batch: simplify multiple expressions **in place**, sharing the same
// assumption-derived bounds (parsed once, reused for every element).
// Each exprs[i] is overwritten with its simplified form.
// NULL/sentinel entries in exprs are left unchanged. NULL/sentinel
// entries in assumptions are silently skipped.
// OOM: if any simplification hits OOM, ALL entries in exprs are set to
// NULL (the arena is likely exhausted, so no expression is trustworthy).
// Precondition: exprs must be non-NULL when n > 0; assumptions must be
// non-NULL when n_assumptions > 0. No-op when n == 0.
void ixs_simplify_batch(ixs_ctx *ctx, ixs_node **exprs, size_t n,
                         ixs_node *const *assumptions, size_t n_assumptions);

// Entailment check: is a boolean expression provably true or false
// under the given assumptions?  Uses interval propagation only (no
// rewriting).  Returns IXS_CHECK_TRUE, IXS_CHECK_FALSE, or
// IXS_CHECK_UNKNOWN.  Lighter than ixs_simplify for pure truth queries.
typedef enum { IXS_CHECK_TRUE, IXS_CHECK_FALSE, IXS_CHECK_UNKNOWN } ixs_check_result;
ixs_check_result ixs_check(ixs_ctx *ctx, ixs_node *expr,
                           ixs_node *const *assumptions, size_t n_assumptions);

// Expand: distribute MUL over ADD recursively (sum-of-products form).
// Recurses into subexpressions (floor args, piecewise branches, etc.).
// The canonical form keeps products factored; call this when you need
// expanded form for term-by-term analysis.  NULL-safe.
ixs_node *ixs_expand(ixs_ctx *ctx, ixs_node *expr);

// Rule-hit statistics (requires -DIXS_STATS at compile time).
// When disabled, nstats returns 0, stat returns 0/NULL, reset is a no-op.
// Stats are per-context and not thread-safe for a shared context.
size_t   ixs_ctx_nstats(ixs_ctx *ctx);      // distinct rules that fired
uint64_t ixs_ctx_stat(ixs_ctx *ctx, size_t index, const char **name);
void     ixs_ctx_stats_reset(ixs_ctx *ctx);  // zero all counters
```

**Rule-hit statistics** (`-DIXS_STATS`): When compiled with `IXS_STATS`,
the `try_rules()` dispatch automatically records a hit count for each
rule that fires, using the rule name from the `ixs_rule` table. Counts
are stored in a per-context open-addressing hash table (128 slots, keyed
on rule-name pointer identity). Rules not dispatched through `try_rules`
(e.g., ADD/MUL canonicalization) use the `IXS_STAT_HIT(ctx)` macro
directly. CMake: `-DENABLE_STATS=ON`. The Python
binding exposes `ctx.stats()` (returns a `{name: count}` dict) and
`ctx.stats_reset()`. The Python wheel does not enable stats by default;
build from source with `-DENABLE_STATS=ON` for profiling.

Usage pattern:

```c
ixs_ctx *ctx = ixs_ctx_create();

// Assumptions are just boolean expressions built with the same API
ixs_node *T0  = ixs_sym(ctx, "$T0");
ixs_node *T1  = ixs_sym(ctx, "$T1");
ixs_node *M   = ixs_sym(ctx, "M");
ixs_node *N   = ixs_sym(ctx, "N");
ixs_node *K   = ixs_sym(ctx, "K");
ixs_node *assumptions[] = {
    ixs_cmp(ctx, T0, IXS_CMP_GE, ixs_int(ctx, 0)),   // $T0 >= 0
    ixs_cmp(ctx, T0, IXS_CMP_LT, ixs_int(ctx, 256)), // $T0 < 256
    ixs_cmp(ctx, T1, IXS_CMP_GE, ixs_int(ctx, 0)),   // $T1 >= 0
    ixs_cmp(ctx, T1, IXS_CMP_LT, ixs_int(ctx, 4)),   // $T1 < 4
    ixs_cmp(ctx, M,  IXS_CMP_GE, ixs_int(ctx, 1)),   // M >= 1
    ixs_cmp(ctx, N,  IXS_CMP_GE, ixs_int(ctx, 1)),   // N >= 1
    ixs_cmp(ctx, K,  IXS_CMP_GE, ixs_int(ctx, 1)),   // K >= 1
};
size_t n_assumptions = sizeof(assumptions) / sizeof(assumptions[0]);

ixs_node *expr = ixs_parse(ctx, input, strlen(input));
if (!expr) { /* OOM */ return; }
if (ixs_is_parse_error(expr)) {
    fprintf(stderr, "syntax: %s\n", ixs_ctx_error(ctx, 0));
    ixs_ctx_clear_errors(ctx);
    return;
}

ixs_node *simplified = ixs_simplify(ctx, expr, assumptions, n_assumptions);
if (!simplified) { /* OOM */ return; }

if (ixs_is_domain_error(simplified)) {
    for (size_t i = 0; i < ixs_ctx_nerrors(ctx); i++)
        fprintf(stderr, "error: %s\n", ixs_ctx_error(ctx, i));
    ixs_ctx_clear_errors(ctx);
} else {
    char buf[4096];
    ixs_print(simplified, buf, sizeof(buf));
    printf("%s\n", buf);
}

ixs_ctx_destroy(ctx);  // frees everything
```

### Node Introspection API

Nodes are opaque — `struct ixs_node` is internal. The only introspection
today is `ixs_node_tag`, `ixs_node_int_val`, and `ixs_node_hash`.
Consumers need structural destructuring, generic traversal, and DAG
walking.

**Representation mismatch with sympy**: sympy's `Add(a, 2*b, 3)` has
`args = [a, 2*b, 3]` — flat list of sub-expressions. ixsimpl's ADD is
`coeff=3, terms=[{a,1},{b,2}]` — structured. Similarly, sympy MUL has
`Pow(x,-1)` as a first-class arg; ixsimpl stores exponents as `int32_t`
on each mulfactor. The C API exposes ixsimpl's actual structure; the
Python binding layer reconstructs sympy-compatible `.args` if needed.
The accessor API is tied to the current node layout; changes to the
internal representation of ADD/MUL are API-breaking.

#### Type-specific accessors

One function per field. Caller must check `ixs_node_tag` first — calling
the wrong accessor is UB (same contract as `ixs_node_int_val`). No
allocation, no ctx needed.

```c
/* Only valid when tag is IXS_RAT. */
int64_t ixs_node_rat_num(ixs_node *node);
int64_t ixs_node_rat_den(ixs_node *node);

/* Only valid when tag is IXS_SYM.  Pointer valid for ctx lifetime. */
const char *ixs_node_sym_name(ixs_node *node);

/* Only valid when tag is IXS_ADD.  i must be < nterms.
 * ADD = coeff + sum(term_coeff[i] * term[i]). */
ixs_node *ixs_node_add_coeff(ixs_node *node);
uint32_t ixs_node_add_nterms(ixs_node *node);
ixs_node *ixs_node_add_term(ixs_node *node, uint32_t i);
ixs_node *ixs_node_add_term_coeff(ixs_node *node, uint32_t i);

/* Only valid when tag is IXS_MUL.  i must be < nfactors.
 * MUL = coeff * product(base[i] ^ exp[i]). */
ixs_node *ixs_node_mul_coeff(ixs_node *node);
uint32_t ixs_node_mul_nfactors(ixs_node *node);
ixs_node *ixs_node_mul_factor_base(ixs_node *node, uint32_t i);
int32_t ixs_node_mul_factor_exp(ixs_node *node, uint32_t i);

/* Only valid when tag is IXS_FLOOR, IXS_CEIL, or IXS_NOT. */
ixs_node *ixs_node_unary_arg(ixs_node *node);

/* Only valid when tag is IXS_MOD, IXS_MAX, IXS_MIN,
 * IXS_XOR, or IXS_CMP. */
ixs_node *ixs_node_binary_lhs(ixs_node *node);
ixs_node *ixs_node_binary_rhs(ixs_node *node);

/* Only valid when tag is IXS_CMP. */
ixs_cmp_op ixs_node_cmp_op(ixs_node *node);

/* Only valid when tag is IXS_PIECEWISE.  i must be < ncases. */
uint32_t ixs_node_pw_ncases(ixs_node *node);
ixs_node *ixs_node_pw_value(ixs_node *node, uint32_t i);
ixs_node *ixs_node_pw_cond(ixs_node *node, uint32_t i);

/* Only valid when tag is IXS_AND or IXS_OR.  i must be < nargs. */
uint32_t ixs_node_logic_nargs(ixs_node *node);
ixs_node *ixs_node_logic_arg(ixs_node *node, uint32_t i);
```

~20 functions, all trivial one-liners into the internal union fields
(except `ixs_node_unary_arg` which branches on tag internally).

#### Generic child access

```c
/* Number of child node pointers.  Leaves return 0. */
uint32_t ixs_node_nchildren(ixs_node *node);

/* i-th child node.  i must be < ixs_node_nchildren(node). */
ixs_node *ixs_node_child(ixs_node *node, uint32_t i);
```

Single point of truth for "which child-node pointers does this node
have?" — used by the walk and available to FFI callers for custom
traversal without callbacks.  Child order matches the type-specific
accessors:

| Tag | nchildren | order |
|-----|-----------|-------|
| ADD | 1 + 2*nterms | coeff, (term_coeff[0], term[0]), ... |
| MUL | 1 + nfactors | coeff, base[0], base[1], ... |
| binary | 2 | lhs, rhs |
| unary | 1 | arg |
| PW | 2*ncases | (value[0], cond[0]), ... |
| AND/OR | nargs | arg[0], arg[1], ... |
| leaves | 0 | — |

Non-node data (MUL exponents, CMP operator) is not exposed through
this API — use the type-specific accessors for that.  The generic
child API is intended for traversal, not for structural operations
like hashing, equality, or printing.

#### Tree walk

```c
/* Callback action: controls walk behavior after visiting a node. */
typedef enum {
  IXS_WALK_CONTINUE,  /* recurse into children */
  IXS_WALK_SKIP,      /* skip this node's children (pre-order only) */
  IXS_WALK_STOP       /* stop the entire walk */
} ixs_walk_action;

/* Callback must return exactly one of the three values above.
 * Any other return value is undefined behavior. */
typedef ixs_walk_action (*ixs_visit_fn)(ixs_node *node, void *userdata);

/* Pre-order: visit node, then recurse into children.
 * Returns root on completion, the stopping node on STOP, NULL on OOM.
 * NULL root returns NULL (no-op).
 * Sentinels (ERROR, PARSE_ERROR) are visited as leaves; the callback
 * must check ixs_node_tag before using type-specific accessors.
 * SKIP prevents descent into children. */
ixs_node *ixs_walk_pre(ixs_ctx *ctx, ixs_node *root,
                       ixs_visit_fn fn, void *userdata);

/* Post-order: recurse into children, then visit node.
 * Same return/NULL/sentinel semantics as ixs_walk_pre.
 * SKIP is a no-op in post-order (children already visited). */
ixs_node *ixs_walk_post(ixs_ctx *ctx, ixs_node *root,
                        ixs_visit_fn fn, void *userdata);
```

**Return value**:
- `root` — walk completed, all reachable nodes visited.
- other non-NULL — callback returned STOP on that node.
- `NULL` — root was NULL (or OOM if a future iterative implementation
  exhausts its scratch stack).

The caller distinguishes NULL-root from OOM because they know what they
passed. The edge case where STOP fires on root itself (returns `root`,
same as completion) is detectable via userdata if it matters.

No dedup: the walk follows the tree shape, not the DAG. Hash-consed
graphs with heavy sharing can have exponentially more tree paths than
unique nodes, so callers doing exhaustive collection (e.g.
`free_symbols`) should maintain their own visited set keyed on pointer
identity. Callers that don't need dedup (printing, conversion) get the
fast path without paying for a hash set.

`ctx` is accepted so that a future iterative implementation can use the
scratch arena for an explicit stack, avoiding stack overflow on deep trees.
The current implementation is recursive and never returns NULL for non-NULL
root, but is subject to stack depth limits on very deep trees (practical
limit ~10k depth). Bead 4-3j0 tracks bounding recursion depth.

Use cases:
- `ixs_walk_pre`: top-down — pattern matching with subtree-skipping,
  symbol-dependency checks ("does symbol X appear?" — stop early),
  conversion to external representations. Replaces sympy's
  `preorder_traversal`.
- `ixs_walk_post`: bottom-up — numeric evaluation, size/depth
  computation, any transform that needs children results first.

#### Contracts

- **NULL node**: passing NULL to any type-specific accessor is UB.
- **Wrong tag**: calling a type-specific accessor on the wrong tag is UB
  (same as `ixs_node_int_val`).
- **Invalid index**: out-of-range index arguments are UB. Implementations
  contain `assert()` for debug builds but no runtime checks in release.
- **Walk preconditions**: `ctx` and `fn` must be non-NULL. `root` may be
  NULL (returns NULL). The callback must not destroy `ctx`. All other
  operations (constructors, simplify, subs, parse) are safe — nodes are
  immutable, and scratch arena save/restore nests correctly via LIFO.

## Performance Design

### Why this can be 100-1000x faster than SymPy

1. **No Python overhead**: Every SymPy operation goes through Python's object
   model, GIL, reference counting, and dynamic dispatch. A single `floor(x+1)`
   in SymPy creates multiple Python objects and calls `__new__`, `__init__`,
   `__hash__`, `__eq__` in Python. In C, it's a hash table lookup and a
   pointer.

2. **Hash-consing eliminates redundant work**: The 609 expressions share most
   of their subexpressions. In SymPy, each expression re-creates and
   re-simplifies common subtrees. With hash-consing, `ceiling(M/128)` is
   created once, stored once, and every expression that uses it gets the
   same pointer for free.

3. **Arena allocation**: Zero per-node malloc/free overhead. Excellent cache
   locality.

4. **No wasted generality**: SymPy checks for trigonometric identities,
   logarithmic simplification, polynomial factoring, etc. on every expression.
   This library checks only for the ~30 rules that actually apply.

5. **Batch mode**: Processing all 609 expressions in one context means the
   hash-consing table is warmed up, and later expressions (which are variants
   of earlier ones) are nearly free.

### Target performance

| Metric | SymPy (current) | Target |
|---|---|---|
| 609 expressions total | 41.4 s | < 50 ms |
| Average per expression | 68 ms | < 0.1 ms |
| Worst case | 388 ms | < 5 ms |

### Memory budget

Estimated peak memory for one `ixs_ctx` processing all 609 corpus expressions
in batch mode (the primary use case):

- Unique nodes (after hash-consing): ~10,000-30,000
- Node size: ~64 bytes average
- Hash table: ~128 KB
- Arena: ~2-4 MB
- Error list, symbol table: negligible
- **Total: < 8 MB per context**

## Build and CI

**CMake options**:

- `ENABLE_ASAN` (default `OFF`): Build with AddressSanitizer
  (`-fsanitize=address -fno-omit-frame-pointer`).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build
ctest --test-dir build
```

**GitHub Actions** (`.github/workflows/ci.yml`) runs on every push to
`master` and on pull requests:

| Job | What it does |
|-----|-------------|
| Release | Build + test with optimizations |
| Debug + ASAN | Build + test under AddressSanitizer |
| Python bindings + fuzz | Build extension with ASAN, run Hypothesis fuzz tests |
| Lint | `pre-commit` (clang-format, whitespace, REUSE) |
| REUSE compliance | License metadata check |

## File Structure

```
ixsimpl/
├── include/
│   └── ixsimpl.h            # public C API (single header)
├── src/
│   ├── arena.c               # arena allocator
│   ├── arena.h
│   ├── rational.c            # exact rational arithmetic
│   ├── rational.h
│   ├── node.c                # node creation, hash-consing, canonical forms
│   ├── node.h
│   ├── parser.c              # recursive descent parser
│   ├── parser.h
│   ├── simplify.c            # rewrite rules engine
│   ├── simplify.h
│   ├── expand.c              # MUL-over-ADD distribution
│   ├── expand.h
│   ├── bounds.c              # bound storage, propagation, assumption extraction
│   ├── bounds.h
│   ├── interval.c            # interval arithmetic (add, mul, recip, intersect)
│   ├── interval.h
│   ├── print.c               # output formatters
│   ├── print.h
│   └── ctx.c                 # context management, public API impl
├── bindings/
│   ├── cpp/
│   │   └── ixsimpl.hpp       # C++ header-only wrapper
│   └── python/
│       ├── ixsimpl.pyi       # type stubs
│       └── _ixsimpl.c        # CPython extension module
├── test/
│   ├── test_rational.c
│   ├── test_parser.c
│   ├── test_simplify.c
│   ├── test_expand.c           # expand (MUL-over-ADD distribution) tests
│   ├── test_edge_cases.c      # edge cases, error paths, sentinel propagation
│   ├── test_corpus.c          # run against the 609-expression corpus
│   ├── test_python.py         # Python binding tests
│   ├── corpus.txt             # the reference expressions
│   ├── corpus_expected.txt    # pre-generated SymPy simplified outputs
│   └── corpus_assumptions.txt # shared assumption set for corpus tests
├── bench/
│   └── bench_corpus.c        # benchmark: time all 609 expressions
├── scripts/
│   ├── gen_expected.py        # generate corpus_expected.txt via SymPy
│   └── requirements-gen.txt   # pinned SymPy version for generation
├── CMakeLists.txt
├── pyproject.toml             # Python package build (PEP 517)
└── LICENSE                    # MIT
```

Estimated total: 5,000-8,000 lines of C, ~500 lines C++ header, ~800 lines
Python extension.

## Language Bindings

### C++ Binding — `ixsimpl.hpp`

A thin header-only RAII wrapper over the C API. No additional runtime cost.

```cpp
namespace ixs {

class Context {
    ixs_ctx *ctx_;
public:
    Context() : ctx_(ixs_ctx_create()) {
        if (!ctx_) throw std::bad_alloc();
    }
    ~Context() { ixs_ctx_destroy(ctx_); }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ixs_ctx *raw() const { return ctx_; }
};

class Expr {
    ixs_ctx *ctx_;
    ixs_node *node_;
public:
    Expr(ixs_ctx *ctx, ixs_node *node) : ctx_(ctx), node_(node) {}

    static Expr parse(Context &ctx, std::string_view input) {
        return {ctx.raw(), ixs_parse(ctx.raw(), input.data(), input.size())};
    }
    static Expr sym(Context &ctx, const char *name) {
        return {ctx.raw(), ixs_sym(ctx.raw(), name)};
    }
    static Expr integer(Context &ctx, int64_t v) {
        return {ctx.raw(), ixs_int(ctx.raw(), v)};
    }

    Expr simplify(const Expr *assumptions, size_t n_assumptions) const {
        std::vector<ixs_node*> raw(n_assumptions);
        for (size_t i = 0; i < n_assumptions; i++)
            raw[i] = assumptions[i].raw();
        return {ctx_, ixs_simplify(ctx_, node_, raw.data(), raw.size())};
    }
    Expr simplify() const {
        return {ctx_, ixs_simplify(ctx_, node_, nullptr, 0)};
    }
    Expr subs(Expr target, Expr repl) const {
        return {ctx_, ixs_subs(ctx_, node_, target.node_, repl.node_)};
    }

    Expr operator+(Expr rhs) const { return {ctx_, ixs_add(ctx_, node_, rhs.node_)}; }
    Expr operator-(Expr rhs) const { return {ctx_, ixs_sub(ctx_, node_, rhs.node_)}; }
    Expr operator-()         const { return {ctx_, ixs_neg(ctx_, node_)}; }
    Expr operator*(Expr rhs) const { return {ctx_, ixs_mul(ctx_, node_, rhs.node_)}; }
    Expr operator/(Expr rhs) const { return {ctx_, ixs_div(ctx_, node_, rhs.node_)}; }
    bool operator==(Expr rhs) const { return ixs_same_node(node_, rhs.node_); }

    std::string str() const {
        if (!node_) return {};
        size_t n = ixs_print(node_, nullptr, 0);
        std::string s(n + 1, '\0');
        ixs_print(node_, s.data(), n + 1);
        s.resize(n);
        return s;
    }

    bool is_null() const { return node_ == nullptr; }
    bool is_error() const { return node_ && ixs_is_error(node_); }
    bool is_parse_error() const { return node_ && ixs_is_parse_error(node_); }
    bool is_domain_error() const { return node_ && ixs_is_domain_error(node_); }
    explicit operator bool() const { return node_ && !ixs_is_error(node_); }

    ixs_node *raw() const { return node_; }
    ixs_ctx *raw_ctx() const { return ctx_; }
};

inline Expr floor(Expr x) { return {x.raw_ctx(), ixs_floor(x.raw_ctx(), x.raw())}; }
inline Expr ceil(Expr x)  { return {x.raw_ctx(), ixs_ceil(x.raw_ctx(), x.raw())}; }
inline Expr mod(Expr a, Expr b) { return {a.raw_ctx(), ixs_mod(a.raw_ctx(), a.raw(), b.raw())}; }
inline Expr max(Expr a, Expr b) { return {a.raw_ctx(), ixs_max(a.raw_ctx(), a.raw(), b.raw())}; }
inline Expr min(Expr a, Expr b) { return {a.raw_ctx(), ixs_min(a.raw_ctx(), a.raw(), b.raw())}; }
inline Expr xor_(Expr a, Expr b) { return {a.raw_ctx(), ixs_xor(a.raw_ctx(), a.raw(), b.raw())}; }
inline Expr pw(std::initializer_list<std::pair<Expr, Expr>> branches); // piecewise

} // namespace ixs
```

Key properties:
- `Context` owns the arena; RAII cleans up everything on destruction.
  Constructor throws `std::bad_alloc` on OOM.
- `Expr` is a lightweight value type (two pointers). Cheap to copy.
  **`Expr` must not outlive its `Context`** — destroying a `Context`
  invalidates all `Expr` values created from it (dangling pointers).
- `operator bool()` returns `true` only for valid, non-error expressions.
  `is_null()` checks OOM, `is_parse_error()`/`is_domain_error()` check
  specific sentinels, `is_error()` checks either.
- Operator overloading for natural expression building.
- No heap allocations in expression construction or simplification beyond
  what the C library does internally. (`str()` allocates a `std::string`.)
- NULL and sentinel propagate through operators (same as C API).
- **Cross-context contract** applies: all `Expr` values passed to an
  operation (including assumptions in `simplify()`) must belong to the
  same `Context`. Mixing contexts is undefined behavior.

### Python Binding — `_ixsimpl.c`

A CPython C extension module (no pybind11/ctypes dependency). Exposes two
Python types: `Context` and `Expr`.

```python
import ixsimpl

ctx = ixsimpl.Context()

T0 = ctx.sym("$T0")
M  = ctx.sym("M")
assumptions = [T0 >= 0, T0 < 256, M >= 1]

expr = ctx.parse("128*floor($T0/64) + 4*floor(Mod($T0, 64)/16)")
result = expr.simplify(assumptions=assumptions)
print(result)              # SymPy-compatible string
print(result.to_c())       # C code string

# Programmatic construction
x = ctx.sym("x")
y = ctx.sym("y")
e = ixsimpl.floor(x + y) + 3
print(e.simplify(assumptions=assumptions))

# Module-level convenience functions
e2 = ixsimpl.mod(x, 4)
e3 = ixsimpl.max_(x, y)   # trailing _ avoids shadowing builtin max
e4 = ixsimpl.min_(x, y)
e5 = ixsimpl.pw((x, x >= 0), (-x, ctx.true_()))  # piecewise
cond = ixsimpl.and_(x >= 0, y >= 0)               # boolean combinators
cond2 = ixsimpl.or_(x >= 0, y >= 0)
cond3 = ixsimpl.not_(x >= 0)
e5 = ixsimpl.ceil(x / 4)

# Batch
exprs = [ctx.parse(line) for line in lines]
ctx.simplify_batch(exprs, assumptions=assumptions)
```

Implementation:
- `_ixsimpl.c` is a single-file CPython extension using the stable ABI
  (`Py_LIMITED_API`), so one `.so` works across Python 3.7+.
- `Context` Python object wraps `ixs_ctx*`; destructor calls
  `ixs_ctx_destroy`.
- `Expr` Python object holds a reference to its `Context` (preventing
  premature GC) and wraps `ixs_node*`.
- `__repr__` and `__str__` call `ixs_print`. Sentinel prints as `"<error>"`.
- `__int__` returns the integer value if the node is `IXS_INT`; raises
  `TypeError` otherwise. Used by `eval_ixs` for numerical evaluation.
- `__eq__` is pointer comparison (O(1) via hash-consing).
- `__hash__` returns the node's precomputed hash.
- `Expr.is_error` property — `True` for either sentinel.
- `Expr.is_parse_error` / `Expr.is_domain_error` — specific checks.
- `Context.errors` property — returns list of error strings; `Context.clear_errors()` resets.
- Operator overloading: `__add__`, `__mul__`, `__sub__`, `__neg__`,
  `__ge__`, `__gt__`, `__le__`, `__lt__`, `__eq__` (comparisons return
  `Expr` nodes, not Python `bool`, so they can be used as assumptions).
- `Context.int_(val)` creates an `IXS_INT` node (wraps `ixs_int`).
- NULL (OOM) raises `MemoryError`. Sentinel propagates as a regular Expr.
- Module-level functions: `ixsimpl.floor()`, `ixsimpl.ceil()`,
  `ixsimpl.mod()`, `ixsimpl.max_()`, `ixsimpl.min_()`, `ixsimpl.xor_()`,
  `ixsimpl.and_()`, `ixsimpl.or_()`, `ixsimpl.not_()`,
  `ixsimpl.pw((val, cond), ...)`.
  Trailing underscore on `max_`/`min_`/`xor_`/`and_`/`or_`/`not_` avoids
  shadowing Python builtins.
- `pyproject.toml` builds the extension; no runtime dependencies.

This binding adds ~800 lines of C and is the primary interface for testing
against SymPy (run both, compare outputs).

## Implementation Plan

### Phase 1: Foundation

- Arena allocator
- Rational arithmetic with full test coverage (floored division/Mod)
- Node types, hash-consing table, symbol interning
- Canonical Add/Mul construction with flattening and term collection
- Basic constant folding
- SymPy-format printer
- Generate `test/corpus_expected.txt` by running SymPy on all 609 corpus
  expressions (one-time script `scripts/gen_expected.py`, checked in). The
  script reads `corpus.txt` and `corpus_assumptions.txt`, applies the
  `Mod(p, q, evaluate=False)` workaround (see SymPy #28744 section), and
  writes one simplified expression per line. The SymPy version is pinned in
  `scripts/requirements-gen.txt` (e.g., `sympy==1.14.0`). This is the ground
  truth for all subsequent phases.

**Milestone**: Can construct `3*x + 2*x + 1` and get `5*x + 1`. Corpus
expected outputs are available for comparison.

### Phase 2: Parser + Floor/Ceil/Mod

- Recursive descent parser for the full grammar
- `floor()`, `ceiling()` constructors with basic rules
- `Mod()` constructor with basic rules
- `Max()`, `Min()`, `xor()` constructors

**Milestone**: Can parse and round-trip all 609 expressions. Simplification of
constant subexpressions works.

### Phase 3: Piecewise + Boolean

- Piecewise node type with condition simplification
- Boolean algebra (And/Or/Not) with basic simplification
- Comparison normalization
- Piecewise propagation through arithmetic

**Milestone**: Expressions with Piecewise are simplified correctly. Many
expressions in the corpus should already simplify significantly.

### Phase 4: Advanced Rules + Bound Analysis

- Bound tracking infrastructure
- Domain-aware floor/ceil/Mod rules
- Batch simplification mode
- Full test against corpus
- Benchmark against SymPy baseline

**Milestone**: All 609 expressions produce correct results. Performance target
met (< 50ms total).

### Phase 5: Bindings

- C++ header-only wrapper (`ixsimpl.hpp`)
- CPython extension module (`_ixsimpl.c`)
- Python test suite comparing output against SymPy
- `pyproject.toml` for pip-installable package

**Milestone**: `pip install .` works; Python tests pass against SymPy oracle.

### Phase 6: Hardening + Integration

- Fuzz testing (generate random expressions, verify against SymPy)
- Edge cases: overflow, division by zero, degenerate Piecewise
- C code output mode
- Documentation
- Integration with the host compiler

## Testing Strategy

1. **Unit tests**: Each rule in isolation (test_rational, test_simplify).
2. **Corpus test**: Parse all 609 expressions, simplify, verify output matches
   SymPy's output (or is provably equivalent via random evaluation).

   **Corpus file format** — `corpus.txt` uses one expression per non-blank
   line, prefixed with timing info: `simplify time: X.XXXXs: <expression>`.
   The loader strips the prefix up to and including the second `: ` to extract
   the raw expression. Blank lines are skipped.
   `corpus_expected.txt` contains one simplified expression per line,
   aligned 1:1 with the non-blank lines of `corpus.txt` (same count, same
   order). Lines that SymPy cannot simplify are copied verbatim.

   **Corpus assumptions** — The corpus test uses the following fixed
   assumption set (matching the kernel's known variable ranges):

   ```
   $T0 >= 0, $T0 < 256
   $T1 >= 0, $T1 < 4
   $T2 >= 0
   $WG0 >= 0, $WG1 >= 0
   $GPR_NUM >= 0
   M >= 1, N >= 1, K >= 1
   _M_div_32 >= 0, _N_div_32 >= 0, _K_div_256 >= 0
   $ARGK >= 0
   $MMA_ACC >= 0, $MMA_LHS_SCALE >= 0, $MMA_RHS_SCALE >= 0
   $MMA_SCALE_FP4 >= 0
   $index0 >= 0, $index1 >= 0
   _aligned >= 0
   ```

   These are stored in `test/corpus_assumptions.txt` (one assumption per
   line, same syntax as the parser accepts) and loaded by the corpus test
   and the `corpus_expected.txt` generation script. Using a shared
   assumption file ensures SymPy and ixsimpl see identical constraints.
3. **Equivalence oracle**: For any simplified expression `s` from input `e`,
   verify `s == e` by substituting random values for all variables and
   checking numerical equality. This catches bugs without requiring
   exact output matching.
4. **Negative/error-path tests**: Verify correct behavior for invalid inputs:
   - Parse errors: `"foo bar +"` → `IXS_PARSE_ERROR`
   - Depth limit: deeply nested input → `IXS_PARSE_ERROR`
   - Domain error in parsed input: `"1/0 + x"` → `IXS_ERROR`
   - Division by zero via API: `ixs_mod(ctx, x, zero)` → `IXS_ERROR`
   - Integer literal overflow: `"99999999999999999999"` → `IXS_ERROR`
   - `ixs_rat(ctx, 1, 0)` → `IXS_ERROR`
   - NULL propagation: `ixs_add(ctx, NULL, x)` → NULL
   - Sentinel propagation: `ixs_floor(ctx, sentinel)` → same sentinel, no new error
   - Piecewise sentinel in dead branch: drops cleanly
   - `ixs_is_error` true for both, `ixs_is_parse_error` / `ixs_is_domain_error` specific
5. **Fuzz testing**: Hypothesis-based (see below).
6. **Benchmark**: Time all 609 expressions, compare against SymPy baseline.
   Track regressions in CI.  `bench/bench_sympy.py` runs ixsimpl vs
   `sympy.simplify` and `sympy.cancel` on the full corpus.  Typical
   results: ixsimpl ~23 ms total vs sympy.cancel ~25 s (~1000x) and
   sympy.simplify ~1000 s (~45000x).

### Fuzz Testing with Hypothesis

Property-based fuzz testing uses Python's
[Hypothesis](https://hypothesis.readthedocs.io/) library to generate random
expressions within the grammar, simplify them with both ixsimpl and SymPy,
and verify equivalence via numerical evaluation.

```python
import random
from hypothesis import given, strategies as st, settings, assume
import sympy
import ixsimpl

# Strategy: generate random expression trees within the grammar
sym_names = st.sampled_from(["x", "y", "z", "w"])
small_ints = st.integers(min_value=-64, max_value=64)
pos_ints = st.integers(min_value=1, max_value=32)

@st.composite
def expressions(draw, max_depth=4):
    if max_depth <= 0 or draw(st.booleans()):
        return draw(st.one_of(sym_names, small_ints))
    ops = _OPS_WITH_PW if include_piecewise else _OPS_BASE
    op = draw(st.sampled_from(ops))
    a = draw(expressions(max_depth=max_depth - 1))
    if op in ("floor", "ceiling"):
        return (op, a)
    b = draw(expressions(max_depth=max_depth - 1))
    if op == "mod":
        b = draw(pos_ints)  # modulus must be positive
    if op == "div":
        b = draw(pos_ints)  # divisor must be nonzero
    if op == "piecewise":
        cond = draw(conditions(max_depth=max_depth - 1))
        default = draw(expressions(max_depth=max_depth - 1))
        return ("piecewise", a, cond, default)
    return (op, a, b)

@st.composite
def conditions(draw, max_depth=2):
    if max_depth <= 0 or draw(st.booleans()):
        a = draw(expressions(max_depth=2))
        b = draw(expressions(max_depth=2))
        op = draw(st.sampled_from([">=", ">", "<=", "<", "==", "!="]))
        return ("cmp", op, a, b)
    combiner = draw(st.sampled_from(["and", "or", "not"]))
    c1 = draw(conditions(max_depth=max_depth - 1))
    if combiner == "not":
        return ("not", c1)
    c2 = draw(conditions(max_depth=max_depth - 1))
    return (combiner, c1, c2)

def to_sympy(tree):
    """Convert expression tree to SymPy expression."""
    ...

def to_ixsimpl(ctx, tree):
    """Convert expression tree to ixsimpl expression."""
    ...

def eval_expr(tree, env):
    """Evaluate expression tree numerically using Python arithmetic.
    Raises ZeroDivisionError/ValueError on undefined operations."""
    ...

def eval_ixs(expr, ctx, env):
    """Evaluate ixsimpl Expr by substituting all variables via subs,
    then reading the resulting integer constant.
    Returns int or raises ValueError if result is not a constant/integer."""
    result = expr.subs({name: ctx.int_(val) for name, val in env.items()})
    if result.is_error:
        raise ValueError("sentinel")
    # Expr.__int__ returns IXS_INT value; raises TypeError for non-INT nodes
    # (including IXS_RAT — valid index expressions always reduce to integers)
    try:
        return int(result)
    except TypeError:
        raise ValueError(f"result is not an integer constant: {result}")

_VARS = ["x", "y", "z", "w"]

def _env_st(lo=1, hi=100):
    return st.fixed_dictionaries({v: st.integers(lo, hi) for v in _VARS})

@given(expr=expressions(), envs=st.lists(_env_st(0, 100), min_size=1, max_size=10))
@settings(max_examples=10000)
def test_simplify_matches_numerical(expr, envs):
    """Simplification preserves semantics: evaluate original and simplified
    at random points, check they agree."""
    ctx = ixsimpl.Context()
    ixs_expr = to_ixsimpl(ctx, expr)
    assume(not ixs_expr.is_error)  # skip if construction hit domain error
    ixs_simplified = ixs_expr.simplify()
    assume(not ixs_simplified.is_error)  # skip if simplification hit error

    for env in envs:
        try:
            orig = eval_expr(expr, env)
        except (ZeroDivisionError, ValueError, TypeError):
            continue  # skip points where original is undefined
        simp = eval_ixs(ixs_simplified, ctx, env)
        assert orig == simp, f"Mismatch: {orig} != {simp} at {env}"

@given(expr=expressions(), envs=st.lists(_env_st(), min_size=1, max_size=10))
@settings(max_examples=5000)
def test_matches_sympy(expr, envs):
    """Cross-check against SymPy: both should produce numerically
    equivalent results."""
    ctx = ixsimpl.Context()
    ixs_result = to_ixsimpl(ctx, expr).simplify()
    assume(not ixs_result.is_error)

    sp_expr = to_sympy(expr)
    sp_result = sympy.simplify(sp_expr)

    for env in envs:
        try:
            ixs_val = eval_ixs(ixs_result, ctx, env)
            sp_val = int(sp_result.subs({sympy.Symbol(k): v for k, v in env.items()}))
        except (ZeroDivisionError, ValueError, TypeError):
            continue
        assert ixs_val == sp_val, f"Divergence at {env}"
```

**Known SymPy bug to work around**: SymPy issue
[#28744](https://github.com/sympy/sympy/issues/28744) — `Mod` incorrectly
squares inner `Mod` subexpressions when the variable has `integer=True`
assumption. The bug is in `sympy/core/mod.py` lines 166-172: factors of type
`Mod` are duplicated into both `mod_l` and `non_mod_l`, causing them to
appear squared. The fix (merged to `master` in Dec 2025) has not been included
in any SymPy release yet (latest release is 1.14.0, April 2025).

Concrete workarounds for the fuzz test oracle:

1. **Reconstruct Mod with `evaluate=False`** when using SymPy as oracle.
   The bug is in SymPy's auto-evaluation of `Mod()` — when `integer=True`
   symbols are present, `Mod(k*Mod(x,n), m)` silently squares the inner
   Mod factor. The fix (from IREE Wave's `_bounds_simplify_once`) is to
   rebuild Mod nodes with `sympy.Mod(p, q, evaluate=False)` after
   simplifying their arguments, bypassing the buggy code path. This lets
   us keep `integer=True` on symbols (which is needed for other SymPy
   rewrites to fire) while avoiding the specific Mod bug. Non-Mod nodes
   are reconstructed normally via `expr.func(*simplified_args)`.
2. **Primary oracle is numerical evaluation**, not SymPy's symbolic output.
   For each test case, substitute 10+ random integer values into both the
   original and simplified expressions and check equality. This catches bugs
   in both SymPy and ixsimpl independently.
3. **Pin SymPy version** for `corpus_expected.txt` generation and document it.
   When a new SymPy release includes the fix, re-generate and note the
   version.
4. **Hypothesis `@example` decorators** for the specific pattern from #28744:
   `Mod(2*Mod(x, 3), 5)` — ensure this is always tested and known to
   diverge from buggy SymPy.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Simplification produces wrong results | Critical | Numerical equivalence oracle; fuzz testing |
| 64-bit rational overflow | Medium | Checked arithmetic → sentinel + error list; 128-bit fallback if needed later |
| New expression patterns in future workloads | Medium | Grammar is extensible; add new node types as needed |
| Hash-consing table becomes bottleneck | Low | Linear probing with power-of-2 sizing; rehash threshold 70% |
| Simplification rules interact badly (infinite loops) | Medium | Fixed-point iteration limit (64); monotonically size-reducing rules |

## Future Ideas

- **Arena-backed hash table.** Replace the `calloc`/`free`-managed hash
  table with arena-allocated non-contiguous blocks (sizes S, S, 2S, 4S,
  8S, ...).  Old blocks remain live after resize (total capacity doubles by
  appending one new block), so there is zero arena waste.  Linear index
  decomposition to (block, offset) is O(1) via highest-bit.  Trade-off:
  extra indirection per probe and cross-block cache misses on long chains.
  Enables a fully `malloc`-free library where `ixs_ctx_destroy` is just
  two `ixs_arena_destroy` calls.

## Non-Goals

- General symbolic algebra (polynomial factoring, GCD, etc.)
- Floating-point arithmetic
- Calculus (differentiation, integration, series)
- Trigonometric or transcendental functions
- Matrix operations
- Equation solving
- Bindings beyond C++ and Python
