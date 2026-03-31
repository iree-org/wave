/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * ixsimpl.h -- public API for the index expression simplifier.
 *
 * All nodes are hash-consed and owned by their ixs_ctx.  Nodes from
 * different contexts must never be mixed in the same operation.
 *
 * Error model (three tiers, checked in this order):
 *   NULL         -- out of memory.  Propagates: any op receiving NULL
 *                   returns NULL.
 *   PARSE_ERROR  -- malformed input.  Propagates through arithmetic.
 *   ERROR        -- domain error (e.g. division by zero).  Same.
 * Check with ixs_is_error / ixs_is_parse_error / ixs_is_domain_error.
 * Human-readable messages accumulate in the context error list.
 */

#ifndef IXSIMPL_H
#define IXSIMPL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ixs_ctx ixs_ctx;
typedef struct ixs_node ixs_node;

typedef enum {
  IXS_CMP_GT,
  IXS_CMP_GE,
  IXS_CMP_LT,
  IXS_CMP_LE,
  IXS_CMP_EQ,
  IXS_CMP_NE
} ixs_cmp_op;

/* --- Context lifecycle ------------------------------------------------- */

/* Create a new context.  Returns NULL on allocation failure. */
ixs_ctx *ixs_ctx_create(void);

/* Destroy ctx and free all nodes allocated within it. */
void ixs_ctx_destroy(ixs_ctx *ctx);

/* --- Error list -------------------------------------------------------- */

/* Number of accumulated error messages. */
size_t ixs_ctx_nerrors(ixs_ctx *ctx);

/* Retrieve the i-th error message (0-based).  Pointer valid until next
 * mutating call on ctx. */
const char *ixs_ctx_error(ixs_ctx *ctx, size_t index);

/* Discard all accumulated errors. */
void ixs_ctx_clear_errors(ixs_ctx *ctx);

/* --- Sentinel checks --------------------------------------------------- */

/* True if node is any sentinel (PARSE_ERROR or ERROR). */
bool ixs_is_error(ixs_node *node);

/* True if node is specifically a parse error sentinel. */
bool ixs_is_parse_error(ixs_node *node);

/* True if node is specifically a domain error sentinel. */
bool ixs_is_domain_error(ixs_node *node);

/* --- Parse ------------------------------------------------------------- */

/* Parse a SymPy-style expression from input[0..len-1].
 * input must be NUL-terminated at or before input[len].
 * Returns the simplified AST, PARSE_ERROR on bad syntax, NULL on OOM. */
ixs_node *ixs_parse(ixs_ctx *ctx, const char *input, size_t len);

/* --- Constructors ------------------------------------------------------ */

/* All constructors return NULL on OOM.  Node arguments must belong to ctx. */

ixs_node *ixs_int(ixs_ctx *ctx, int64_t val);
ixs_node *ixs_rat(ixs_ctx *ctx, int64_t p, int64_t q);
ixs_node *ixs_sym(ixs_ctx *ctx, const char *name);

ixs_node *ixs_add(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_mul(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_neg(ixs_ctx *ctx, ixs_node *a);
ixs_node *ixs_sub(ixs_ctx *ctx, ixs_node *a, ixs_node *b);

/* Exact rational division: a/b where b != 0.  Returns ERROR on b == 0. */
ixs_node *ixs_div(ixs_ctx *ctx, ixs_node *a, ixs_node *b);

ixs_node *ixs_floor(ixs_ctx *ctx, ixs_node *x);
ixs_node *ixs_ceil(ixs_ctx *ctx, ixs_node *x);

/* Floored modulo (Python/SymPy semantics).  Returns ERROR on b == 0. */
ixs_node *ixs_mod(ixs_ctx *ctx, ixs_node *a, ixs_node *b);

ixs_node *ixs_max(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_min(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_xor(ixs_ctx *ctx, ixs_node *a, ixs_node *b);

/* Piecewise: n branches.  values[i] is returned when conds[i] is true;
 * last branch is the default (conds[n-1] should be ixs_true). */
ixs_node *ixs_pw(ixs_ctx *ctx, uint32_t n, ixs_node **values, ixs_node **conds);

ixs_node *ixs_cmp(ixs_ctx *ctx, ixs_node *a, ixs_cmp_op op, ixs_node *b);
ixs_node *ixs_and(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_or(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
ixs_node *ixs_not(ixs_ctx *ctx, ixs_node *a);
ixs_node *ixs_true(ixs_ctx *ctx);
ixs_node *ixs_false(ixs_ctx *ctx);

/* --- Entailment checking ----------------------------------------------- */

typedef enum {
  IXS_CHECK_TRUE,
  IXS_CHECK_FALSE,
  IXS_CHECK_UNKNOWN
} ixs_check_result;

/* Check whether a comparison is provably true or false given the
 * assumptions, using interval propagation.  expr must be a CMP node
 * in normalized form (lhs op 0) -- this is automatic when constructed
 * via ixs_cmp().  Returns UNKNOWN when bounds are insufficient, when
 * expr is not a CMP, or on OOM.  Lighter than ixs_simplify: no
 * rewriting, just bounds setup + interval check. */
ixs_check_result ixs_check(ixs_ctx *ctx, ixs_node *expr,
                           ixs_node *const *assumptions, size_t n_assumptions);

/* --- Simplification ---------------------------------------------------- */

/* Simplify expr under the given assumptions (array of CMP/AND/OR nodes).
 * Pass NULL/0 for no assumptions.  Returns simplified node, or sentinel. */
ixs_node *ixs_simplify(ixs_ctx *ctx, ixs_node *expr,
                       ixs_node *const *assumptions, size_t n_assumptions);

/* Simplify exprs[0..n-1] in place, sharing the same assumption set.
 * Each element is replaced by its simplified form.  On OOM, all
 * elements are set to NULL.  NULL or sentinel entries are skipped.
 * Bounds are parsed from assumptions once and reused across all elements. */
void ixs_simplify_batch(ixs_ctx *ctx, ixs_node **exprs, size_t n,
                        ixs_node *const *assumptions, size_t n_assumptions);

/* Distribute MUL over ADD (expand products of sums into sums of products).
 * Recurses into subexpressions (floor args, piecewise branches, etc.).
 * Powers are expanded by repeated multiplication (capped at exponent 64).
 * NULL-safe. */
ixs_node *ixs_expand(ixs_ctx *ctx, ixs_node *expr);

/* --- Comparison and substitution --------------------------------------- */

/* Pointer equality on hash-consed nodes.  Safe to call with NULL. */
bool ixs_same_node(ixs_node *a, ixs_node *b);

/* Return expr with all occurrences of target replaced by replacement.
 * target can be any node (symbol, subexpression, constant, etc.).
 * Uses pointer equality (hash-consed), so matching is O(1) per node. */
ixs_node *ixs_subs(ixs_ctx *ctx, ixs_node *expr, ixs_node *target,
                   ixs_node *replacement);

/* Simultaneous multi-target substitution.  Replaces targets[i] with
 * replacements[i] in a single pass.  No replacement is recursed into,
 * so {A->B, B->C} applied to A+B yields B+C, not C+C.
 * Duplicate targets: first matching entry wins. */
ixs_node *ixs_subs_multi(ixs_ctx *ctx, ixs_node *expr, uint32_t nsubs,
                         ixs_node *const *targets,
                         ixs_node *const *replacements);

/* --- Output ------------------------------------------------------------ */

/* Print in SymPy-compatible syntax.  Returns bytes written (excl. NUL).
 * Output is truncated if bufsize is insufficient. */
size_t ixs_print(ixs_node *expr, char *buf, size_t bufsize);

/* Print in C syntax where possible; falls back to SymPy style. */
size_t ixs_print_c(ixs_node *expr, char *buf, size_t bufsize);

/* --- Introspection ----------------------------------------------------- */

typedef enum {
  IXS_INT,
  IXS_RAT,
  IXS_SYM,
  IXS_ADD,
  IXS_MUL,
  IXS_FLOOR,
  IXS_CEIL,
  IXS_MOD,
  IXS_PIECEWISE,
  IXS_MAX,
  IXS_MIN,
  IXS_XOR,
  IXS_CMP,
  IXS_AND,
  IXS_OR,
  IXS_NOT,
  IXS_TRUE,
  IXS_FALSE,
  IXS_ERROR,
  IXS_PARSE_ERROR
} ixs_tag;

/* All introspection functions require non-NULL node. */

ixs_tag ixs_node_tag(ixs_node *node);

/* Only valid when tag is IXS_INT. */
int64_t ixs_node_int_val(ixs_node *node);

/* Structural hash (deterministic, not an address).  Useful for
 * external hash tables; not guaranteed stable across library versions. */
uint32_t ixs_node_hash(ixs_node *node);

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

/* --- Generic child access ----------------------------------------------- */

/* Number of child node pointers.  Leaves return 0. */
uint32_t ixs_node_nchildren(ixs_node *node);

/* i-th child node.  i must be < ixs_node_nchildren(node).
 * Child order matches the type-specific accessors:
 *   ADD: coeff, (term_coeff[0], term[0]), (term_coeff[1], term[1]), ...
 *   MUL: coeff, base[0], base[1], ...   (exponents are int32_t, not nodes)
 *   binary: lhs, rhs
 *   unary: arg
 *   PW:  (value[0], cond[0]), (value[1], cond[1]), ...
 *   AND/OR: arg[0], arg[1], ... */
ixs_node *ixs_node_child(ixs_node *node, uint32_t i);

/* --- Rule-hit statistics (requires -DIXS_STATS at compile time) -------- */

/* Number of distinct rules that have fired.  Returns 0 if compiled
 * without IXS_STATS. */
size_t ixs_ctx_nstats(ixs_ctx *ctx);

/* Retrieve the i-th stat entry (arbitrary order, 0-based).
 * Sets *name to the rule name and returns the hit count.
 * Returns 0 with *name = NULL for out-of-range indices. */
uint64_t ixs_ctx_stat(ixs_ctx *ctx, size_t index, const char **name);

/* Reset all counters to zero. */
void ixs_ctx_stats_reset(ixs_ctx *ctx);

/* Total number of distinct rule/transform names registered in the
 * simplifier (rule tables + ad-hoc transforms).  Context-independent. */
size_t ixs_nrules(void);

/* The i-th rule name (0-based).  Returns NULL if out of range. */
const char *ixs_rule_name(size_t index);

/* --- Tree walk --------------------------------------------------------- */

typedef enum {
  IXS_WALK_CONTINUE,
  IXS_WALK_SKIP,
  IXS_WALK_STOP
} ixs_walk_action;

/* Callback must return exactly one of the three values above.
 * Any other return value is undefined behavior. */
typedef ixs_walk_action (*ixs_visit_fn)(ixs_node *node, void *userdata);

/* Pre-order: visit node, then recurse into children.
 * Returns root on completion, the stopping node on STOP, NULL if root
 * is NULL.  A future iterative implementation may also return NULL on
 * OOM (scratch arena exhausted); ctx is accepted for that purpose.
 * Sentinels (ERROR, PARSE_ERROR) are visited as leaves; the callback
 * must check ixs_node_tag before using type-specific accessors.
 * SKIP prevents descent into children. */
ixs_node *ixs_walk_pre(ixs_ctx *ctx, ixs_node *root, ixs_visit_fn fn,
                       void *userdata);

/* Post-order: recurse into children, then visit node.
 * Same return/NULL/sentinel semantics as ixs_walk_pre.
 * SKIP is a no-op in post-order (children already visited). */
ixs_node *ixs_walk_post(ixs_ctx *ctx, ixs_node *root, ixs_visit_fn fn,
                        void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* IXSIMPL_H */
