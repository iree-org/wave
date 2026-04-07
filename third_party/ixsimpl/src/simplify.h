/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_SIMPLIFY_H
#define IXS_SIMPLIFY_H

#include "internal.h"

#include "node.h"

/*
 * Smart constructors: apply canonicalization rules, then hash-cons.
 * Called by the public API in ctx.c and by the parser.
 */

IXS_STATIC ixs_node *simp_add(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_mul(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_neg(ixs_ctx *ctx, ixs_node *a);
IXS_STATIC ixs_node *simp_sub(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_div(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_floor(ixs_ctx *ctx, ixs_node *x);
IXS_STATIC ixs_node *simp_ceil(ixs_ctx *ctx, ixs_node *x);
IXS_STATIC ixs_node *simp_mod(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_max(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_min(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_xor(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_pw(ixs_ctx *ctx, uint32_t n, ixs_node **values,
                             ixs_node **conds);
IXS_STATIC ixs_node *simp_cmp(ixs_ctx *ctx, ixs_node *a, ixs_cmp_op op,
                              ixs_node *b);
IXS_STATIC ixs_node *simp_and(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_or(ixs_ctx *ctx, ixs_node *a, ixs_node *b);
IXS_STATIC ixs_node *simp_not(ixs_ctx *ctx, ixs_node *a);

/* Top-down simplification pass with assumptions + bound analysis. */
IXS_STATIC ixs_node *simp_simplify(ixs_ctx *ctx, ixs_node *expr,
                                   ixs_node *const *assumptions,
                                   size_t n_assumptions);

/* Batch: simplify exprs[0..n-1] in place, building bounds once. */
IXS_STATIC void simp_simplify_batch(ixs_ctx *ctx, ixs_node **exprs, size_t n,
                                    ixs_node *const *assumptions,
                                    size_t n_assumptions);

/* Entailment check: bounds-only, no rewriting. */
IXS_STATIC ixs_check_result simp_check(ixs_ctx *ctx, ixs_node *expr,
                                       ixs_node *const *assumptions,
                                       size_t n_assumptions);

/* Substitution: replace all occurrences of target with replacement. */
IXS_STATIC ixs_node *simp_subs(ixs_ctx *ctx, ixs_node *expr, ixs_node *target,
                               ixs_node *replacement);

/* Simultaneous multi-target substitution. */
IXS_STATIC ixs_node *simp_subs_multi(ixs_ctx *ctx, ixs_node *expr,
                                     uint32_t nsubs, ixs_node *const *targets,
                                     ixs_node *const *replacements);

#endif /* IXS_SIMPLIFY_H */
