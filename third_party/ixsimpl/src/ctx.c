/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "expand.h"
#include "node.h"
#include "parser.h"
#include "print.h"
#include "simplify.h"
#include <stdlib.h>
#include <string.h>

static ixs_node *ctx_err(ixs_ctx *ctx, const char *msg) {
  ixs_ctx_push_error(ctx, "%s", msg);
  return ctx->sentinel_error;
}

/* ------------------------------------------------------------------ */
/*  Singleton creation                                                */
/* ------------------------------------------------------------------ */

static ixs_node *make_singleton(ixs_ctx *ctx, ixs_tag tag, uint32_t seed) {
  ixs_node *n = ixs_arena_alloc(&ctx->arena, sizeof(ixs_node), sizeof(void *));
  if (!n)
    return NULL;
  memset(n, 0, sizeof(*n));
  n->tag = tag;
  n->hash = (uint32_t)tag * 2654435761u;
  n->hash ^= seed;
  n->hash *= 0x9e3779b9u;
  /* Insert into hash table. */
  return ixs_htab_intern(ctx, n);
}

/* ------------------------------------------------------------------ */
/*  Context lifecycle                                                 */
/* ------------------------------------------------------------------ */

ixs_ctx *ixs_ctx_create(void) {
  ixs_ctx tmp;
  ixs_ctx *ctx;
  memset(&tmp, 0, sizeof(tmp));

  ixs_arena_init(&tmp.arena, IXS_ARENA_DEFAULT_SIZE);
  ixs_arena_init(&tmp.scratch, IXS_ARENA_DEFAULT_SIZE);

  if (!ixs_htab_init(&tmp))
    return NULL;

  /* Create singletons. */
  tmp.sentinel_error = make_singleton(&tmp, IXS_ERROR, 0xDEAD);
  tmp.sentinel_parse_error = make_singleton(&tmp, IXS_PARSE_ERROR, 0xBEEF);
  tmp.node_true = make_singleton(&tmp, IXS_TRUE, 1);
  tmp.node_false = make_singleton(&tmp, IXS_FALSE, 0);

  if (!tmp.sentinel_error || !tmp.sentinel_parse_error || !tmp.node_true ||
      !tmp.node_false)
    goto fail;

  tmp.node_zero = ixs_node_int(&tmp, 0);
  tmp.node_one = ixs_node_int(&tmp, 1);

  if (!tmp.node_zero || !tmp.node_one)
    goto fail;

#ifdef IXS_STATS
  tmp.stats = ixs_arena_alloc(
      &tmp.arena, IXS_STATS_CAP * sizeof(ixs_stat_entry), sizeof(void *));
  if (!tmp.stats)
    goto fail;
  memset(tmp.stats, 0, IXS_STATS_CAP * sizeof(ixs_stat_entry));
#endif

  /* Emplace ctx into its own arena — one fewer heap allocation. */
  ctx = ixs_arena_alloc(&tmp.arena, sizeof(ixs_ctx), sizeof(void *));
  if (!ctx)
    goto fail;
  memcpy(ctx, &tmp, sizeof(ixs_ctx));
  return ctx;

fail:
  ixs_htab_destroy(&tmp);
  ixs_arena_destroy(&tmp.scratch);
  ixs_arena_destroy(&tmp.arena);
  return NULL;
}

void ixs_ctx_destroy(ixs_ctx *ctx) {
  if (!ctx)
    return;
  ixs_htab_destroy(ctx);
  ixs_arena_destroy(&ctx->scratch);
  /* ctx itself lives inside the main arena; snapshot before freeing. */
  ixs_arena arena = ctx->arena;
  ixs_arena_destroy(&arena);
}

/* ------------------------------------------------------------------ */
/*  Error list                                                        */
/* ------------------------------------------------------------------ */

size_t ixs_ctx_nerrors(ixs_ctx *ctx) { return ctx->nerrors; }

const char *ixs_ctx_error(ixs_ctx *ctx, size_t index) {
  if (index >= ctx->nerrors)
    return NULL;
  return ctx->errors[index];
}

void ixs_ctx_clear_errors(ixs_ctx *ctx) { ctx->nerrors = 0; }

/* ------------------------------------------------------------------ */
/*  Rule-hit statistics                                                */
/* ------------------------------------------------------------------ */

size_t ixs_ctx_nstats(ixs_ctx *ctx) {
#ifdef IXS_STATS
  size_t n = 0;
  size_t i;
  for (i = 0; i < IXS_STATS_CAP; i++) {
    if (ctx->stats[i].name)
      n++;
  }
  return n;
#else
  (void)ctx;
  return 0;
#endif
}

uint64_t ixs_ctx_stat(ixs_ctx *ctx, size_t index, const char **name) {
#ifdef IXS_STATS
  size_t seen = 0;
  size_t i;
  for (i = 0; i < IXS_STATS_CAP; i++) {
    if (ctx->stats[i].name) {
      if (seen == index) {
        if (name)
          *name = ctx->stats[i].name;
        return ctx->stats[i].count;
      }
      seen++;
    }
  }
#else
  (void)ctx;
  (void)index;
#endif
  if (name)
    *name = NULL;
  return 0;
}

void ixs_ctx_stats_reset(ixs_ctx *ctx) {
#ifdef IXS_STATS
  memset(ctx->stats, 0, IXS_STATS_CAP * sizeof(ixs_stat_entry));
#else
  (void)ctx;
#endif
}

/* ------------------------------------------------------------------ */
/*  Sentinel checks                                                   */
/* ------------------------------------------------------------------ */

bool ixs_is_error(ixs_node *node) {
  return node && (node->tag == IXS_ERROR || node->tag == IXS_PARSE_ERROR);
}

bool ixs_is_parse_error(ixs_node *node) {
  return node && node->tag == IXS_PARSE_ERROR;
}

bool ixs_is_domain_error(ixs_node *node) {
  return node && node->tag == IXS_ERROR;
}

/* ------------------------------------------------------------------ */
/*  Constructors (delegate to simplify.c)                             */
/* ------------------------------------------------------------------ */

ixs_node *ixs_int(ixs_ctx *ctx, int64_t val) { return ixs_node_int(ctx, val); }

ixs_node *ixs_rat(ixs_ctx *ctx, int64_t p, int64_t q) {
  if (q == 0)
    return ctx_err(ctx, "ixs_rat: denominator is zero");
  int64_t rp, rq;
  if (!ixs_rat_normalize(p, q, &rp, &rq))
    return ctx_err(ctx, "ixs_rat: rational overflow");
  return ixs_node_rat(ctx, rp, rq);
}

ixs_node *ixs_sym(ixs_ctx *ctx, const char *name) {
  return ixs_node_sym(ctx, name, strlen(name));
}

ixs_node *ixs_add(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_add(ctx, a, b);
}

ixs_node *ixs_mul(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_mul(ctx, a, b);
}

ixs_node *ixs_neg(ixs_ctx *ctx, ixs_node *a) { return simp_neg(ctx, a); }

ixs_node *ixs_sub(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_sub(ctx, a, b);
}

ixs_node *ixs_div(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_div(ctx, a, b);
}

ixs_node *ixs_floor(ixs_ctx *ctx, ixs_node *x) { return simp_floor(ctx, x); }

ixs_node *ixs_ceil(ixs_ctx *ctx, ixs_node *x) { return simp_ceil(ctx, x); }

ixs_node *ixs_mod(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_mod(ctx, a, b);
}

ixs_node *ixs_max(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_max(ctx, a, b);
}

ixs_node *ixs_min(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_min(ctx, a, b);
}

ixs_node *ixs_xor(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_xor(ctx, a, b);
}

ixs_node *ixs_pw(ixs_ctx *ctx, uint32_t n, ixs_node **values,
                 ixs_node **conds) {
  return simp_pw(ctx, n, values, conds);
}

ixs_node *ixs_cmp(ixs_ctx *ctx, ixs_node *a, ixs_cmp_op op, ixs_node *b) {
  return simp_cmp(ctx, a, op, b);
}

ixs_node *ixs_and(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_and(ctx, a, b);
}

ixs_node *ixs_or(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  return simp_or(ctx, a, b);
}

ixs_node *ixs_not(ixs_ctx *ctx, ixs_node *a) { return simp_not(ctx, a); }

ixs_node *ixs_true(ixs_ctx *ctx) { return ctx->node_true; }

ixs_node *ixs_false(ixs_ctx *ctx) { return ctx->node_false; }

/* ------------------------------------------------------------------ */
/*  Parse                                                             */
/* ------------------------------------------------------------------ */

ixs_node *ixs_parse(ixs_ctx *ctx, const char *input, size_t len) {
  if (!input)
    return ctx->sentinel_parse_error;
  return ixs_parse_impl(ctx, input, len);
}

/* ------------------------------------------------------------------ */
/*  Simplification                                                    */
/* ------------------------------------------------------------------ */

ixs_node *ixs_expand(ixs_ctx *ctx, ixs_node *expr) {
  return expand_impl(ctx, expr);
}

ixs_node *ixs_simplify(ixs_ctx *ctx, ixs_node *expr,
                       ixs_node *const *assumptions, size_t n_assumptions) {
  return simp_simplify(ctx, expr, assumptions, n_assumptions);
}

void ixs_simplify_batch(ixs_ctx *ctx, ixs_node **exprs, size_t n,
                        ixs_node *const *assumptions, size_t n_assumptions) {
  simp_simplify_batch(ctx, exprs, n, assumptions, n_assumptions);
}

/* ------------------------------------------------------------------ */
/*  Entailment checking                                               */
/* ------------------------------------------------------------------ */

ixs_check_result ixs_check(ixs_ctx *ctx, ixs_node *expr,
                           ixs_node *const *assumptions, size_t n_assumptions) {
  return simp_check(ctx, expr, assumptions, n_assumptions);
}

/* ------------------------------------------------------------------ */
/*  Comparison and substitution                                       */
/* ------------------------------------------------------------------ */

bool ixs_same_node(ixs_node *a, ixs_node *b) { return a == b; }

ixs_node *ixs_subs(ixs_ctx *ctx, ixs_node *expr, ixs_node *target,
                   ixs_node *replacement) {
  return simp_subs(ctx, expr, target, replacement);
}

ixs_node *ixs_subs_multi(ixs_ctx *ctx, ixs_node *expr, uint32_t nsubs,
                         ixs_node *const *targets,
                         ixs_node *const *replacements) {
  return simp_subs_multi(ctx, expr, nsubs, targets, replacements);
}

/* ------------------------------------------------------------------ */
/*  Output                                                            */
/* ------------------------------------------------------------------ */

size_t ixs_print(ixs_node *expr, char *buf, size_t bufsize) {
  return ixs_print_impl(expr, buf, bufsize);
}

size_t ixs_print_c(ixs_node *expr, char *buf, size_t bufsize) {
  return ixs_print_c_impl(expr, buf, bufsize);
}

/* ------------------------------------------------------------------ */
/*  Introspection                                                     */
/* ------------------------------------------------------------------ */

ixs_tag ixs_node_tag(ixs_node *node) { return node->tag; }

int64_t ixs_node_int_val(ixs_node *node) { return node->u.ival; }

uint32_t ixs_node_hash(ixs_node *node) { return node->hash; }
