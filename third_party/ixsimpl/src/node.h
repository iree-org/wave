/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_NODE_H
#define IXS_NODE_H

#include "internal.h"

#include "arena.h"
#include "rational.h"
#include <ixsimpl.h>

/* --- Internal node struct --- */

typedef struct ixs_addterm {
  struct ixs_node *term;
  struct ixs_node *coeff; /* IXS_INT or IXS_RAT, nonzero */
} ixs_addterm;

typedef struct ixs_mulfactor {
  struct ixs_node *base;
  int32_t exp; /* nonzero */
} ixs_mulfactor;

typedef struct ixs_pwcase {
  struct ixs_node *value;
  struct ixs_node *cond;
} ixs_pwcase;

struct ixs_node {
  ixs_tag tag;
  uint32_t hash;
  union ixs_node_data {
    int64_t ival; /* IXS_INT */
    struct {
      int64_t p, q;
    } rat;            /* IXS_RAT */
    const char *name; /* IXS_SYM */
    struct {          /* IXS_ADD */
      struct ixs_node *coeff;
      uint32_t nterms;
      ixs_addterm *terms;
    } add;
    struct { /* IXS_MUL */
      struct ixs_node *coeff;
      uint32_t nfactors;
      ixs_mulfactor *factors;
    } mul;
    struct { /* IXS_FLOOR, IXS_CEIL */
      struct ixs_node *arg;
    } unary;
    struct { /* IXS_MOD, IXS_MAX, IXS_MIN, IXS_XOR, IXS_CMP */
      struct ixs_node *lhs;
      struct ixs_node *rhs;
      ixs_cmp_op cmp_op;
    } binary;
    struct { /* IXS_PIECEWISE */
      uint32_t ncases;
      ixs_pwcase *cases;
    } pw;
    struct { /* IXS_AND, IXS_OR */
      uint32_t nargs;
      struct ixs_node **args;
    } logic;
    struct { /* IXS_NOT */
      struct ixs_node *arg;
    } unary_bool;
  } u;
};

/* --- Rule-hit statistics (compile with -DIXS_STATS to enable) --- */

#ifdef IXS_STATS
#define IXS_STATS_CAP 128
typedef struct {
  const char *name; /* rule name pointer from __func__; NULL = empty slot */
  uint64_t count;
} ixs_stat_entry;

static inline void ixs_stat_hit(ixs_stat_entry *stats, const char *fn) {
  size_t mask = IXS_STATS_CAP - 1;
  size_t idx = ((uintptr_t)fn >> 3) & mask;
  size_t probes;
  for (probes = 0; probes < IXS_STATS_CAP; probes++) {
    if (!stats[idx].name) {
      stats[idx].name = fn;
      stats[idx].count = 1;
      return;
    }
    if (stats[idx].name == fn) {
      stats[idx].count++;
      return;
    }
    idx = (idx + 1) & mask;
  }
}
#define IXS_STAT_HIT(ctx) ixs_stat_hit((ctx)->stats, __func__)
#else
#define IXS_STAT_HIT(ctx) ((void)(ctx))
#endif

/* --- Internal context --- */

struct ixs_ctx {
  ixs_arena arena;
  ixs_arena scratch;

  /* Hash-consing table (open addressing, linear probing) */
  ixs_node **htab;
  size_t htab_cap;
  size_t htab_used;

  /* Error list (arena-managed pointer array, strings also arena) */
  const char **errors;
  size_t nerrors;
  size_t errors_cap;

  /* Singletons */
  ixs_node *sentinel_error;
  ixs_node *sentinel_parse_error;
  ixs_node *node_true;
  ixs_node *node_false;
  ixs_node *node_zero;
  ixs_node *node_one;

#ifdef IXS_STATS
  ixs_stat_entry *stats;
#endif
};

/* --- Hash-consing --- */

#define IXS_HTAB_INIT_CAP 1024
#define IXS_HTAB_LOAD_NUM 7
#define IXS_HTAB_LOAD_DEN 10

/* Initialize/destroy the hash table (malloc-managed). */
IXS_STATIC bool ixs_htab_init(ixs_ctx *ctx);
IXS_STATIC void ixs_htab_destroy(ixs_ctx *ctx);

/*
 * Intern a node: if an equal node already exists, return it;
 * otherwise insert this one. The node must already be arena-allocated
 * with hash computed. Returns NULL on OOM (rehash failure).
 */
IXS_STATIC ixs_node *ixs_htab_intern(ixs_ctx *ctx, ixs_node *node);

/* --- Raw node constructors (no simplification) --- */

IXS_STATIC ixs_node *ixs_node_int(ixs_ctx *ctx, int64_t val);
IXS_STATIC ixs_node *ixs_node_rat(ixs_ctx *ctx, int64_t p, int64_t q);
IXS_STATIC ixs_node *ixs_node_sym(ixs_ctx *ctx, const char *name, size_t len);
IXS_STATIC ixs_node *ixs_node_add(ixs_ctx *ctx, ixs_node *coeff,
                                  uint32_t nterms, ixs_addterm *terms);
IXS_STATIC ixs_node *ixs_node_mul(ixs_ctx *ctx, ixs_node *coeff,
                                  uint32_t nfactors, ixs_mulfactor *factors);
IXS_STATIC ixs_node *ixs_node_floor(ixs_ctx *ctx, ixs_node *arg);
IXS_STATIC ixs_node *ixs_node_ceil(ixs_ctx *ctx, ixs_node *arg);
IXS_STATIC ixs_node *ixs_node_binary(ixs_ctx *ctx, ixs_tag tag, ixs_node *lhs,
                                     ixs_node *rhs, ixs_cmp_op op);
IXS_STATIC ixs_node *ixs_node_pw(ixs_ctx *ctx, uint32_t ncases,
                                 ixs_pwcase *cases);
IXS_STATIC ixs_node *ixs_node_logic(ixs_ctx *ctx, ixs_tag tag, uint32_t nargs,
                                    ixs_node **args);
IXS_STATIC ixs_node *ixs_node_not(ixs_ctx *ctx, ixs_node *arg);

/* --- Node comparison (total order for canonical sorting) --- */

IXS_STATIC int ixs_node_cmp(const ixs_node *a, const ixs_node *b);
IXS_STATIC bool ixs_node_equal(const ixs_node *a, const ixs_node *b);

/* --- Utility --- */

IXS_STATIC bool ixs_node_is_const(const ixs_node *n);
IXS_STATIC bool ixs_node_is_zero(const ixs_node *n);
IXS_STATIC bool ixs_node_is_one(const ixs_node *n);
IXS_STATIC void ixs_node_get_rat(const ixs_node *n, int64_t *p, int64_t *q);
IXS_STATIC bool ixs_node_is_sentinel(const ixs_node *n);

/* True if the node is guaranteed to produce an integer for all
 * variable assignments.  Conservative: may return false for some
 * integer-valued expressions. */
IXS_STATIC bool ixs_node_is_integer_valued(const ixs_node *n);

/* Error list helper. fmt is printf-style. */
IXS_STATIC void ixs_ctx_push_error(ixs_ctx *ctx, const char *fmt, ...);

/* NULL/sentinel propagation for binary ops. Returns the propagated
 * sentinel if either arg is NULL/sentinel, or NULL if both are clean. */
IXS_STATIC ixs_node *ixs_propagate2(ixs_node *a, ixs_node *b);

/* Same for unary. */
IXS_STATIC ixs_node *ixs_propagate1(ixs_node *a);

#endif /* IXS_NODE_H */
