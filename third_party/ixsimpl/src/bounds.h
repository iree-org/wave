/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_BOUNDS_H
#define IXS_BOUNDS_H

#include "internal.h"

#include "interval.h"
#include "node.h"
#include <ixsimpl.h>

/*
 * Lightweight interval analysis for bound-dependent rewrites.
 * Stores per-variable intervals extracted from comparison assumptions,
 * and propagates through expression structure.
 *
 * The var table is a growable array on the scratch arena.  Lookup is
 * linear by pointer equality (symbol names are interned).  If this
 * ever becomes a bottleneck, swap the array for an open-addressing
 * hash map keyed on the interned name pointer -- the interface is
 * already designed to make that a drop-in replacement.
 */

typedef struct {
  const char *name; /* interned pointer -- identity compare only */
  ixs_interval iv;
  int64_t modulus;   /* 0 = no info, >0 = sym ≡ remainder (mod modulus) */
  int64_t remainder; /* in [0, modulus) when modulus > 0                */
} ixs_var_bound;

typedef struct {
  ixs_node *expr; /* expression pointer -- identity compare only */
  ixs_interval iv;
} ixs_expr_bound;

typedef struct {
  ixs_var_bound *vars; /* arena-allocated growable array */
  size_t nvars;
  size_t cap;
  ixs_expr_bound *exprs; /* per-expression overrides from branch conditions */
  size_t nexprs;
  size_t expr_cap;
  ixs_arena *scratch; /* borrowed; must outlive ixs_bounds */
} ixs_bounds;

/* Returns false on OOM (arena exhausted). */
IXS_STATIC bool ixs_bounds_init(ixs_bounds *b, ixs_arena *scratch);

/* No-op; bounds memory is reclaimed by scratch arena restore. */
IXS_STATIC void ixs_bounds_destroy(ixs_bounds *b);

/* Deep-copy bounds onto the scratch arena. Returns false on OOM. */
IXS_STATIC bool ixs_bounds_fork(ixs_bounds *dst, const ixs_bounds *src);

/* Extract variable bounds from an assumption (e.g., $T0 >= 0). */
IXS_STATIC void ixs_bounds_add_assumption(ixs_bounds *b, ixs_node *assumption);

/* Store an explicit interval for an arbitrary expression node. */
IXS_STATIC void ixs_bounds_add_expr(ixs_bounds *b, ixs_node *expr,
                                    ixs_interval iv);

/* Get the interval for an expression using propagation rules. */
IXS_STATIC ixs_interval ixs_bounds_get(ixs_bounds *b, ixs_node *expr);

/* Full modulus/remainder query.  Returns true when info is available.
 * On success *mod > 0 and 0 <= *rem < *mod. */
IXS_STATIC bool ixs_bounds_get_modrem(ixs_bounds *b, const char *name,
                                      int64_t *mod, int64_t *rem);

/* Initialize bounds from an array of assumption nodes.
 * Skips NULL and sentinel assumptions.  Returns false on OOM. */
IXS_STATIC bool ixs_bounds_build(ixs_bounds *b, ixs_arena *scratch,
                                 ixs_node *const *assumptions,
                                 size_t n_assumptions);

/* Check a normalized CMP node (lhs op 0) against current bounds.
 * Returns IXS_CHECK_TRUE / FALSE / UNKNOWN.  Non-CMP input or
 * non-zero rhs returns UNKNOWN. */
IXS_STATIC ixs_check_result ixs_bounds_check(ixs_bounds *b, ixs_node *cmp);

#endif /* IXS_BOUNDS_H */
