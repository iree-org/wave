/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_RATIONAL_H
#define IXS_RATIONAL_H

#include "internal.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Overflow-safe rational arithmetic on int64_t p/q pairs.
 * All results are in lowest terms with q > 0.
 * Functions returning bool: true = success, false = overflow.
 * Division/mod by zero → false.
 */

/* Binary GCD, handles INT64_MIN. Inputs may be negative. */
IXS_STATIC int64_t ixs_gcd(int64_t a, int64_t b);

/* Normalize p/q to lowest terms, q > 0. false on bad input or overflow. */
IXS_STATIC bool ixs_rat_normalize(int64_t p, int64_t q, int64_t *rp,
                                  int64_t *rq);

IXS_STATIC bool ixs_rat_add(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq);
IXS_STATIC bool ixs_rat_sub(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq);
IXS_STATIC bool ixs_rat_mul(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq);
IXS_STATIC bool ixs_rat_div(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq);
IXS_STATIC bool ixs_rat_neg(int64_t p, int64_t q, int64_t *rp, int64_t *rq);

/* Floored division: floor(p/q). q must be > 0. */
IXS_STATIC int64_t ixs_rat_floor(int64_t p, int64_t q);

/* Ceiling: ceil(p/q). q must be > 0. */
IXS_STATIC int64_t ixs_rat_ceil(int64_t p, int64_t q);

/* Floored mod: a mod b = a - b*floor(a/b). b > 0. */
IXS_STATIC bool ixs_rat_mod(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq);

/* Compare: returns -1, 0, +1. Overflow-safe (cross-multiply). */
IXS_STATIC int ixs_rat_cmp(int64_t ap, int64_t aq, int64_t bp, int64_t bq);

static inline bool ixs_rat_is_zero(int64_t p) { return p == 0; }
static inline bool ixs_rat_is_one(int64_t p, int64_t q) {
  return p == 1 && q == 1;
}
static inline bool ixs_rat_is_neg(int64_t p) { return p < 0; }
static inline bool ixs_rat_is_int(int64_t q) { return q == 1; }

/* Overflow-checked int64 arithmetic. */
IXS_STATIC bool ixs_safe_add(int64_t a, int64_t b, int64_t *r);
IXS_STATIC bool ixs_safe_sub(int64_t a, int64_t b, int64_t *r);
IXS_STATIC bool ixs_safe_mul(int64_t a, int64_t b, int64_t *r);
IXS_STATIC bool ixs_safe_neg(int64_t a, int64_t *r);

#endif /* IXS_RATIONAL_H */
