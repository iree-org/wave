/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_INTERVAL_H
#define IXS_INTERVAL_H

#include "internal.h"

#include "rational.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  int64_t lo_p, lo_q; /* lower bound (rational), inclusive */
  int64_t hi_p, hi_q; /* upper bound (rational), inclusive */
  bool valid;
} ixs_interval;

#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807LL - 1)
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#endif

static inline bool ixs_interval_is_neg_inf(int64_t p, int64_t q) {
  return p == INT64_MIN && q == 1;
}

static inline bool ixs_interval_is_pos_inf(int64_t p, int64_t q) {
  return p == INT64_MAX && q == 1;
}

static inline void ixs_interval_set_neg_inf(int64_t *p, int64_t *q) {
  *p = INT64_MIN;
  *q = 1;
}

static inline void ixs_interval_set_pos_inf(int64_t *p, int64_t *q) {
  *p = INT64_MAX;
  *q = 1;
}

static inline ixs_interval ixs_interval_unknown(void) {
  ixs_interval iv;
  iv.valid = false;
  iv.lo_p = iv.lo_q = iv.hi_p = iv.hi_q = 0;
  return iv;
}

static inline ixs_interval ixs_interval_exact(int64_t p, int64_t q) {
  ixs_interval iv;
  iv.valid = true;
  iv.lo_p = iv.hi_p = p;
  iv.lo_q = iv.hi_q = q;
  return iv;
}

/* True when iv is a single integer value; writes it to *val if non-NULL. */
static inline bool ixs_interval_is_point_int(ixs_interval iv, int64_t *val) {
  if (iv.valid && iv.lo_q == 1 && iv.hi_q == 1 && iv.lo_p == iv.hi_p) {
    if (val)
      *val = iv.lo_p;
    return true;
  }
  return false;
}

static inline ixs_interval ixs_interval_range(int64_t lo_p, int64_t lo_q,
                                              int64_t hi_p, int64_t hi_q) {
  ixs_interval iv;
  iv.valid = true;
  iv.lo_p = lo_p;
  iv.lo_q = lo_q;
  iv.hi_p = hi_p;
  iv.hi_q = hi_q;
  return iv;
}

/* Widen one endpoint to +/-infinity based on the sign of the product. */
IXS_STATIC void iv_endpoint_widen(int64_t ap, int64_t bp, int64_t *rp,
                                  int64_t *rq);

IXS_STATIC ixs_interval iv_add(ixs_interval a, ixs_interval b);
IXS_STATIC ixs_interval iv_mul_const(ixs_interval a, int64_t cp, int64_t cq);
IXS_STATIC ixs_interval iv_mul(ixs_interval a, ixs_interval b);

/* Reciprocal of a strictly positive interval. Returns unknown if
 * the interval contains zero or is invalid. */
IXS_STATIC ixs_interval iv_recip(ixs_interval a);

IXS_STATIC ixs_interval iv_intersect(ixs_interval a, ixs_interval b);

#endif /* IXS_INTERVAL_H */
