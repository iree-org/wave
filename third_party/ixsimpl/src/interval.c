/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "interval.h"

IXS_STATIC void iv_endpoint_widen(int64_t ap, int64_t bp, int64_t *rp,
                                  int64_t *rq) {
  bool neg = (ap < 0) != ixs_rat_is_neg(bp);
  if (neg)
    ixs_interval_set_neg_inf(rp, rq);
  else
    ixs_interval_set_pos_inf(rp, rq);
}

IXS_STATIC ixs_interval iv_add(ixs_interval a, ixs_interval b) {
  ixs_interval r;
  if (!a.valid || !b.valid)
    return ixs_interval_unknown();
  r.valid = true;
  if (!ixs_rat_add(a.lo_p, a.lo_q, b.lo_p, b.lo_q, &r.lo_p, &r.lo_q)) {
    if (ixs_rat_is_neg(a.lo_p) || ixs_rat_is_neg(b.lo_p))
      ixs_interval_set_neg_inf(&r.lo_p, &r.lo_q);
    else
      ixs_interval_set_pos_inf(&r.lo_p, &r.lo_q);
  }
  if (!ixs_rat_add(a.hi_p, a.hi_q, b.hi_p, b.hi_q, &r.hi_p, &r.hi_q)) {
    if (!ixs_rat_is_neg(a.hi_p) || !ixs_rat_is_neg(b.hi_p))
      ixs_interval_set_pos_inf(&r.hi_p, &r.hi_q);
    else
      ixs_interval_set_neg_inf(&r.hi_p, &r.hi_q);
  }
  return r;
}

IXS_STATIC ixs_interval iv_mul_const(ixs_interval a, int64_t cp, int64_t cq) {
  ixs_interval r;
  if (!a.valid)
    return ixs_interval_unknown();
  if (cp == 0)
    return ixs_interval_exact(0, 1);
  r.valid = true;
  if (!ixs_rat_mul(a.lo_p, a.lo_q, cp, cq, &r.lo_p, &r.lo_q))
    iv_endpoint_widen(a.lo_p, cp, &r.lo_p, &r.lo_q);
  if (!ixs_rat_mul(a.hi_p, a.hi_q, cp, cq, &r.hi_p, &r.hi_q))
    iv_endpoint_widen(a.hi_p, cp, &r.hi_p, &r.hi_q);
  if (ixs_rat_is_neg(cp)) {
    int64_t tmp_p = r.lo_p, tmp_q = r.lo_q;
    r.lo_p = r.hi_p;
    r.lo_q = r.hi_q;
    r.hi_p = tmp_p;
    r.hi_q = tmp_q;
  }
  return r;
}

IXS_STATIC ixs_interval iv_mul(ixs_interval a, ixs_interval b) {
  ixs_interval r;
  int64_t ap[4], aq[4], bp[4], bq[4], rp[4], rq[4];
  uint32_t i;
  if (!a.valid || !b.valid)
    return ixs_interval_unknown();
  ap[0] = a.lo_p;
  aq[0] = a.lo_q;
  bp[0] = b.lo_p;
  bq[0] = b.lo_q;
  ap[1] = a.lo_p;
  aq[1] = a.lo_q;
  bp[1] = b.hi_p;
  bq[1] = b.hi_q;
  ap[2] = a.hi_p;
  aq[2] = a.hi_q;
  bp[2] = b.lo_p;
  bq[2] = b.lo_q;
  ap[3] = a.hi_p;
  aq[3] = a.hi_q;
  bp[3] = b.hi_p;
  bq[3] = b.hi_q;
  for (i = 0; i < 4; i++) {
    if (!ixs_rat_mul(ap[i], aq[i], bp[i], bq[i], &rp[i], &rq[i]))
      iv_endpoint_widen(ap[i], bp[i], &rp[i], &rq[i]);
  }
  r.valid = true;
  r.lo_p = rp[0];
  r.lo_q = rq[0];
  r.hi_p = rp[0];
  r.hi_q = rq[0];
  for (i = 1; i < 4; i++) {
    if (ixs_rat_cmp(rp[i], rq[i], r.lo_p, r.lo_q) < 0) {
      r.lo_p = rp[i];
      r.lo_q = rq[i];
    }
    if (ixs_rat_cmp(rp[i], rq[i], r.hi_p, r.hi_q) > 0) {
      r.hi_p = rp[i];
      r.hi_q = rq[i];
    }
  }
  return r;
}

IXS_STATIC ixs_interval iv_recip(ixs_interval a) {
  ixs_interval r;
  if (!a.valid || ixs_rat_cmp(a.lo_p, a.lo_q, 0, 1) <= 0)
    return ixs_interval_unknown();
  r.valid = true;
  if (ixs_interval_is_pos_inf(a.hi_p, a.hi_q)) {
    r.lo_p = 0;
    r.lo_q = 1;
  } else {
    r.lo_p = a.hi_q;
    r.lo_q = a.hi_p;
  }
  r.hi_p = a.lo_q;
  r.hi_q = a.lo_p;
  return r;
}

IXS_STATIC ixs_interval iv_intersect(ixs_interval a, ixs_interval b) {
  ixs_interval r;
  if (!a.valid)
    return b;
  if (!b.valid)
    return a;
  r.valid = true;
  if (ixs_rat_cmp(a.lo_p, a.lo_q, b.lo_p, b.lo_q) >= 0) {
    r.lo_p = a.lo_p;
    r.lo_q = a.lo_q;
  } else {
    r.lo_p = b.lo_p;
    r.lo_q = b.lo_q;
  }
  if (ixs_rat_cmp(a.hi_p, a.hi_q, b.hi_p, b.hi_q) <= 0) {
    r.hi_p = a.hi_p;
    r.hi_q = a.hi_q;
  } else {
    r.hi_p = b.hi_p;
    r.hi_q = b.hi_q;
  }
  if (ixs_rat_cmp(r.lo_p, r.lo_q, r.hi_p, r.hi_q) > 0)
    r.valid = false;
  return r;
}
