/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "rational.h"
#include <limits.h>

#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807LL - 1)
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#endif

/* --- Safe arithmetic --- */

IXS_STATIC bool ixs_safe_add(int64_t a, int64_t b, int64_t *r) {
  if (b > 0 && a > INT64_MAX - b)
    return false;
  if (b < 0 && a < INT64_MIN - b)
    return false;
  *r = a + b;
  return true;
}

IXS_STATIC bool ixs_safe_sub(int64_t a, int64_t b, int64_t *r) {
  if (b < 0 && a > INT64_MAX + b)
    return false;
  if (b > 0 && a < INT64_MIN + b)
    return false;
  *r = a - b;
  return true;
}

IXS_STATIC bool ixs_safe_mul(int64_t a, int64_t b, int64_t *r) {
  if (a == 0 || b == 0) {
    *r = 0;
    return true;
  }
  if (a == 1) {
    *r = b;
    return true;
  }
  if (b == 1) {
    *r = a;
    return true;
  }
  if (a == -1)
    return ixs_safe_neg(b, r);
  if (b == -1)
    return ixs_safe_neg(a, r);

  /* Both non-zero, neither +-1. Check overflow. */
  if (a > 0) {
    if (b > 0) {
      if (a > INT64_MAX / b)
        return false;
    } else {
      if (b < INT64_MIN / a)
        return false;
    }
  } else {
    if (b > 0) {
      if (a < INT64_MIN / b)
        return false;
    } else {
      if (a < INT64_MAX / b)
        return false; /* (-a)*(-b) overflow */
    }
  }
  *r = a * b;
  return true;
}

IXS_STATIC bool ixs_safe_neg(int64_t a, int64_t *r) {
  if (a == INT64_MIN)
    return false;
  *r = -a;
  return true;
}

/* --- GCD --- */

/*
 * Binary GCD. Handles INT64_MIN by treating magnitudes as unsigned.
 * gcd(0, 0) = 0. Result is always >= 0.
 */
static uint64_t to_unsigned_mag(int64_t x) {
  if (x >= 0)
    return (uint64_t)x;
  /* x == INT64_MIN: magnitude is 2^63 = (uint64_t)INT64_MAX + 1 */
  return (uint64_t)(-(x + 1)) + 1u;
}

static int64_t u64_to_i64_clamped(uint64_t u) {
  return (u > (uint64_t)INT64_MAX) ? INT64_MAX : (int64_t)u;
}

IXS_STATIC int64_t ixs_gcd(int64_t a, int64_t b) {
  uint64_t u = to_unsigned_mag(a);
  uint64_t v = to_unsigned_mag(b);
  unsigned shift;

  if (u == 0)
    return u64_to_i64_clamped(v);
  if (v == 0)
    return u64_to_i64_clamped(u);

  /* Factor out common 2s */
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }
  while ((u & 1) == 0)
    u >>= 1;

  do {
    while ((v & 1) == 0)
      v >>= 1;
    if (u > v) {
      uint64_t t = u;
      u = v;
      v = t;
    }
    v -= u;
  } while (v != 0);

  u <<= shift;
  return u64_to_i64_clamped(u);
}

/* --- Normalize --- */

IXS_STATIC bool ixs_rat_normalize(int64_t p, int64_t q, int64_t *rp,
                                  int64_t *rq) {
  if (q == 0)
    return false;

  if (p == 0) {
    *rp = 0;
    *rq = 1;
    return true;
  }

  /* Make q positive */
  if (q < 0) {
    if (q == INT64_MIN || p == INT64_MIN)
      return false;
    p = -p;
    q = -q;
  }

  int64_t g = ixs_gcd(p, q);
  if (g > 1) {
    p /= g;
    q /= g;
  }

  *rp = p;
  *rq = q;
  return true;
}

/* --- Arithmetic --- */

IXS_STATIC bool ixs_rat_add(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq) {
  if (aq == 0 || bq == 0)
    return false;
  /*
   * a/aq + b/bq = (a*bq + b*aq) / (aq*bq)
   * Reduce first to limit overflow: divide by gcd(aq, bq).
   */
  int64_t g = ixs_gcd(aq, bq);
  int64_t aq_r = aq / g;
  int64_t bq_r = bq / g;

  int64_t t1, t2, num, den;
  if (!ixs_safe_mul(ap, bq_r, &t1))
    return false;
  if (!ixs_safe_mul(bp, aq_r, &t2))
    return false;
  if (!ixs_safe_add(t1, t2, &num))
    return false;
  if (!ixs_safe_mul(aq_r, bq, &den))
    return false;

  return ixs_rat_normalize(num, den, rp, rq);
}

IXS_STATIC bool ixs_rat_sub(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq) {
  int64_t neg_bp;
  if (!ixs_safe_neg(bp, &neg_bp))
    return false;
  return ixs_rat_add(ap, aq, neg_bp, bq, rp, rq);
}

IXS_STATIC bool ixs_rat_mul(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq) {
  if (aq == 0 || bq == 0)
    return false;
  /* Cross-reduce to limit overflow: gcd(ap, bq) and gcd(bp, aq) */
  int64_t g1 = ixs_gcd(ap, bq);
  int64_t g2 = ixs_gcd(bp, aq);
  int64_t a2 = ap / g1;
  int64_t b2 = bq / g1;
  int64_t c2 = bp / g2;
  int64_t d2 = aq / g2;

  int64_t num, den;
  if (!ixs_safe_mul(a2, c2, &num))
    return false;
  if (!ixs_safe_mul(d2, b2, &den))
    return false;

  return ixs_rat_normalize(num, den, rp, rq);
}

IXS_STATIC bool ixs_rat_div(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq) {
  if (bp == 0)
    return false;
  return ixs_rat_mul(ap, aq, bq, bp, rp, rq);
}

IXS_STATIC bool ixs_rat_neg(int64_t p, int64_t q, int64_t *rp, int64_t *rq) {
  if (!ixs_safe_neg(p, rp))
    return false;
  *rq = q;
  return true;
}

/* --- Floor / Ceil --- */

/*
 * Floored division: floor(p / q) for q > 0.
 * C division truncates toward zero; we adjust for negative dividends.
 */
IXS_STATIC int64_t ixs_rat_floor(int64_t p, int64_t q) {
  int64_t d = p / q;
  int64_t r = p % q;
  /* If remainder is negative, floor is one less than truncated. */
  if (r < 0)
    d -= 1;
  return d;
}

IXS_STATIC int64_t ixs_rat_ceil(int64_t p, int64_t q) {
  int64_t d = p / q;
  int64_t r = p % q;
  /* If remainder is positive, ceil is one more than truncated. */
  if (r > 0)
    d += 1;
  return d;
}

/* --- Floored mod --- */

IXS_STATIC bool ixs_rat_mod(int64_t ap, int64_t aq, int64_t bp, int64_t bq,
                            int64_t *rp, int64_t *rq) {
  /* mod(a, b) = a - b * floor(a / b) */
  int64_t dp, dq;
  if (!ixs_rat_div(ap, aq, bp, bq, &dp, &dq))
    return false;

  /* f = floor(dp/dq) */
  int64_t f = ixs_rat_floor(dp, dq);

  /* b * f */
  int64_t tp, tq;
  if (!ixs_rat_mul(bp, bq, f, 1, &tp, &tq))
    return false;

  /* a - b*f */
  return ixs_rat_sub(ap, aq, tp, tq, rp, rq);
}

/* --- Compare --- */

IXS_STATIC int ixs_rat_cmp(int64_t ap, int64_t aq, int64_t bp, int64_t bq) {
  /*
   * Compare a/aq vs b/bq where aq, bq > 0.
   * Equivalent to sign(a*bq - b*aq).
   * Use 128-bit to avoid overflow. Pure C99: emulate with two 64-bit ops.
   */
  /* For the domain of this library, values are small enough that
   * cross-multiply won't overflow. If it does, fall back to comparing
   * differences of reduced fractions. */
  int64_t lhs, rhs;
  bool ok1 = ixs_safe_mul(ap, bq, &lhs);
  bool ok2 = ixs_safe_mul(bp, aq, &rhs);

  if (ok1 && ok2) {
    if (lhs < rhs)
      return -1;
    if (lhs > rhs)
      return 1;
    return 0;
  }

  /* Fallback: 128-bit cross-multiply using portable C99 arithmetic.
   * Compare sign(ap*bq - bp*aq) without overflow. */
  {
    /* Compute lhs128 = ap*bq and rhs128 = bp*aq as signed 128-bit values,
     * represented as (sign, hi, lo) where value = sign * (hi*2^64 + lo). */
    uint64_t al = to_unsigned_mag(ap), bl = to_unsigned_mag(bq);
    uint64_t cl = to_unsigned_mag(bp), dl = to_unsigned_mag(aq);
    int lhs_sign = ((ap < 0) != (bq < 0)) ? -1 : 1;
    int rhs_sign = ((bp < 0) != (aq < 0)) ? -1 : 1;
    if (ap == 0)
      lhs_sign = 0;
    if (bp == 0)
      rhs_sign = 0;

    /* Unsigned 64x64 -> 128 multiply: split into 32-bit halves */
    uint64_t a_lo = al & 0xFFFFFFFFu, a_hi = al >> 32;
    uint64_t b_lo = bl & 0xFFFFFFFFu, b_hi = bl >> 32;
    uint64_t ll = a_lo * b_lo;
    uint64_t lh = a_lo * b_hi;
    uint64_t hl = a_hi * b_lo;
    uint64_t hh = a_hi * b_hi;
    uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    uint64_t lhs_lo = (ll & 0xFFFFFFFFu) | (mid << 32);
    uint64_t lhs_hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);

    uint64_t c_lo = cl & 0xFFFFFFFFu, c_hi = cl >> 32;
    uint64_t d_lo = dl & 0xFFFFFFFFu, d_hi = dl >> 32;
    ll = c_lo * d_lo;
    lh = c_lo * d_hi;
    hl = c_hi * d_lo;
    hh = c_hi * d_hi;
    mid = (ll >> 32) + (lh & 0xFFFFFFFFu) + (hl & 0xFFFFFFFFu);
    uint64_t rhs_lo = (ll & 0xFFFFFFFFu) | (mid << 32);
    uint64_t rhs_hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);

    /* Compare signed 128-bit values */
    if (lhs_sign != rhs_sign)
      return (lhs_sign > rhs_sign) ? 1 : -1;
    if (lhs_sign == 0)
      return 0;

    int mag_cmp;
    if (lhs_hi != rhs_hi)
      mag_cmp = (lhs_hi > rhs_hi) ? 1 : -1;
    else if (lhs_lo != rhs_lo)
      mag_cmp = (lhs_lo > rhs_lo) ? 1 : -1;
    else
      return 0;

    return lhs_sign > 0 ? mag_cmp : -mag_cmp;
  }
}
