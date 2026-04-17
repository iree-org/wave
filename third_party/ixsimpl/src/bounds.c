/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "bounds.h"
#include <limits.h>
#include <string.h>

#define BOUNDS_INIT_CAP 16

IXS_STATIC bool ixs_bounds_init(ixs_bounds *b, ixs_arena *scratch) {
  b->scratch = scratch;
  b->nvars = 0;
  b->cap = BOUNDS_INIT_CAP;
  b->vars = ixs_arena_alloc(scratch, BOUNDS_INIT_CAP * sizeof(*b->vars),
                            sizeof(void *));
  b->nexprs = 0;
  b->expr_cap = 0;
  b->exprs = NULL;
  return b->vars != NULL;
}

/* All bounds storage lives in the scratch arena; no per-object cleanup. */
IXS_STATIC void ixs_bounds_destroy(ixs_bounds *b) { (void)b; }

IXS_STATIC bool ixs_bounds_fork(ixs_bounds *dst, const ixs_bounds *src) {
  dst->scratch = src->scratch;
  dst->nvars = src->nvars;
  dst->cap = src->nvars ? src->nvars : 1;
  dst->vars = ixs_arena_alloc(dst->scratch, dst->cap * sizeof(*dst->vars),
                              sizeof(void *));
  if (!dst->vars)
    return false;
  if (src->nvars)
    memcpy(dst->vars, src->vars, src->nvars * sizeof(*src->vars));
  dst->nexprs = src->nexprs;
  dst->expr_cap = src->nexprs ? src->nexprs : 0;
  dst->exprs = NULL;
  if (src->nexprs) {
    dst->exprs = ixs_arena_alloc(
        dst->scratch, dst->expr_cap * sizeof(*dst->exprs), sizeof(void *));
    if (!dst->exprs)
      return false;
    memcpy(dst->exprs, src->exprs, src->nexprs * sizeof(*src->exprs));
  }
  return true;
}

static ixs_var_bound *find_var(ixs_bounds *b, const char *name) {
  size_t i;
  if (!b->vars)
    return NULL;
  for (i = 0; i < b->nvars; i++) {
    if (b->vars[i].name == name)
      return &b->vars[i];
  }
  return NULL;
}

static ixs_var_bound *get_or_create_var(ixs_bounds *b, const char *name) {
  ixs_var_bound *v = find_var(b, name);
  if (v)
    return v;
  if (!b->vars)
    return NULL;
  if (b->nvars >= b->cap) {
    ixs_var_bound *grown =
        ixs_arena_grow(b->scratch, b->vars, b->cap * sizeof(*b->vars),
                       b->cap * 2 * sizeof(*b->vars), sizeof(void *));
    if (!grown)
      return NULL;
    b->vars = grown;
    b->cap *= 2;
  }
  v = &b->vars[b->nvars++];
  v->name = name;
  v->iv.valid = true;
  ixs_interval_set_neg_inf(&v->iv.lo_p, &v->iv.lo_q);
  ixs_interval_set_pos_inf(&v->iv.hi_p, &v->iv.hi_q);
  v->modulus = 0;
  v->remainder = 0;
  return v;
}

/* Record sym ≡ rem (mod m).  Merges with existing info via CRT.
 * Contradictory or overflowing constraints are silently ignored
 * (best-effort, consistent with the interval contradiction policy). */
static void apply_modrem(ixs_bounds *b, const char *name, int64_t m,
                         int64_t rem) {
  ixs_var_bound *v;
  int64_t g, new_mod, old_mod, step, m_div_g, target, k;
  if (m <= 0)
    return;
  rem = ((rem % m) + m) % m;
  v = get_or_create_var(b, name);
  if (!v)
    return;
  if (v->modulus == 0) {
    v->modulus = m;
    v->remainder = rem;
    return;
  }
  old_mod = v->modulus;
  g = ixs_gcd(old_mod, m);
  if (((rem - v->remainder) % g + g) % g != 0)
    return;
  if (old_mod > INT64_MAX / (m / g))
    return;
  new_mod = old_mod / g * m;
  /* Solve old_mod/g * k ≡ (rem - v->remainder)/g  (mod m/g) by brute search.
   * gcd(old_mod/g, m/g) == 1 guarantees a unique solution.  Moduli are
   * small in practice (thread tile sizes), so the linear scan is fine. */
  step = old_mod / g;
  m_div_g = m / g;
  target = ((((rem - v->remainder) / g) % m_div_g) + m_div_g) % m_div_g;
  for (k = 0; k < m_div_g; k++) {
    if (((uint64_t)step * (uint64_t)k) % (uint64_t)m_div_g == (uint64_t)target)
      break;
  }
  if (k >= m_div_g)
    return;
  v->modulus = new_mod;
  v->remainder =
      (int64_t)(((uint64_t)v->remainder + (uint64_t)old_mod * (uint64_t)k) %
                (uint64_t)new_mod);
}

/* Recognize Mod(sym, M) == R as a modular congruence.
 *
 * Depends on the CMP normalizer in simp_cmp (cmp_normalize_to_zero):
 * "Mod(sym,M) == R" is rewritten to "(Mod(sym,M) - R) == 0", producing an
 * ADD node.  We must handle both the direct form (R == 0, no normalization)
 * and the normalized ADD form (R != 0). */
static void extract_modrem(ixs_bounds *b, ixs_node *a) {
  ixs_node *mod_node;
  int64_t rem_val;

  if (a->tag != IXS_CMP || a->u.binary.cmp_op != IXS_CMP_EQ)
    return;

  ixs_node *lhs = a->u.binary.lhs;
  ixs_node *rhs = a->u.binary.rhs;

  /* Direct: Mod(sym, M) == 0 */
  if (lhs->tag == IXS_MOD && ixs_node_is_zero(rhs)) {
    mod_node = lhs;
    rem_val = 0;
  } else if (rhs->tag == IXS_MOD && ixs_node_is_zero(lhs)) {
    mod_node = rhs;
    rem_val = 0;
  }
  /* Normalized: ADD(k, c*Mod(sym, M)) == 0, where c = ±1 and k is integer.
   * c=1:  Mod(sym,M) == -k;   c=-1: Mod(sym,M) == k. */
  else if (ixs_node_is_zero(rhs) && lhs->tag == IXS_ADD &&
           lhs->u.add.nterms == 1 && lhs->u.add.terms[0].term->tag == IXS_MOD) {
    int64_t cp, cq, kp, kq;
    ixs_node_get_rat(lhs->u.add.terms[0].coeff, &cp, &cq);
    ixs_node_get_rat(lhs->u.add.coeff, &kp, &kq);
    if (cq != 1 || kq != 1)
      return;
    if (cp == 1) {
      if (kp == INT64_MIN)
        return;
      rem_val = -kp;
    } else if (cp == -1) {
      rem_val = kp;
    } else {
      return;
    }
    mod_node = lhs->u.add.terms[0].term;
  } else {
    return;
  }

  /* Validate Mod operands and record the congruence. */
  {
    ixs_node *dividend = mod_node->u.binary.lhs;
    ixs_node *modulus = mod_node->u.binary.rhs;
    if (dividend->tag != IXS_SYM || modulus->tag != IXS_INT ||
        modulus->u.ival <= 0)
      return;
    rem_val = ((rem_val % modulus->u.ival) + modulus->u.ival) % modulus->u.ival;
    apply_modrem(b, dividend->u.name, modulus->u.ival, rem_val);
  }
}

static ixs_cmp_op flip_cmp(ixs_cmp_op op) {
  switch (op) {
  case IXS_CMP_GE:
    return IXS_CMP_LE;
  case IXS_CMP_GT:
    return IXS_CMP_LT;
  case IXS_CMP_LE:
    return IXS_CMP_GE;
  case IXS_CMP_LT:
    return IXS_CMP_GT;
  default:
    return op;
  }
}

/*
 * Apply "sym op const" bound to the variable's interval.
 */
static void apply_sym_cmp_const(ixs_bounds *b, const char *name, ixs_cmp_op op,
                                int64_t cp, int64_t cq) {
  ixs_var_bound *v = get_or_create_var(b, name);
  if (!v)
    return;
  switch (op) {
  case IXS_CMP_GE:
    if (ixs_rat_cmp(cp, cq, v->iv.lo_p, v->iv.lo_q) > 0) {
      v->iv.lo_p = cp;
      v->iv.lo_q = cq;
    }
    break;
  case IXS_CMP_GT: {
    int64_t lo;
    if (!ixs_safe_add(ixs_rat_floor(cp, cq), 1, &lo))
      break;
    if (ixs_rat_cmp(lo, 1, v->iv.lo_p, v->iv.lo_q) > 0) {
      v->iv.lo_p = lo;
      v->iv.lo_q = 1;
    }
    break;
  }
  case IXS_CMP_LE:
    if (ixs_rat_cmp(cp, cq, v->iv.hi_p, v->iv.hi_q) < 0) {
      v->iv.hi_p = cp;
      v->iv.hi_q = cq;
    }
    break;
  case IXS_CMP_LT: {
    int64_t hi;
    if (!ixs_safe_sub(ixs_rat_ceil(cp, cq), 1, &hi))
      break;
    if (ixs_rat_cmp(hi, 1, v->iv.hi_p, v->iv.hi_q) < 0) {
      v->iv.hi_p = hi;
      v->iv.hi_q = 1;
    }
    break;
  }
  case IXS_CMP_EQ:
    v->iv.lo_p = cp;
    v->iv.lo_q = cq;
    v->iv.hi_p = cp;
    v->iv.hi_q = cq;
    break;
  case IXS_CMP_NE:
    break;
  }
}

IXS_STATIC void ixs_bounds_add_expr(ixs_bounds *b, ixs_node *expr,
                                    ixs_interval iv) {
  ixs_expr_bound *eb;
  if (!b || !expr || !iv.valid)
    return;
  if (b->nexprs >= b->expr_cap) {
    size_t new_cap = b->expr_cap ? b->expr_cap * 2 : 4;
    ixs_expr_bound *new_arr =
        ixs_arena_alloc(b->scratch, new_cap * sizeof(*new_arr), sizeof(void *));
    if (!new_arr)
      return;
    if (b->nexprs)
      memcpy(new_arr, b->exprs, b->nexprs * sizeof(*b->exprs));
    b->exprs = new_arr;
    b->expr_cap = new_cap;
  }
  eb = &b->exprs[b->nexprs++];
  eb->expr = expr;
  eb->iv = iv;
}

/*
 * Extract interval bounds and modular congruence from a comparison.
 * Patterns: sym >= 0, sym < N, Mod(sym, M) == R, etc.
 */
IXS_STATIC void ixs_bounds_add_assumption(ixs_bounds *b, ixs_node *a) {
  if (a->tag != IXS_CMP)
    return;

  extract_modrem(b, a);

  ixs_node *lhs = a->u.binary.lhs;
  ixs_node *rhs = a->u.binary.rhs;
  ixs_cmp_op op = a->u.binary.cmp_op;

  /* Normalize to "sym op const" form. */
  if (lhs->tag == IXS_SYM && ixs_node_is_const(rhs)) {
    int64_t rp, rq;
    ixs_node_get_rat(rhs, &rp, &rq);
    apply_sym_cmp_const(b, lhs->u.name, op, rp, rq);
    return;
  }
  if (rhs->tag == IXS_SYM && ixs_node_is_const(lhs)) {
    int64_t lp, lq;
    ixs_node_get_rat(lhs, &lp, &lq);
    apply_sym_cmp_const(b, rhs->u.name, flip_cmp(op), lp, lq);
    return;
  }

  /*
   * Pattern: (sym - const) cmp 0  (from comparison normalization).
   * The lhs is an ADD with one SYM term and a constant offset.
   */
  if (ixs_node_is_zero(rhs) && lhs->tag == IXS_ADD && lhs->u.add.nterms == 1 &&
      lhs->u.add.terms[0].term->tag == IXS_SYM) {
    int64_t tp, tq, kp, kq;
    ixs_node_get_rat(lhs->u.add.terms[0].coeff, &tp, &tq);
    ixs_node_get_rat(lhs->u.add.coeff, &kp, &kq);

    /* We have: tp/tq * sym + kp/kq  OP  0, i.e. sym OP' (-kp/kq) / (tp/tq).
     * Dividing by tp/tq flips the comparison when tp/tq < 0. */
    if (tp == 0)
      return;

    /* Compute bound = -k / c = (-kp/kq) / (tp/tq) = (-kp * tq) / (kq * tp) */
    int64_t np, nq;
    if (!ixs_rat_neg(kp, kq, &np, &nq))
      return;
    int64_t raw_p, raw_q;
    if (!ixs_rat_mul(np, nq, tq, tp, &raw_p, &raw_q))
      return;
    int64_t rp2, rq2;
    if (!ixs_rat_normalize(raw_p, raw_q, &rp2, &rq2))
      return;

    ixs_cmp_op eff_op = (ixs_rat_cmp(tp, tq, 0, 1) < 0) ? flip_cmp(op) : op;
    apply_sym_cmp_const(b, lhs->u.add.terms[0].term->u.name, eff_op, rp2, rq2);
    return;
  }

  /* Fallback: expr op 0 for non-symbol lhs. Store as expression bound. */
  if (ixs_node_is_zero(rhs) && ixs_node_is_integer_valued(lhs)) {
    ixs_interval eb_iv;
    eb_iv.valid = false;
    switch (op) {
    case IXS_CMP_GT:
      eb_iv.valid = true;
      eb_iv.lo_p = 1;
      eb_iv.lo_q = 1;
      ixs_interval_set_pos_inf(&eb_iv.hi_p, &eb_iv.hi_q);
      break;
    case IXS_CMP_GE:
      eb_iv.valid = true;
      eb_iv.lo_p = 0;
      eb_iv.lo_q = 1;
      ixs_interval_set_pos_inf(&eb_iv.hi_p, &eb_iv.hi_q);
      break;
    case IXS_CMP_LT:
      eb_iv.valid = true;
      ixs_interval_set_neg_inf(&eb_iv.lo_p, &eb_iv.lo_q);
      eb_iv.hi_p = -1;
      eb_iv.hi_q = 1;
      break;
    case IXS_CMP_LE:
      eb_iv.valid = true;
      ixs_interval_set_neg_inf(&eb_iv.lo_p, &eb_iv.lo_q);
      eb_iv.hi_p = 0;
      eb_iv.hi_q = 1;
      break;
    default:
      break;
    }
    if (eb_iv.valid)
      ixs_bounds_add_expr(b, lhs, eb_iv);
  }
}

/* Conservative positive divisor of expr's value when integer-valued:
 * MUL — absolute coefficient; ADD — gcd of constant and all term
 * coefficients; everything else — 1.  Deliberately ignores nested
 * structure; the result is a lower bound on the true step. */
static int64_t mod_dividend_step(ixs_node *expr) {
  int64_t p, q, g;
  uint32_t i;
  switch (expr->tag) {
  case IXS_MUL:
    ixs_node_get_rat(expr->u.mul.coeff, &p, &q);
    return (q == 1) ? ixs_gcd(p, 0) : 1;
  case IXS_ADD:
    ixs_node_get_rat(expr->u.add.coeff, &p, &q);
    if (q != 1)
      return 1;
    g = ixs_gcd(p, 0);
    for (i = 0; i < expr->u.add.nterms; i++) {
      ixs_node_get_rat(expr->u.add.terms[i].coeff, &p, &q);
      if (q != 1)
        return 1;
      g = ixs_gcd(g, p);
    }
    return (g > 0) ? g : 1;
  default:
    return 1;
  }
}

static ixs_interval bounds_get_propagated(ixs_bounds *b, ixs_node *expr) {
  uint32_t i;
  if (!expr)
    return ixs_interval_unknown();

  switch (expr->tag) {
  case IXS_INT:
    return ixs_interval_exact(expr->u.ival, 1);
  case IXS_RAT:
    return ixs_interval_exact(expr->u.rat.p, expr->u.rat.q);
  case IXS_SYM: {
    ixs_var_bound *v = find_var(b, expr->u.name);
    if (v)
      return v->iv;
    return ixs_interval_unknown();
  }
  case IXS_ADD: {
    /* Start with constant term. */
    ixs_interval result = ixs_bounds_get(b, expr->u.add.coeff);
    for (i = 0; i < expr->u.add.nterms; i++) {
      ixs_interval ti = ixs_bounds_get(b, expr->u.add.terms[i].term);
      int64_t cp, cq;
      ixs_node_get_rat(expr->u.add.terms[i].coeff, &cp, &cq);
      ixs_interval scaled = iv_mul_const(ti, cp, cq);
      result = iv_add(result, scaled);
    }
    return result;
  }
  case IXS_MUL: {
    int64_t cp, cq;
    ixs_node_get_rat(expr->u.mul.coeff, &cp, &cq);
    ixs_interval result = ixs_interval_exact(cp, cq);
    for (i = 0; i < expr->u.mul.nfactors; i++) {
      ixs_interval fi = ixs_bounds_get(b, expr->u.mul.factors[i].base);
      if (!fi.valid)
        return ixs_interval_unknown();
      int32_t exp = expr->u.mul.factors[i].exp;
      if (exp == 1) {
        result = iv_mul(result, fi);
      } else if (exp == -1) {
        ixs_interval ri = iv_recip(fi);
        if (!ri.valid)
          return ixs_interval_unknown();
        result = iv_mul(result, ri);
      } else {
        return ixs_interval_unknown();
      }
    }
    return result;
  }
  case IXS_MOD: {
    /* Mod(x, m) in [0, m-1] only when x is integer-valued and m is a
     * positive integer.  For non-integer dividends the range is the
     * half-open [0, m) which we cannot represent tightly.
     *
     * Tighter: if x is always a multiple of d, Mod(x, m) is a multiple
     * of gcd(d, m), so the upper bound drops to m - gcd(d, m). */
    ixs_node *m = expr->u.binary.rhs;
    if (m->tag == IXS_INT && m->u.ival > 0) {
      ixs_interval pi = ixs_bounds_get(b, expr->u.binary.lhs);
      if (pi.valid && pi.lo_q == 1 && pi.hi_q == 1 && pi.lo_p >= 0 &&
          pi.hi_p < m->u.ival)
        return pi;
      if (ixs_node_is_integer_valued(expr->u.binary.lhs)) {
        int64_t step = mod_dividend_step(expr->u.binary.lhs);
        int64_t g = ixs_gcd(step, m->u.ival);
        return ixs_interval_range(0, 1, m->u.ival - g, 1);
      }
    }
    return ixs_interval_unknown();
  }
  case IXS_FLOOR: {
    ixs_interval ai = ixs_bounds_get(b, expr->u.unary.arg);
    if (!ai.valid)
      return ixs_interval_unknown();
    return ixs_interval_range(ixs_rat_floor(ai.lo_p, ai.lo_q), 1,
                              ixs_rat_floor(ai.hi_p, ai.hi_q), 1);
  }
  case IXS_CEIL: {
    ixs_interval ai = ixs_bounds_get(b, expr->u.unary.arg);
    if (!ai.valid)
      return ixs_interval_unknown();
    return ixs_interval_range(ixs_rat_ceil(ai.lo_p, ai.lo_q), 1,
                              ixs_rat_ceil(ai.hi_p, ai.hi_q), 1);
  }
  case IXS_MAX: {
    ixs_interval li = ixs_bounds_get(b, expr->u.binary.lhs);
    ixs_interval ri = ixs_bounds_get(b, expr->u.binary.rhs);
    if (!li.valid || !ri.valid)
      return ixs_interval_unknown();
    ixs_interval result;
    result.valid = true;
    /* max(lo_l, lo_r) <= max(l, r) <= max(hi_l, hi_r) */
    result.lo_p = ixs_rat_cmp(li.lo_p, li.lo_q, ri.lo_p, ri.lo_q) >= 0
                      ? li.lo_p
                      : ri.lo_p;
    result.lo_q = ixs_rat_cmp(li.lo_p, li.lo_q, ri.lo_p, ri.lo_q) >= 0
                      ? li.lo_q
                      : ri.lo_q;
    result.hi_p = ixs_rat_cmp(li.hi_p, li.hi_q, ri.hi_p, ri.hi_q) >= 0
                      ? li.hi_p
                      : ri.hi_p;
    result.hi_q = ixs_rat_cmp(li.hi_p, li.hi_q, ri.hi_p, ri.hi_q) >= 0
                      ? li.hi_q
                      : ri.hi_q;
    return result;
  }
  case IXS_MIN: {
    ixs_interval li = ixs_bounds_get(b, expr->u.binary.lhs);
    ixs_interval ri = ixs_bounds_get(b, expr->u.binary.rhs);
    if (!li.valid || !ri.valid)
      return ixs_interval_unknown();
    ixs_interval result;
    result.valid = true;
    result.lo_p = ixs_rat_cmp(li.lo_p, li.lo_q, ri.lo_p, ri.lo_q) <= 0
                      ? li.lo_p
                      : ri.lo_p;
    result.lo_q = ixs_rat_cmp(li.lo_p, li.lo_q, ri.lo_p, ri.lo_q) <= 0
                      ? li.lo_q
                      : ri.lo_q;
    result.hi_p = ixs_rat_cmp(li.hi_p, li.hi_q, ri.hi_p, ri.hi_q) <= 0
                      ? li.hi_p
                      : ri.hi_p;
    result.hi_q = ixs_rat_cmp(li.hi_p, li.hi_q, ri.hi_p, ri.hi_q) <= 0
                      ? li.hi_q
                      : ri.hi_q;
    return result;
  }
  default:
    return ixs_interval_unknown();
  }
}

IXS_STATIC ixs_interval ixs_bounds_get(ixs_bounds *b, ixs_node *expr) {
  ixs_interval iv = bounds_get_propagated(b, expr);
  if (b->nexprs && expr) {
    size_t j;
    for (j = 0; j < b->nexprs; j++) {
      if (b->exprs[j].expr == expr)
        iv = iv_intersect(iv, b->exprs[j].iv);
    }
  }
  return iv;
}

IXS_STATIC ixs_check_result ixs_bounds_check(ixs_bounds *b, ixs_node *cmp) {
  ixs_interval iv;

  if (!cmp || cmp->tag != IXS_CMP || !ixs_node_is_zero(cmp->u.binary.rhs))
    return IXS_CHECK_UNKNOWN;

  iv = ixs_bounds_get(b, cmp->u.binary.lhs);
  if (!iv.valid)
    return IXS_CHECK_UNKNOWN;

  switch (cmp->u.binary.cmp_op) {
  case IXS_CMP_GT:
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) > 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) <= 0)
      return IXS_CHECK_FALSE;
    break;
  case IXS_CMP_GE:
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) >= 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) < 0)
      return IXS_CHECK_FALSE;
    break;
  case IXS_CMP_LT:
    if (ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) < 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) >= 0)
      return IXS_CHECK_FALSE;
    break;
  case IXS_CMP_LE:
    if (ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) <= 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) > 0)
      return IXS_CHECK_FALSE;
    break;
  case IXS_CMP_EQ:
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) == 0 &&
        ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) == 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) > 0 ||
        ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) < 0)
      return IXS_CHECK_FALSE;
    break;
  case IXS_CMP_NE:
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) > 0 ||
        ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) < 0)
      return IXS_CHECK_TRUE;
    if (ixs_rat_cmp(iv.lo_p, iv.lo_q, 0, 1) == 0 &&
        ixs_rat_cmp(iv.hi_p, iv.hi_q, 0, 1) == 0)
      return IXS_CHECK_FALSE;
    break;
  }
  return IXS_CHECK_UNKNOWN;
}

IXS_STATIC bool ixs_bounds_build(ixs_bounds *b, ixs_arena *scratch,
                                 ixs_node *const *assumptions,
                                 size_t n_assumptions) {
  if (!ixs_bounds_init(b, scratch))
    return false;
  if (assumptions) {
    size_t i;
    for (i = 0; i < n_assumptions; i++) {
      ixs_node *a = assumptions[i];
      if (!a || ixs_node_is_sentinel(a))
        continue;
      ixs_bounds_add_assumption(b, a);
    }
  }
  return true;
}

IXS_STATIC bool ixs_bounds_get_modrem(ixs_bounds *b, const char *name,
                                      int64_t *mod, int64_t *rem) {
  ixs_var_bound *v;
  if (!mod || !rem)
    return false;
  v = find_var(b, name);
  if (!v || v->modulus <= 0)
    return false;
  *mod = v->modulus;
  *rem = v->remainder;
  return true;
}
