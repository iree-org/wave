/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <ixsimpl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_check.h"

static char buf[4096];

static const char *pr(ixs_node *n) {
  ixs_print(n, buf, sizeof(buf));
  return buf;
}

/* ---- Global context with stats accumulation ---- */

static ixs_ctx *g_ctx;

static ixs_ctx *ctx_create_or_die(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  if (!ctx) {
    fprintf(stderr, "ixs_ctx_create failed\n");
    exit(1);
  }
  return ctx;
}

static void atexit_handler(void) {
  size_t i, j, n_fired, n_rules;
  size_t unhit = 0;

  if (!g_ctx)
    return;

  n_fired = ixs_ctx_nstats(g_ctx);
  n_rules = ixs_nrules();

  printf("\n--- rule hit stats (%zu / %zu rules fired) ---\n", n_fired,
         n_rules);
  for (i = 0; i < n_fired; i++) {
    const char *name;
    uint64_t count = ixs_ctx_stat(g_ctx, i, &name);
    printf("  %-30s %8" PRIu64 "\n", name, count);
  }

  for (j = 0; j < n_rules; j++) {
    const char *rule = ixs_rule_name(j);
    bool found = false;
    for (i = 0; i < n_fired; i++) {
      const char *name;
      ixs_ctx_stat(g_ctx, i, &name);
      if (strcmp(name, rule) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      if (!unhit)
        printf("\n--- UNTESTED rules ---\n");
      printf("  %s\n", rule);
      unhit++;
    }
  }

  if (unhit) {
    fprintf(stderr, "%zu rule(s) never fired\n", unhit);
    ixs_ctx_destroy(g_ctx);
    g_ctx = NULL;
    _Exit(1);
  }

  printf("all %zu known rules exercised\n", n_rules);
  ixs_ctx_destroy(g_ctx);
  g_ctx = NULL;
}

static ixs_ctx *get_ctx(void) {
  if (!g_ctx) {
    g_ctx = ctx_create_or_die();
    atexit(atexit_handler);
  }
  return g_ctx;
}

static void test_add_canonicalize(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");

  /* x + x -> 2*x */
  ixs_node *r = ixs_add(ctx, x, x);
  CHECK(strcmp(pr(r), "2*x") == 0);

  /* x + 0 -> x */
  r = ixs_add(ctx, x, ixs_int(ctx, 0));
  CHECK(r == x);

  /* 0 + x -> x */
  r = ixs_add(ctx, ixs_int(ctx, 0), x);
  CHECK(r == x);

  /* 3 + 4 -> 7 */
  r = ixs_add(ctx, ixs_int(ctx, 3), ixs_int(ctx, 4));
  CHECK(ixs_node_int_val(r) == 7);

  /* (x + y) + (x + y) -> 2*x + 2*y */
  ixs_node *xy = ixs_add(ctx, x, y);
  r = ixs_add(ctx, xy, xy);
  CHECK(r && !ixs_is_error(r));
}

static void test_mul_canonicalize(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* x * 1 -> x */
  ixs_node *r = ixs_mul(ctx, x, ixs_int(ctx, 1));
  CHECK(r == x);

  /* 1 * x -> x */
  r = ixs_mul(ctx, ixs_int(ctx, 1), x);
  CHECK(r == x);

  /* x * 0 -> 0 */
  r = ixs_mul(ctx, x, ixs_int(ctx, 0));
  CHECK(ixs_node_int_val(r) == 0);

  /* 3 * 4 -> 12 */
  r = ixs_mul(ctx, ixs_int(ctx, 3), ixs_int(ctx, 4));
  CHECK(ixs_node_int_val(r) == 12);

  /* x * x -> x**2 */
  r = ixs_mul(ctx, x, x);
  CHECK(r && !ixs_is_error(r));
}

static void test_hash_consing(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x1 = ixs_sym(ctx, "x");
  ixs_node *x2 = ixs_sym(ctx, "x");
  CHECK(x1 == x2);

  ixs_node *a = ixs_add(ctx, x1, ixs_int(ctx, 1));
  ixs_node *b = ixs_add(ctx, x2, ixs_int(ctx, 1));
  CHECK(a == b);
}

static void test_floor_rules(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* floor(5) -> 5 */
  CHECK(ixs_floor(ctx, ixs_int(ctx, 5)) == ixs_int(ctx, 5));

  /* floor(7/2) -> 3 */
  CHECK(ixs_node_int_val(ixs_floor(ctx, ixs_rat(ctx, 7, 2))) == 3);

  /* floor(floor(x)) -> floor(x) */
  ixs_node *fx = ixs_floor(ctx, x);
  CHECK(ixs_floor(ctx, fx) == fx);

  /* floor(ceil(x)) -> ceil(x) */
  ixs_node *cx = ixs_ceil(ctx, x);
  CHECK(ixs_floor(ctx, cx) == cx);

  /* floor(x + 3) -> floor(x) + 3 */
  ixs_node *xp3 = ixs_add(ctx, x, ixs_int(ctx, 3));
  ixs_node *fxp3 = ixs_floor(ctx, xp3);
  ixs_node *fxp3_expected = ixs_add(ctx, ixs_floor(ctx, x), ixs_int(ctx, 3));
  CHECK(fxp3 == fxp3_expected);

  /* floor(x + 1/2) -> x  (x is integer-valued: SYM) */
  CHECK(ixs_floor(ctx, ixs_add(ctx, x, ixs_rat(ctx, 1, 2))) == x);

  /* floor extraction from ADD: floor(2*floor(x/3) + y/2)
   * -> 2*floor(x/3) + floor(y/2) */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *fx3 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *sum = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), fx3),
                            ixs_div(ctx, y, ixs_int(ctx, 2)));
    ixs_node *result = ixs_floor(ctx, sum);
    ixs_node *expected =
        ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), fx3),
                ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 2))));
    CHECK(result == expected);
  }

  /* floor extraction from MUL*ADD:
   * floor((4*floor(x/3) + y) / 2) -> 2*floor(x/3) + floor(y/2) */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *fx3 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *sum = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 4), fx3), y);
    ixs_node *result = ixs_floor(ctx, ixs_div(ctx, sum, ixs_int(ctx, 2)));
    ixs_node *expected =
        ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), fx3),
                ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 2))));
    CHECK(result == expected);
  }

  /* floor extraction with symbolic denominator:
   * floor((6*K*floor(x/3) + y) / (2*K)) -> 3*floor(x/3) + floor(y/(2*K)) */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *K = ixs_sym(ctx, "K");
    ixs_node *fx3 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *sum =
        ixs_add(ctx, ixs_mul(ctx, ixs_mul(ctx, ixs_int(ctx, 6), K), fx3), y);
    ixs_node *denom = ixs_mul(ctx, ixs_int(ctx, 2), K);
    ixs_node *result = ixs_floor(ctx, ixs_div(ctx, sum, denom));
    /* Build expected with decomposed form: (1/2)*K^(-1)*y */
    ixs_node *y_over_2K = ixs_mul(ctx, ixs_rat(ctx, 1, 2), ixs_div(ctx, y, K));
    ixs_node *expected = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 3), fx3),
                                 ixs_floor(ctx, y_over_2K));
    CHECK(result == expected);
  }

  /* ceil(x + 1/2) -> x + 1  (x is integer-valued: SYM) */
  CHECK(ixs_ceil(ctx, ixs_add(ctx, x, ixs_rat(ctx, 1, 2))) ==
        ixs_add(ctx, x, ixs_int(ctx, 1)));

  /* ceil extraction from MUL*ADD:
   * ceil((4*ceil(x/3) + y) / 2) -> 2*ceil(x/3) + ceil(y/2) */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *cx3 = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *sum = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 4), cx3), y);
    ixs_node *result = ixs_ceil(ctx, ixs_div(ctx, sum, ixs_int(ctx, 2)));
    ixs_node *expected =
        ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), cx3),
                ixs_ceil(ctx, ixs_div(ctx, y, ixs_int(ctx, 2))));
    CHECK(result == expected);
  }
}

static void test_mod_rules(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* Mod(floor(x), 1) -> 0 (only integer-valued args fold) */
  CHECK(ixs_node_int_val(ixs_mod(ctx, ixs_floor(ctx, x), ixs_int(ctx, 1))) ==
        0);

  /* Mod(17, 5) -> 2 */
  CHECK(ixs_node_int_val(ixs_mod(ctx, ixs_int(ctx, 17), ixs_int(ctx, 5))) == 2);

  /* Mod(Mod(x, 5), 5) -> Mod(x, 5) */
  ixs_node *mx5 = ixs_mod(ctx, x, ixs_int(ctx, 5));
  CHECK(ixs_mod(ctx, mx5, ixs_int(ctx, 5)) == mx5);

  /* Mod with non-integer argument must NOT fold to 0.
   * Mod(x*(x+1/3), 1) is NOT zero -- e.g. at x=2 it equals 2/3.
   * Regression: mod_bounds_elim called is_known_divisible without
   * checking integer-valuedness of the numerator. */
  {
    ixs_node *rat_prod = ixs_mul(ctx, x, ixs_add(ctx, x, ixs_rat(ctx, 1, 3)));
    ixs_node *m1 = ixs_mod(ctx, rat_prod, ixs_int(ctx, 1));
    ixs_node *neg = ixs_mul(ctx, ixs_int(ctx, -1), m1);
    ixs_node *fl = ixs_floor(ctx, neg);
    ixs_node *assumes[] = {
        ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
        ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 10)),
    };
    ixs_node *r = ixs_simplify(ctx, fl, assumes, 2);
    CHECK(r != ixs_int(ctx, 0));
  }

  /* ceiling(Mod(-Mod(y/2, 1), 1)) must NOT fold to 0.
   * At y=1: Mod(1/2, 1)=1/2, -1/2, Mod(-1/2, 1)=1/2, ceil=1. */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *m_inner =
        ixs_mod(ctx, ixs_div(ctx, y, ixs_int(ctx, 2)), ixs_int(ctx, 1));
    ixs_node *m_outer =
        ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, -1), m_inner), ixs_int(ctx, 1));
    ixs_node *ce = ixs_ceil(ctx, m_outer);
    ixs_node *assumes[] = {
        ixs_cmp(ctx, y, IXS_CMP_GE, ixs_int(ctx, 0)),
        ixs_cmp(ctx, y, IXS_CMP_LE, ixs_int(ctx, 10)),
    };
    ixs_node *r = ixs_simplify(ctx, ce, assumes, 2);
    CHECK(r != ixs_int(ctx, 0));
  }

  /* Division with compound MUL divisor: a / (a/c) -> c */
  {
    ixs_node *K = ixs_sym(ctx, "K");
    ixs_node *K_over_32 = ixs_div(ctx, K, ixs_int(ctx, 32));

    /* K / (K/32) -> 32 */
    ixs_node *q1 = ixs_div(ctx, K, K_over_32);
    CHECK(ixs_node_int_val(q1) == 32);

    /* 8*K / (K/32) -> 256 */
    ixs_node *q2 = ixs_div(ctx, ixs_mul(ctx, ixs_int(ctx, 8), K), K_over_32);
    CHECK(ixs_node_int_val(q2) == 256);
  }

  /* Symbolic modulus: Mod(T0 + 8*K*T1, K/32) -> Mod(T0, K/32) */
  {
    ixs_node *K = ixs_sym(ctx, "K");
    ixs_node *t0 = ixs_sym(ctx, "t0");
    ixs_node *t1 = ixs_sym(ctx, "t1");
    ixs_node *t2 = ixs_sym(ctx, "t2");
    ixs_node *K32 = ixs_div(ctx, K, ixs_int(ctx, 32));
    ixs_node *eight_K_t1 = ixs_mul(ctx, ixs_int(ctx, 8), ixs_mul(ctx, K, t1));
    ixs_node *K_t2 = ixs_mul(ctx, K, t2);

    /* Single addend stripped */
    ixs_node *m1 = ixs_mod(ctx, ixs_add(ctx, t0, eight_K_t1), K32);
    CHECK(m1 == ixs_mod(ctx, t0, K32));

    /* Two addends stripped */
    ixs_node *sum = ixs_add(ctx, t0, ixs_add(ctx, eight_K_t1, K_t2));
    ixs_node *m2 = ixs_mod(ctx, sum, K32);
    CHECK(m2 == ixs_mod(ctx, t0, K32));

    /* Non-multiple addend preserved */
    ixs_node *m3 = ixs_mod(ctx, ixs_add(ctx, t0, t1), K32);
    CHECK(m3 != ixs_mod(ctx, t0, K32));
  }

  /* Scale factor extraction: Mod(16*a + 1, 128*d) -> 16*Mod(a, 8*d) + 1
   * The rule is bounds-gated, so it fires only during ixs_simplify. */
  {
    ixs_node *a = ixs_sym(ctx, "a");
    ixs_node *d = ixs_sym(ctx, "d");
    ixs_node *lhs = ixs_mod(
        ctx, ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 16), a), ixs_int(ctx, 1)),
        ixs_mul(ctx, ixs_int(ctx, 128), d));
    ixs_node *eight_d = ixs_mul(ctx, ixs_int(ctx, 8), d);
    ixs_node *expected =
        ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 16), ixs_mod(ctx, a, eight_d)),
                ixs_int(ctx, 1));
    ixs_node *simplified = ixs_simplify(ctx, lhs, NULL, 0);
    CHECK(simplified == expected);

    /* Zero remainder: Mod(16*a, 128*d) -> 16*Mod(a, 8*d) */
    ixs_node *lhs2 = ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 16), a),
                             ixs_mul(ctx, ixs_int(ctx, 128), d));
    ixs_node *exp2 = ixs_mul(ctx, ixs_int(ctx, 16), ixs_mod(ctx, a, eight_d));
    ixs_node *simp2 = ixs_simplify(ctx, lhs2, NULL, 0);
    CHECK(simp2 == exp2);

    /* Coprime: Mod(3*a + 1, 7*d) -- gcd(3,7)=1, no extraction. */
    ixs_node *coprime = ixs_mod(
        ctx, ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 3), a), ixs_int(ctx, 1)),
        ixs_mul(ctx, ixs_int(ctx, 7), d));
    CHECK(ixs_node_tag(coprime) == IXS_MOD);
    CHECK(ixs_simplify(ctx, coprime, NULL, 0) == coprime);

    /* r >= g: Mod(16*a + 17, 128*d) -- r=17 >= gcd(16,128)=16. */
    ixs_node *big_r = ixs_mod(
        ctx, ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 16), a), ixs_int(ctx, 17)),
        ixs_mul(ctx, ixs_int(ctx, 128), d));
    CHECK(ixs_simplify(ctx, big_r, NULL, 0) == big_r);
  }
}

static void test_boolean(void) {
  ixs_ctx *ctx = get_ctx();

  /* True & x -> x */
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *cmp = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  CHECK(ixs_and(ctx, ixs_true(ctx), cmp) == cmp);

  /* False & x -> False */
  CHECK(ixs_and(ctx, ixs_false(ctx), cmp) == ixs_false(ctx));

  /* True | x -> True */
  CHECK(ixs_or(ctx, ixs_true(ctx), cmp) == ixs_true(ctx));

  /* ~True -> False */
  CHECK(ixs_not(ctx, ixs_true(ctx)) == ixs_false(ctx));

  /* ~~x -> x */
  CHECK(ixs_not(ctx, ixs_not(ctx, cmp)) == cmp);
}

static void test_simplify_with_bounds(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *T0 = ixs_sym(ctx, "$T0");

  ixs_node *assumptions[] = {
      ixs_cmp(ctx, T0, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, T0, IXS_CMP_LT, ixs_int(ctx, 256)),
  };

  /* Mod($T0, 256) with 0 <= $T0 < 256 -> $T0 */
  ixs_node *expr = ixs_mod(ctx, T0, ixs_int(ctx, 256));
  ixs_node *simplified = ixs_simplify(ctx, expr, assumptions, 2);
  CHECK(simplified == T0);
}

static void test_substitution(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* x + 1 with x=5 -> 6 */
  ixs_node *expr = ixs_add(ctx, x, ixs_int(ctx, 1));
  ixs_node *result = ixs_subs(ctx, expr, x, ixs_int(ctx, 5));
  CHECK(result && ixs_node_int_val(result) == 6);

  /* floor(x/2) with x=7 -> 3 */
  expr = ixs_floor(ctx, ixs_mul(ctx, x, ixs_rat(ctx, 1, 2)));
  result = ixs_subs(ctx, expr, x, ixs_int(ctx, 7));
  CHECK(result && ixs_node_int_val(result) == 3);

  /* Subtree replacement: replace Mod(x,4) with y in a larger expression */
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *mod_x4 = ixs_mod(ctx, x, ixs_int(ctx, 4));
  expr = ixs_add(ctx, mod_x4, ixs_int(ctx, 10));
  result = ixs_subs(ctx, expr, mod_x4, y);
  CHECK(result && strcmp(pr(result), "10 + y") == 0);

  /* Replace constant: 2 -> 3 in 2*x + 2 */
  ixs_node *two = ixs_int(ctx, 2);
  ixs_node *three = ixs_int(ctx, 3);
  expr = ixs_add(ctx, ixs_mul(ctx, two, x), two);
  result = ixs_subs(ctx, expr, two, three);
  CHECK(result && strcmp(pr(result), "3 + 3*x") == 0);

  /* No match: target not present leaves expression unchanged */
  expr = ixs_add(ctx, x, ixs_int(ctx, 1));
  result = ixs_subs(ctx, expr, y, ixs_int(ctx, 99));
  CHECK(result && strcmp(pr(result), "1 + x") == 0);

  /* Multi-occurrence: Mod(x,4) + 2*Mod(x,4) with Mod(x,4)->y gives 3*y */
  expr = ixs_add(ctx, mod_x4, ixs_mul(ctx, ixs_int(ctx, 2), mod_x4));
  result = ixs_subs(ctx, expr, mod_x4, y);
  CHECK(result && strcmp(pr(result), "3*y") == 0);
}

static void test_subs_multi(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *z = ixs_sym(ctx, "z");

  /* Simultaneous: {x->y, y->x} in x+y gives y+x = x+y (swap, not collapse). */
  {
    ixs_node *targets[] = {x, y};
    ixs_node *repls[] = {y, x};
    ixs_node *expr = ixs_add(ctx, x, ixs_mul(ctx, ixs_int(ctx, 2), y));
    ixs_node *r = ixs_subs_multi(ctx, expr, 2, targets, repls);
    CHECK(r == ixs_add(ctx, y, ixs_mul(ctx, ixs_int(ctx, 2), x)));
  }

  /* Multiple constants: {x->3, y->5} in x*y -> 15. */
  {
    ixs_node *targets[] = {x, y};
    ixs_node *repls[] = {ixs_int(ctx, 3), ixs_int(ctx, 5)};
    ixs_node *expr = ixs_mul(ctx, x, y);
    ixs_node *r = ixs_subs_multi(ctx, expr, 2, targets, repls);
    CHECK(r == ixs_int(ctx, 15));
  }

  /* Three targets: {x->1, y->2, z->3} in x+y+z -> 6. */
  {
    ixs_node *targets[] = {x, y, z};
    ixs_node *repls[] = {ixs_int(ctx, 1), ixs_int(ctx, 2), ixs_int(ctx, 3)};
    ixs_node *expr = ixs_add(ctx, x, ixs_add(ctx, y, z));
    ixs_node *r = ixs_subs_multi(ctx, expr, 3, targets, repls);
    CHECK(r == ixs_int(ctx, 6));
  }

  /* nsubs=0 returns expr unchanged. */
  {
    ixs_node *expr = ixs_add(ctx, x, y);
    CHECK(ixs_subs_multi(ctx, expr, 0, NULL, NULL) == expr);
  }

  /* Piecewise: subs into both branches. */
  {
    ixs_node *c = ixs_cmp(ctx, z, IXS_CMP_GT, ixs_int(ctx, 0));
    ixs_node *vals[] = {x, y};
    ixs_node *conds[] = {c, ixs_true(ctx)};
    ixs_node *pw = ixs_pw(ctx, 2, vals, conds);
    ixs_node *targets[] = {x, y};
    ixs_node *repls[] = {ixs_int(ctx, 10), ixs_int(ctx, 10)};
    ixs_node *r = ixs_subs_multi(ctx, pw, 2, targets, repls);
    CHECK(r == ixs_int(ctx, 10));
  }

  /* Negative: sequential would differ from simultaneous.
   * {x->y, y->42} in x should give y, not 42. */
  {
    ixs_node *targets[] = {x, y};
    ixs_node *repls[] = {y, ixs_int(ctx, 42)};
    ixs_node *r = ixs_subs_multi(ctx, x, 2, targets, repls);
    CHECK(r == y);
  }
}

/* Local context: error/sentinel tests push domain errors and clear them,
 * which would pollute the shared context's error list. */
static void test_sentinel_propagation(void) {
  ixs_ctx *ctx = ctx_create_or_die();
  ixs_node *x = ixs_sym(ctx, "x");

  /* NULL propagation */
  CHECK(ixs_add(ctx, NULL, x) == NULL);
  CHECK(ixs_mul(ctx, x, NULL) == NULL);

  /* Sentinel propagation */
  ixs_node *err = ixs_mod(ctx, x, ixs_int(ctx, 0));
  CHECK(ixs_is_domain_error(err));
  ixs_ctx_clear_errors(ctx);

  ixs_node *r = ixs_add(ctx, err, x);
  CHECK(ixs_is_domain_error(r));

  r = ixs_floor(ctx, err);
  CHECK(ixs_is_domain_error(r));

  ixs_ctx_destroy(ctx);
}

static void test_floor_bounds_collapse(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  ixs_node *assumptions[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 64)),
  };

  /* floor(x/64) with 0 <= x < 64 -> 0 */
  ixs_node *expr = ixs_floor(ctx, ixs_mul(ctx, x, ixs_rat(ctx, 1, 64)));
  ixs_node *r = ixs_simplify(ctx, expr, assumptions, 2);
  CHECK(r && ixs_node_int_val(r) == 0);

  /* ceiling(x/64) with 0 <= x < 64: ceil(0/64)=0, ceil(63/64)=1 — NOT constant
   */
  expr = ixs_ceil(ctx, ixs_mul(ctx, x, ixs_rat(ctx, 1, 64)));
  r = ixs_simplify(ctx, expr, assumptions, 2);
  CHECK(r && !ixs_is_error(r));
  /* Should NOT fold to a constant (0 != 1). */
  CHECK(ixs_node_tag(r) != IXS_INT);

  /* floor(x/32) with 0 <= x < 32 -> 0 */
  ixs_node *a32[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 32)),
  };
  expr = ixs_floor(ctx, ixs_mul(ctx, x, ixs_rat(ctx, 1, 32)));
  r = ixs_simplify(ctx, expr, a32, 2);
  CHECK(r && ixs_node_int_val(r) == 0);

  /* ceiling(x/32) with 0 <= x < 1 (i.e. x=0 only) -> 0 */
  ixs_node *a01[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 1)),
  };
  expr = ixs_ceil(ctx, ixs_mul(ctx, x, ixs_rat(ctx, 1, 32)));
  r = ixs_simplify(ctx, expr, a01, 2);
  CHECK(r && ixs_node_int_val(r) == 0);

  /* sym > 5/2 with integer sym -> sym >= 3 (floor(5/2) + 1 = 3) */
  ixs_node *agt[] = {
      ixs_cmp(ctx, x, IXS_CMP_GT, ixs_rat(ctx, 5, 2)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 32)),
  };
  expr = ixs_mod(ctx, x, ixs_int(ctx, 32));
  r = ixs_simplify(ctx, expr, agt, 2);
  CHECK(r == x);

  /* sym < 7/3 with integer sym -> sym <= 1 (ceil(7/3) - 1 = 1) */
  ixs_node *alt[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_rat(ctx, 7, 3)),
  };
  expr = ixs_mod(ctx, x, ixs_int(ctx, 16));
  r = ixs_simplify(ctx, expr, alt, 2);
  CHECK(r == x);

  /* 2*x >= 10 -> x >= 5; 2*x < 20 -> x < 10 -> x <= 9.
   * With x in [5, 9], Mod(x, 16) = x. */
  ixs_node *csym[] = {
      ixs_cmp(ctx, ixs_mul(ctx, ixs_int(ctx, 2), x), IXS_CMP_GE,
              ixs_int(ctx, 10)),
      ixs_cmp(ctx, ixs_mul(ctx, ixs_int(ctx, 2), x), IXS_CMP_LT,
              ixs_int(ctx, 20)),
  };
  expr = ixs_mod(ctx, x, ixs_int(ctx, 16));
  r = ixs_simplify(ctx, expr, csym, 2);
  CHECK(r == x);
}

static void test_mod_bounds_tighten(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* Mod(x, 16) with 0 <= x < 8 -> x (bounds tighter than [0,15]) */
  ixs_node *assumptions[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 8)),
  };
  ixs_node *expr = ixs_mod(ctx, x, ixs_int(ctx, 16));
  ixs_node *r = ixs_simplify(ctx, expr, assumptions, 2);
  CHECK(r == x);

  /* Mod(x, 100) with 0 <= x < 50 -> x */
  ixs_node *a50[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 50)),
  };
  expr = ixs_mod(ctx, x, ixs_int(ctx, 100));
  r = ixs_simplify(ctx, expr, a50, 2);
  CHECK(r == x);

  /* Mod(3/2*x, 1) must NOT get bounds [0,0] — dividend is not integer.
   * ceiling(Mod(3/2*x, 1)) must not collapse to 0. */
  ixs_node *half_x = ixs_div(ctx, x, ixs_int(ctx, 2));
  ixs_node *three_half_x = ixs_add(ctx, half_x, x);
  ixs_node *mod1 = ixs_mod(ctx, three_half_x, ixs_int(ctx, 1));
  ixs_node *ce = ixs_ceil(ctx, mod1);
  r = ixs_simplify(ctx, ce, NULL, 0);
  CHECK(r != ixs_int(ctx, 0));
}

static void test_mod_extract_constant(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* Mod(4*x + 3, 16) -> 3 + Mod(4*x, 16)
   * because |4| divides 16, x is integer-valued, and 3 < gcd(4)=4.
   * (floor(x) -> x since x is integer-valued.) */
  ixs_node *term = ixs_mul(ctx, ixs_int(ctx, 4), x);
  ixs_node *sum = ixs_add(ctx, term, ixs_int(ctx, 3));
  ixs_node *expr = ixs_mod(ctx, sum, ixs_int(ctx, 16));
  ixs_node *r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(strcmp(pr(r), "3 + Mod(4*x, 16)") == 0);

  /* Mod(8*x + 7, 16) -> 7 + Mod(8*x, 16) */
  term = ixs_mul(ctx, ixs_int(ctx, 8), x);
  sum = ixs_add(ctx, term, ixs_int(ctx, 7));
  expr = ixs_mod(ctx, sum, ixs_int(ctx, 16));
  r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(strcmp(pr(r), "7 + Mod(8*x, 16)") == 0);

  /* Mod(4*x + 4, 16): c=4 >= gcd(4)=4, extraction must NOT fire. */
  sum = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 4), x), ixs_int(ctx, 4));
  expr = ixs_mod(ctx, sum, ixs_int(ctx, 16));
  r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(r && !ixs_is_error(r));
  CHECK(strstr(pr(r), "4 + Mod(") == NULL);

  /* Mod(4*(x/2) + 3, 16) -> Mod(2*x + 3, 16).
   * 4*(1/2) collapses to 2, so gcd(2)=2, and 3 >= 2: no extraction. */
  ixs_node *xhalf = ixs_mul(ctx, x, ixs_rat(ctx, 1, 2));
  term = ixs_mul(ctx, ixs_int(ctx, 4), xhalf);
  sum = ixs_add(ctx, term, ixs_int(ctx, 3));
  expr = ixs_mod(ctx, sum, ixs_int(ctx, 16));
  r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(r && !ixs_is_error(r));
  CHECK(strstr(pr(r), "3 + Mod(") == NULL);

  /* Multi-term: Mod(4*x + 6*y + 3, 12).
   * gcd(4, 6) = 2, and 3 >= 2: extraction must NOT fire.
   * (Wave's original min(4,6)=4 would wrongly allow 3 < 4.) */
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *t1 = ixs_mul(ctx, ixs_int(ctx, 4), x);
  ixs_node *t2 = ixs_mul(ctx, ixs_int(ctx, 6), y);
  sum = ixs_add(ctx, ixs_add(ctx, t1, t2), ixs_int(ctx, 3));
  expr = ixs_mod(ctx, sum, ixs_int(ctx, 12));
  r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(r && !ixs_is_error(r));
  CHECK(strstr(pr(r), "3 + Mod(") == NULL);

  /* Multi-term positive: Mod(4*x + 6*y + 1, 12).
   * gcd(4, 6) = 2, and 1 < 2: extraction fires. */
  sum = ixs_add(ctx, ixs_add(ctx, t1, t2), ixs_int(ctx, 1));
  expr = ixs_mod(ctx, sum, ixs_int(ctx, 12));
  r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(strstr(pr(r), "1 + Mod(") != NULL);
}

static void test_floor_drop_small_rational(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* floor(floor(x)/3 + 1/6) with floor(x) >= 0 -> floor(floor(x)/3)
   * because floor(x) is non-neg integer, denom=3, r=1/6, 1/6 < 1/3. */
  ixs_node *assumptions[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
  };
  ixs_node *fx = ixs_floor(ctx, x);
  ixs_node *inner =
      ixs_add(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3)), ixs_rat(ctx, 1, 6));
  ixs_node *expr = ixs_floor(ctx, inner);
  ixs_node *r = ixs_simplify(ctx, expr, assumptions, 1);
  ixs_node *expected =
      ixs_simplify(ctx, ixs_floor(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3))),
                   assumptions, 1);
  CHECK(r == expected);

  /* floor(Mod(x, 8)/4 + 1/8) with 0 <= x -> floor(Mod(x,8)/4)
   * Mod(x, 8) ∈ [0,7] (non-negative integer), denom=4, r=1/8, 1/8 < 1/4. */
  ixs_node *mx8 = ixs_mod(ctx, x, ixs_int(ctx, 8));
  inner =
      ixs_add(ctx, ixs_mul(ctx, mx8, ixs_rat(ctx, 1, 4)), ixs_rat(ctx, 1, 8));
  expr = ixs_floor(ctx, inner);
  r = ixs_simplify(ctx, expr, assumptions, 1);
  expected =
      ixs_simplify(ctx, ixs_floor(ctx, ixs_mul(ctx, mx8, ixs_rat(ctx, 1, 4))),
                   assumptions, 1);
  CHECK(r == expected);

  /* Multi-term: floor(floor(x)/3 + Mod(x,8)/4 + 1/13) with x >= 0.
   * L = lcm(3, 4) = 12, r = 1/13, 1/13 < 1/12: rational is dropped. */
  ixs_node *assumptions2[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
  };
  inner = ixs_add(ctx,
                  ixs_add(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3)),
                          ixs_mul(ctx, mx8, ixs_rat(ctx, 1, 4))),
                  ixs_rat(ctx, 1, 13));
  expr = ixs_floor(ctx, inner);
  r = ixs_simplify(ctx, expr, assumptions2, 1);
  expected = ixs_simplify(
      ctx,
      ixs_floor(ctx, ixs_add(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3)),
                             ixs_mul(ctx, mx8, ixs_rat(ctx, 1, 4)))),
      assumptions2, 1);
  CHECK(r == expected);

  /* floor(floor(x)/3 + 1/3) should NOT drop: 1/3 is not < 1/3. */
  inner =
      ixs_add(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3)), ixs_rat(ctx, 1, 3));
  expr = ixs_floor(ctx, inner);
  r = ixs_simplify(ctx, expr, assumptions, 1);
  CHECK(r && !ixs_is_error(r));
  ixs_node *without_r =
      ixs_simplify(ctx, ixs_floor(ctx, ixs_mul(ctx, fx, ixs_rat(ctx, 1, 3))),
                   assumptions, 1);
  CHECK(!ixs_same_node(r, without_r));
}

static void test_nested_floor_ceil(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* floor(floor(x/3) / 5) -> floor(x/15) */
  ixs_node *inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
  ixs_node *e = ixs_floor(ctx, ixs_div(ctx, inner, ixs_int(ctx, 5)));
  CHECK(e && strcmp(pr(e), "floor(1/15*x)") == 0);

  /* ceiling(ceiling(x/4) / 3) -> ceiling(x/12) */
  inner = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  e = ixs_ceil(ctx, ixs_div(ctx, inner, ixs_int(ctx, 3)));
  CHECK(e && strcmp(pr(e), "ceiling(1/12*x)") == 0);

  /* floor(floor(x/2) / 2) -> floor(x/4) */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 2)));
  e = ixs_floor(ctx, ixs_div(ctx, inner, ixs_int(ctx, 2)));
  CHECK(e && strcmp(pr(e), "floor(1/4*x)") == 0);

  /* Negative: floor(2*floor(x/3) / 5) should NOT collapse */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
  e = ixs_floor(ctx, ixs_mul(ctx, ixs_rat(ctx, 2, 5), inner));
  CHECK(e && strstr(pr(e), "floor") != NULL);

  /* Mod(a*floor(x/a), a) -> 0 */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  e = ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 4), inner), ixs_int(ctx, 4));
  CHECK(e == ixs_int(ctx, 0));

  /* Mod(6*floor(x/3), 3) -> 0 (6 is multiple of 3) */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
  e = ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 6), inner), ixs_int(ctx, 3));
  CHECK(e == ixs_int(ctx, 0));

  /* Negative: Mod(3*floor(x/4), 4) should NOT simplify to 0 */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  e = ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 3), inner), ixs_int(ctx, 4));
  CHECK(e != ixs_int(ctx, 0));

  /* Negative: ceiling(2*ceiling(x/4) / 3) should NOT collapse */
  inner = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  e = ixs_ceil(ctx, ixs_mul(ctx, ixs_rat(ctx, 2, 3), inner));
  CHECK(e && strstr(pr(e), "ceiling") != NULL);

  /* Negative: floor(floor(x/3) * 2) -> 2*floor(x/3) (integer, no nesting) */
  inner = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
  e = ixs_floor(ctx, ixs_mul(ctx, ixs_int(ctx, 2), inner));
  CHECK(e && strcmp(pr(e), "2*floor(1/3*x)") == 0);
}

static void test_same_node(void) {
  ixs_ctx *ctx = get_ctx();
  CHECK(ixs_same_node(NULL, NULL));
  CHECK(!ixs_same_node(ixs_int(ctx, 1), NULL));
  CHECK(ixs_same_node(ixs_int(ctx, 42), ixs_int(ctx, 42)));
}

static void test_print_roundtrip(void) {
  ixs_ctx *ctx = get_ctx();

  const char *exprs[] = {
      "x + y",     "3*x + 2",   "floor(x/2)", "ceiling(x + 1)",
      "Mod(x, 5)", "Max(x, y)", "Min(x, y)",  "xor(x, y)",
  };
  size_t i;
  for (i = 0; i < sizeof(exprs) / sizeof(exprs[0]); i++) {
    ixs_node *n = ixs_parse(ctx, exprs[i], strlen(exprs[i]));
    CHECK(n && !ixs_is_error(n));

    char out[1024];
    ixs_print(n, out, sizeof(out));

    /* Re-parse the printed output. */
    ixs_node *n2 = ixs_parse(ctx, out, strlen(out));
    CHECK(n2 && !ixs_is_error(n2));
    CHECK(ixs_same_node(n, n2));
  }
}

static void test_divisibility_assumptions(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *N = ixs_sym(ctx, "N");
  ixs_node *M = ixs_sym(ctx, "M");
  ixs_node *r;

  /* Assumption: Mod(K, 32) == 0  (K is divisible by 32) */
  ixs_node *div_K_32[] = {
      ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 32)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };

  /* floor(K/32) -> K/32 when 32 | K */
  ixs_node *e1 = ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 32)));
  r = ixs_simplify(ctx, e1, div_K_32, 1);
  CHECK(strcmp(pr(r), "1/32*K") == 0);

  /* Mod(K, 32) -> 0 when 32 | K */
  ixs_node *e2 = ixs_mod(ctx, K, ixs_int(ctx, 32));
  r = ixs_simplify(ctx, e2, div_K_32, 1);
  CHECK(r == ixs_int(ctx, 0));

  /* floor(K/16) -> K/16 since 32 | K implies 16 | K */
  ixs_node *e3 = ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 16)));
  r = ixs_simplify(ctx, e3, div_K_32, 1);
  CHECK(strcmp(pr(r), "1/16*K") == 0);

  /* Mod(K, 64) should NOT simplify to 0 (32 | K does not imply 64 | K) */
  ixs_node *e4 = ixs_mod(ctx, K, ixs_int(ctx, 64));
  r = ixs_simplify(ctx, e4, div_K_32, 1);
  CHECK(r != ixs_int(ctx, 0));

  /* Mod(3*K, 32) -> 0 when 32 | K */
  ixs_node *e5 =
      ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 3), K), ixs_int(ctx, 32));
  r = ixs_simplify(ctx, e5, div_K_32, 1);
  CHECK(r == ixs_int(ctx, 0));

  /* Multiple assumptions: Mod(K, 32)==0 and Mod(N, 16)==0 */
  ixs_node *multi_div[] = {
      ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 32)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
      ixs_cmp(ctx, ixs_mod(ctx, N, ixs_int(ctx, 16)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, K, ixs_int(ctx, 32)), multi_div, 2);
  CHECK(r == ixs_int(ctx, 0));
  r = ixs_simplify(ctx, ixs_mod(ctx, N, ixs_int(ctx, 16)), multi_div, 2);
  CHECK(r == ixs_int(ctx, 0));
  r = ixs_simplify(ctx, ixs_floor(ctx, ixs_div(ctx, N, ixs_int(ctx, 16))),
                   multi_div, 2);
  CHECK(strcmp(pr(r), "1/16*N") == 0);

  /* Mixed: floor(K/32) + Mod(K, 32) -> K/32 when 32 | K */
  ixs_node *e6 = ixs_add(ctx, ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 32))),
                         ixs_mod(ctx, K, ixs_int(ctx, 32)));
  r = ixs_simplify(ctx, e6, div_K_32, 1);
  CHECK(strcmp(pr(r), "1/32*K") == 0);

  /* Stronger assumption implies weaker: Mod(M, 256)==0 with tile=128 */
  ixs_node *div_M_256[] = {
      ixs_cmp(ctx, ixs_mod(ctx, M, ixs_int(ctx, 256)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, M, ixs_int(ctx, 128)), div_M_256, 1);
  CHECK(r == ixs_int(ctx, 0));
  r = ixs_simplify(ctx, ixs_floor(ctx, ixs_div(ctx, M, ixs_int(ctx, 128))),
                   div_M_256, 1);
  CHECK(strcmp(pr(r), "1/128*M") == 0);

  /* Negative: floor(K/64) with 32|K should NOT drop floor */
  ixs_node *e_neg = ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 64)));
  r = ixs_simplify(ctx, e_neg, div_K_32, 1);
  CHECK(strstr(pr(r), "floor") != NULL);

  /* No assumptions: expressions pass through unchanged */
  ixs_node *e7 = ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 32)));
  r = ixs_simplify(ctx, e7, NULL, 0);
  CHECK(strstr(pr(r), "floor") != NULL);

  /* ceiling(K/32) -> K/32 when 32 | K */
  ixs_node *e8 = ixs_ceil(ctx, ixs_div(ctx, K, ixs_int(ctx, 32)));
  r = ixs_simplify(ctx, e8, div_K_32, 1);
  CHECK(strcmp(pr(r), "1/32*K") == 0);

  /* Multi-factor: floor(K/2 * N) -> K/2 * N when 32|K and 16|N.
   * K/2*N = MUL(1/2, [K^1, N^1]); K absorbs the denominator 2. */
  {
    ixs_node *div_K32_N16[] = {
        ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 32)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
        ixs_cmp(ctx, ixs_mod(ctx, N, ixs_int(ctx, 16)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
    };
    ixs_node *prod = ixs_mul(ctx, ixs_div(ctx, K, ixs_int(ctx, 2)), N);
    r = ixs_simplify(ctx, ixs_floor(ctx, prod), div_K32_N16, 2);
    CHECK(strstr(pr(r), "floor") == NULL);

    /* Negative: floor(K/64 * N) with 32|K -- K not divisible by 64. */
    ixs_node *prod2 = ixs_mul(ctx, ixs_div(ctx, K, ixs_int(ctx, 64)), N);
    r = ixs_simplify(ctx, ixs_floor(ctx, prod2), div_K32_N16, 2);
    CHECK(strstr(pr(r), "floor") != NULL);
  }
}

static void test_large_expressions(void) {
  ixs_ctx *ctx = get_ctx();
  int i;

  /* ADD with >256 distinct terms. */
  {
    ixs_node *sum = ixs_int(ctx, 0);
    char name[16];
    for (i = 0; i < 300; i++) {
      snprintf(name, sizeof(name), "s%d", i);
      sum = ixs_add(ctx, sum, ixs_sym(ctx, name));
      CHECK(sum != NULL && !ixs_is_error(sum));
    }
    CHECK(ixs_node_tag(sum) == IXS_ADD);
  }

  /* MUL with >256 distinct factors. */
  {
    ixs_node *prod = ixs_int(ctx, 1);
    char name[16];
    for (i = 0; i < 300; i++) {
      snprintf(name, sizeof(name), "m%d", i);
      prod = ixs_mul(ctx, prod, ixs_sym(ctx, name));
      CHECK(prod != NULL && !ixs_is_error(prod));
    }
    CHECK(ixs_node_tag(prod) == IXS_MUL);
  }

  /* AND with >256 distinct args. */
  {
    ixs_node *conj = ixs_true(ctx);
    char name[16];
    for (i = 0; i < 300; i++) {
      snprintf(name, sizeof(name), "a%d", i);
      ixs_node *cmp =
          ixs_cmp(ctx, ixs_sym(ctx, name), IXS_CMP_GT, ixs_int(ctx, 0));
      conj = ixs_and(ctx, conj, cmp);
      CHECK(conj != NULL && !ixs_is_error(conj));
    }
    CHECK(ixs_node_tag(conj) == IXS_AND);
  }

  /* Piecewise with >256 cases. */
  {
    ixs_node **vals = malloc(300 * sizeof(*vals));
    ixs_node **conds = malloc(300 * sizeof(*conds));
    CHECK(vals != NULL && conds != NULL);
    char name[16];
    for (i = 0; i < 299; i++) {
      snprintf(name, sizeof(name), "p%d", i);
      vals[i] = ixs_sym(ctx, name);
      conds[i] = ixs_cmp(ctx, ixs_sym(ctx, name), IXS_CMP_GT, ixs_int(ctx, 0));
    }
    vals[299] = ixs_int(ctx, 0);
    conds[299] = ixs_true(ctx);
    ixs_node *pw = ixs_pw(ctx, 300, vals, conds);
    CHECK(pw != NULL && !ixs_is_error(pw));
    free(vals);
    free(conds);
  }
}

static void test_bounds_many_vars(void) {
  ixs_ctx *ctx = get_ctx();
  int i;

  /* Build 100 symbols each with bounds: 0 <= v_i < 256.
   * Then Mod(v_i, 256) should simplify to v_i for all of them. */
  ixs_node *assumptions[200];
  ixs_node *syms[100];
  char name[16];
  for (i = 0; i < 100; i++) {
    snprintf(name, sizeof(name), "v%d", i);
    syms[i] = ixs_sym(ctx, name);
    assumptions[2 * i] = ixs_cmp(ctx, syms[i], IXS_CMP_GE, ixs_int(ctx, 0));
    assumptions[2 * i + 1] =
        ixs_cmp(ctx, syms[i], IXS_CMP_LT, ixs_int(ctx, 256));
  }

  /* Simplify Mod(v_99, 256) — the 100th variable — to v_99. */
  ixs_node *expr = ixs_mod(ctx, syms[99], ixs_int(ctx, 256));
  ixs_node *r = ixs_simplify(ctx, expr, assumptions, 200);
  CHECK(r == syms[99]);

  /* Also check an early one. */
  expr = ixs_mod(ctx, syms[0], ixs_int(ctx, 256));
  r = ixs_simplify(ctx, expr, assumptions, 200);
  CHECK(r == syms[0]);
}

static void test_mod_floor_regression(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *K = ixs_sym(ctx, "K");

  /* Mod(x + k*d, d) -> Mod(x, d): constant multiple of modulus absorbed */
  CHECK(ixs_mod(ctx, ixs_add(ctx, x, ixs_int(ctx, 32)), ixs_int(ctx, 16)) ==
        ixs_mod(ctx, x, ixs_int(ctx, 16)));
  CHECK(ixs_mod(ctx, ixs_add(ctx, x, ixs_int(ctx, 48)), ixs_int(ctx, 16)) ==
        ixs_mod(ctx, x, ixs_int(ctx, 16)));

  /* Mod(n*x, n) -> 0 for integer-valued x */
  CHECK(ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 16), x), ixs_int(ctx, 16)) ==
        ixs_int(ctx, 0));
  CHECK(ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 32), x), ixs_int(ctx, 16)) ==
        ixs_int(ctx, 0));

  /* floor(Mod(x, n)) -> Mod(x, n): Mod of integers is integer-valued */
  ixs_node *mx16 = ixs_mod(ctx, x, ixs_int(ctx, 16));
  CHECK(ixs_floor(ctx, mx16) == mx16);

  /* ceiling(Mod(x, n)) -> Mod(x, n) */
  CHECK(ixs_ceil(ctx, mx16) == mx16);

  /* floor(Mod(x, 64)/16) stays as-is (mod-then-divide is the preferred form).
   */
  ixs_node *subfield = ixs_floor(
      ctx, ixs_div(ctx, ixs_mod(ctx, x, ixs_int(ctx, 64)), ixs_int(ctx, 16)));
  CHECK(ixs_node_tag(subfield) == IXS_FLOOR);

  /* floor(x + 1/2) -> x for integer-valued x (fractional part drops) */
  ixs_node *fhalf = ixs_floor(ctx, ixs_add(ctx, x, ixs_rat(ctx, 1, 2)));
  CHECK(fhalf == x);

  /* ceil(x + 1/2) -> x + 1 for integer-valued x */
  ixs_node *chalf = ixs_ceil(ctx, ixs_add(ctx, x, ixs_rat(ctx, 1, 2)));
  CHECK(chalf == ixs_add(ctx, x, ixs_int(ctx, 1)));

  /* floor((4*floor(x/3) + y) / 2) -> 2*floor(x/3) + floor(y/2)
   * MUL-over-ADD extraction with integer-valued product. */
  ixs_node *fx3 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
  ixs_node *e = ixs_floor(
      ctx, ixs_div(ctx, ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 4), fx3), y),
                   ixs_int(ctx, 2)));
  ixs_node *expected =
      ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), fx3),
              ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 2))));
  CHECK(e == expected);

  /* floor((6*K*floor(x/3) + y) / (2*K)) -> 3*floor(x/3) + floor(y/(2*K))
   * Symbolic denominator cancellation. */
  ixs_node *outer_num =
      ixs_add(ctx, ixs_mul(ctx, ixs_mul(ctx, ixs_int(ctx, 6), K), fx3), y);
  ixs_node *outer_den = ixs_mul(ctx, ixs_int(ctx, 2), K);
  e = ixs_floor(ctx, ixs_div(ctx, outer_num, outer_den));
  CHECK(strcmp(pr(e), "3*floor(1/3*x) + floor(1/2*1/K*y)") == 0);

  /* Mod(8*floor(x/4), 4) -> 0: coefficient is multiple of modulus */
  ixs_node *fx4 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  CHECK(ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, 8), fx4), ixs_int(ctx, 4)) ==
        ixs_int(ctx, 0));

  /* Mod(Mod(x, 32), 16): nested Mod where inner > outer.
   * Currently not collapsed; verify it doesn't crash or produce garbage. */
  ixs_node *nested =
      ixs_mod(ctx, ixs_mod(ctx, x, ixs_int(ctx, 32)), ixs_int(ctx, 16));
  CHECK(nested != NULL && !ixs_is_error(nested));
}

static void test_mod_recognition(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *cx = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 8)));

  /* x - 32*floor(x/32) -> Mod(x, 32) */
  ixs_node *e =
      ixs_add(ctx, x,
              ixs_mul(ctx, ixs_int(ctx, -32),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))));
  CHECK(e == ixs_mod(ctx, x, ixs_int(ctx, 32)));

  /* ceiling(x/8) - 32*floor(ceiling(x/8)/32) -> Mod(ceiling(x/8), 32) */
  e = ixs_add(ctx, cx,
              ixs_mul(ctx, ixs_int(ctx, -32),
                      ixs_floor(ctx, ixs_div(ctx, cx, ixs_int(ctx, 32)))));
  CHECK(e == ixs_mod(ctx, cx, ixs_int(ctx, 32)));

  /* With a scalar: 3*x - 96*floor(x/32) -> 3*Mod(x, 32)  (96 = 3*32) */
  e = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 3), x),
              ixs_mul(ctx, ixs_int(ctx, -96),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))));
  CHECK(e == ixs_mul(ctx, ixs_int(ctx, 3), ixs_mod(ctx, x, ixs_int(ctx, 32))));

  /* Extra terms preserved: y + x - 32*floor(x/32) -> y + Mod(x, 32) */
  e = ixs_add(ctx, ixs_add(ctx, y, x),
              ixs_mul(ctx, ixs_int(ctx, -32),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))));
  CHECK(e == ixs_add(ctx, y, ixs_mod(ctx, x, ixs_int(ctx, 32))));

  /* Constant offset preserved: 5 + x - 32*floor(x/32) -> 5 + Mod(x, 32) */
  e = ixs_add(ctx, ixs_add(ctx, ixs_int(ctx, 5), x),
              ixs_mul(ctx, ixs_int(ctx, -32),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))));
  CHECK(e == ixs_add(ctx, ixs_int(ctx, 5), ixs_mod(ctx, x, ixs_int(ctx, 32))));

  /* No false match: 5*x - 32*floor(x/32) stays as is (5 != 1, 5*32 != 32) */
  e = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 5), x),
              ixs_mul(ctx, ixs_int(ctx, -32),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))));
  CHECK(ixs_node_tag(e) == IXS_ADD);

  /* Ceiling padding: N*ceil(x/N) - x -> Mod(-x, N) */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 32),
                      ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))),
              ixs_mul(ctx, ixs_int(ctx, -1), x));
  CHECK(e == ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, -1), x), ixs_int(ctx, 32)));

  /* Scaled ceiling: 3*32*ceil(x/32) - 3*x -> 3*Mod(-x, 32) */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 96),
                      ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))),
              ixs_mul(ctx, ixs_int(ctx, -3), x));
  CHECK(e == ixs_mul(ctx, ixs_int(ctx, 3),
                     ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, -1), x),
                             ixs_int(ctx, 32))));

  /* Extra terms with ceiling: y + 4*ceil(x/4) - x -> y + Mod(-x, 4) */
  e = ixs_add(ctx,
              ixs_add(ctx, y,
                      ixs_mul(ctx, ixs_int(ctx, 4),
                              ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 4))))),
              ixs_mul(ctx, ixs_int(ctx, -1), x));
  CHECK(e == ixs_add(ctx, y,
                     ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, -1), x),
                             ixs_int(ctx, 4))));

  /* No false match: 32*ceil(x/32) - 5*x stays as is (5 != 1, 5*32 != 32) */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 32),
                      ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)))),
              ixs_mul(ctx, ixs_int(ctx, -5), x));
  CHECK(ixs_node_tag(e) == IXS_ADD);

  /* Symbolic divisor: E - G*floor(E/G) -> Mod(E, G) */
  {
    ixs_node *G = ixs_sym(ctx, "G");
    ixs_node *E = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 192)));
    e = ixs_add(ctx, E,
                ixs_mul(ctx, ixs_int(ctx, -1),
                        ixs_mul(ctx, G, ixs_floor(ctx, ixs_div(ctx, E, G)))));
    CHECK(e == ixs_mod(ctx, E, G));
  }

  /* Scaled symbolic: 3*E - 3*G*floor(E/G) -> 3*Mod(E, G) */
  {
    ixs_node *G = ixs_sym(ctx, "G");
    ixs_node *E = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 192)));
    e = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 3), E),
                ixs_mul(ctx, ixs_int(ctx, -3),
                        ixs_mul(ctx, G, ixs_floor(ctx, ixs_div(ctx, E, G)))));
    CHECK(e == ixs_mul(ctx, ixs_int(ctx, 3), ixs_mod(ctx, E, G)));
  }

  /* Symbolic ceil: G*ceil(x/G) - x -> Mod(-x, G) */
  {
    ixs_node *G = ixs_sym(ctx, "G");
    e = ixs_add(ctx, ixs_mul(ctx, G, ixs_ceil(ctx, ixs_div(ctx, x, G))),
                ixs_mul(ctx, ixs_int(ctx, -1), x));
    CHECK(e == ixs_mod(ctx, ixs_mul(ctx, ixs_int(ctx, -1), x), G));
  }

  /* Negative: E - G*floor(x/G) stays (dividend mismatch) */
  {
    ixs_node *G = ixs_sym(ctx, "G");
    ixs_node *E = ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 192)));
    e = ixs_add(ctx, E,
                ixs_mul(ctx, ixs_int(ctx, -1),
                        ixs_mul(ctx, G, ixs_floor(ctx, ixs_div(ctx, x, G)))));
    CHECK(ixs_node_tag(e) == IXS_ADD);
  }
}

static void test_floor_mod_divisor(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* floor(Mod(x, 64) / 16) stays: the "mod-then-divide" form is the natural
   * hardware idiom for GPU thread index decomposition and maps directly to
   * two affine ops.  Rewriting to Mod(floor(x/16), 4) is complexity-neutral
   * and obscures the hardware mapping. */
  ixs_node *e = ixs_floor(
      ctx, ixs_div(ctx, ixs_mod(ctx, x, ixs_int(ctx, 64)), ixs_int(ctx, 16)));
  CHECK(ixs_node_tag(e) == IXS_FLOOR);

  /* floor(Mod(x, 32) / 32) -> 0 (range of Mod is [0, 31], divided by 32 < 1,
   * floor rounds to 0). */
  e = ixs_floor(
      ctx, ixs_div(ctx, ixs_mod(ctx, x, ixs_int(ctx, 32)), ixs_int(ctx, 32)));
  CHECK(e == ixs_int(ctx, 0));
}

/* A - PW((A+B, c), (A+C, ~c)) + PW((B, c), (C, ~c)) should fold to 0. */
static void test_pw_fold_in_add(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *c = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *A = ixs_mul(ctx, ixs_int(ctx, 128), y);
  ixs_node *B = ixs_mod(ctx, x, ixs_int(ctx, 16));
  ixs_node *C =
      ixs_mul(ctx, ixs_int(ctx, 4),
              ixs_floor(ctx, ixs_div(ctx, ixs_mod(ctx, x, ixs_int(ctx, 64)),
                                     ixs_int(ctx, 16))));

  ixs_node *v1[] = {ixs_add(ctx, A, B), ixs_add(ctx, A, C)};
  ixs_node *c1[] = {c, ixs_true(ctx)};
  ixs_node *pw1 = ixs_pw(ctx, 2, v1, c1);

  ixs_node *v2[] = {B, C};
  ixs_node *c2[] = {c, ixs_true(ctx)};
  ixs_node *pw2 = ixs_pw(ctx, 2, v2, c2);

  /* Verify conditions are pointer-equal (hash-consed). */
  CHECK(ixs_node_pw_cond(pw1, 0) == ixs_node_pw_cond(pw2, 0));
  CHECK(ixs_node_pw_cond(pw1, 1) == ixs_node_pw_cond(pw2, 1));

  /* A - pw1 + pw2 = 0 */
  ixs_node *expr = ixs_add(ctx, ixs_sub(ctx, A, pw1), pw2);
  CHECK(expr == ixs_int(ctx, 0));
  ixs_node *r = ixs_simplify(ctx, expr, NULL, 0);
  CHECK(r == ixs_int(ctx, 0));

  /* Also test via from_sympy-like path: sequential add */
  ixs_node *neg_pw1 = ixs_mul(ctx, ixs_int(ctx, -1), pw1);
  ixs_node *s1 = ixs_add(ctx, A, neg_pw1);
  ixs_node *s2 = ixs_add(ctx, s1, pw2);
  CHECK(s2 == ixs_int(ctx, 0));

  /* Negative: different conditions must NOT fold. */
  ixs_node *d = ixs_cmp(ctx, y, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *v3[] = {B, C};
  ixs_node *c3[] = {d, ixs_true(ctx)};
  ixs_node *pw3 = ixs_pw(ctx, 2, v3, c3);
  ixs_node *no_fold = ixs_add(ctx, ixs_sub(ctx, A, pw1), pw3);
  CHECK(no_fold != ixs_int(ctx, 0));
}

static void test_piecewise_branch_bounds(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *modx = ixs_mod(ctx, x, ixs_int(ctx, 32));

  /* Piecewise((Max(1, Mod(x,32)), Mod(x,32) > 0), (1, True))
   * With x >= 0 assumption, the first branch should collapse Max -> Mod. */
  ixs_node *cond = ixs_cmp(ctx, modx, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *v1 = ixs_max(ctx, ixs_int(ctx, 1), modx);
  ixs_node *v2 = ixs_int(ctx, 1);
  ixs_node *vals[] = {v1, v2};
  ixs_node *cds[] = {cond, ixs_true(ctx)};
  ixs_node *pw = ixs_pw(ctx, 2, vals, cds);

  ixs_node *assume = ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0));
  ixs_node *result = ixs_simplify(ctx, pw, &assume, 1);
  CHECK(result != NULL);

  /* Verify Max(1, ...) no longer appears in the result. */
  {
    char buf[512];
    ixs_print(result, buf, sizeof(buf));
    CHECK(strstr(buf, "Max(") == NULL);
  }
}

static void test_product_zero_branch_collapse(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *N = ixs_sym(ctx, "N");
  ixs_node *M = ixs_sym(ctx, "M");
  ixs_node *C = ixs_ceil(ctx, ixs_div(ctx, N, ixs_int(ctx, 192)));
  ixs_node *CM = ixs_ceil(ctx, ixs_div(ctx, M, ixs_int(ctx, 256)));
  ixs_node *fc = ixs_floor(ctx, ixs_div(ctx, C, ixs_int(ctx, 32)));
  ixs_node *mc = ixs_mod(ctx, C, ixs_int(ctx, 32));

  /* Piecewise((floor(-32*floor(C/32)*ceil(M/256)/Mod(C,32)),
   *            Mod(C,32) > 0 & floor(C/32)*ceil(M/256) <= 0),
   *           (0, True))
   * Should collapse to 0: the guard pins floor(C/32) to 0, making
   * the branch value = floor(0) = 0 = default branch. */
  ixs_node *branch = ixs_floor(
      ctx,
      ixs_div(ctx, ixs_mul(ctx, ixs_int(ctx, -32), ixs_mul(ctx, fc, CM)), mc));
  ixs_node *guard =
      ixs_and(ctx, ixs_cmp(ctx, mc, IXS_CMP_GT, ixs_int(ctx, 0)),
              ixs_cmp(ctx, ixs_mul(ctx, fc, CM), IXS_CMP_LE, ixs_int(ctx, 0)));
  ixs_node *vals[] = {branch, ixs_int(ctx, 0)};
  ixs_node *cds[] = {guard, ixs_true(ctx)};
  ixs_node *pw = ixs_pw(ctx, 2, vals, cds);

  ixs_node *assumes[] = {
      ixs_cmp(ctx, N, IXS_CMP_GE, ixs_int(ctx, 1)),
      ixs_cmp(ctx, M, IXS_CMP_GE, ixs_int(ctx, 1)),
  };
  ixs_node *result = ixs_simplify(ctx, pw, assumes, 2);
  CHECK(result == ixs_int(ctx, 0));

  /* Negative: when both factors could be zero, decomposition must not
   * fire.  floor(x)*floor(y) <= 0 with x,y >= 0: either factor could
   * be zero, so we cannot pin one to 0. */
  {
    ixs_node *x = ixs_sym(ctx, "x");
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *fx = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 32)));
    ixs_node *fy = ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 32)));
    ixs_node *prod = ixs_mul(ctx, fx, fy);
    ixs_node *neg_branch = ixs_floor(ctx, ixs_div(ctx, prod, ixs_int(ctx, 7)));
    ixs_node *neg_guard = ixs_cmp(ctx, prod, IXS_CMP_LE, ixs_int(ctx, 0));
    ixs_node *neg_vals[] = {neg_branch, ixs_int(ctx, 0)};
    ixs_node *neg_cds[] = {neg_guard, ixs_true(ctx)};
    ixs_node *neg_pw = ixs_pw(ctx, 2, neg_vals, neg_cds);
    ixs_node *neg_assumes[] = {
        ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
        ixs_cmp(ctx, y, IXS_CMP_GE, ixs_int(ctx, 0)),
    };
    ixs_node *neg_result = ixs_simplify(ctx, neg_pw, neg_assumes, 2);
    CHECK(neg_result != ixs_int(ctx, 0));
  }
}

static void test_floor_symbolic_denom(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *K = ixs_sym(ctx, "K");

  /* floor(x / (128*K)) -> 0 when 0 <= x <= 127, K >= 1.
   * Bounds: x in [0,127], 128*K in [128, inf), so x/(128*K) in [0, 127/128). */
  ixs_node *e =
      ixs_floor(ctx, ixs_div(ctx, x, ixs_mul(ctx, ixs_int(ctx, 128), K)));
  ixs_node *assumes[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 127)),
      ixs_cmp(ctx, K, IXS_CMP_GE, ixs_int(ctx, 1)),
  };
  ixs_node *r = ixs_simplify(ctx, e, assumes, 3);
  CHECK(r == ixs_int(ctx, 0));

  /* floor(Mod(x, 16) / (128*K)) -> 0 with K >= 1 (Mod range [0,15]). */
  ixs_node *e2 = ixs_floor(ctx, ixs_div(ctx, ixs_mod(ctx, x, ixs_int(ctx, 16)),
                                        ixs_mul(ctx, ixs_int(ctx, 128), K)));
  ixs_node *assume_k = ixs_cmp(ctx, K, IXS_CMP_GE, ixs_int(ctx, 1));
  r = ixs_simplify(ctx, e2, &assume_k, 1);
  CHECK(r == ixs_int(ctx, 0));

  /* Difference: floor(A/D) - floor((A+1)/D) = 0 when A in [0,15], D >= 128. */
  ixs_node *A = ixs_mod(ctx, x, ixs_int(ctx, 16));
  ixs_node *D = ixs_mul(ctx, ixs_int(ctx, 128), K);
  ixs_node *diff = ixs_sub(
      ctx, ixs_floor(ctx, ixs_div(ctx, A, D)),
      ixs_floor(ctx, ixs_div(ctx, ixs_add(ctx, A, ixs_int(ctx, 1)), D)));
  r = ixs_simplify(ctx, diff, &assume_k, 1);
  CHECK(r == ixs_int(ctx, 0));

  /* Negative test: floor(x/K) with x in [0,127], K in [1,...] does NOT
   * collapse to 0 (127/1 = 127, floor could be up to 127). */
  ixs_node *e3 = ixs_floor(ctx, ixs_div(ctx, x, K));
  r = ixs_simplify(ctx, e3, assumes, 3);
  CHECK(r != ixs_int(ctx, 0));
}

static void test_simplify_batch(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  ixs_node *exprs[3];
  exprs[0] = ixs_add(ctx, x, ixs_int(ctx, 0));
  exprs[1] = ixs_mul(ctx, ixs_int(ctx, 1), x);
  exprs[2] = ixs_floor(ctx, ixs_int(ctx, 7));

  ixs_node *assume = ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0));
  ixs_simplify_batch(ctx, exprs, 3, &assume, 1);

  CHECK(exprs[0] == x);
  CHECK(exprs[1] == x);
  CHECK(exprs[2] == ixs_int(ctx, 7));
}

static void test_print_c(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");

  /* floor -> ixs_floor_i */
  ixs_node *fl = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)));
  ixs_print_c(fl, buf, sizeof(buf));
  CHECK(strstr(buf, "ixs_floor_i") != NULL);

  /* Mod -> ixs_mod_i */
  ixs_node *m = ixs_mod(ctx, x, ixs_int(ctx, 8));
  ixs_print_c(m, buf, sizeof(buf));
  CHECK(strstr(buf, "ixs_mod_i") != NULL);

  /* xor -> infix ^ */
  ixs_node *xr = ixs_xor(ctx, x, y);
  ixs_print_c(xr, buf, sizeof(buf));
  CHECK(strstr(buf, " ^ ") != NULL);

  /* integer */
  ixs_print_c(ixs_int(ctx, 42), buf, sizeof(buf));
  CHECK(strcmp(buf, "42") == 0);
}

static void test_floor_drop_fractional_const(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *fl_x = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));

  /* floor(1/2 * floor(x/3) + 1/4)  ->  floor(1/2 * floor(x/3))
   * 1/4 < 1/2 = 1/lcm(denom) */
  ixs_node *a = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)),
                                       ixs_rat(ctx, 1, 4)));
  ixs_node *expected = ixs_floor(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)));
  CHECK(a == expected);

  /* floor(1/2 * floor(x/3) + 15/32) also drops (15/32 < 1/2) */
  ixs_node *b = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)),
                                       ixs_rat(ctx, 15, 32)));
  CHECK(b == expected);

  /* floor(1/2 * floor(x/3) + 1/2) does NOT drop (1/2 >= 1/2) */
  ixs_node *c = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)),
                                       ixs_rat(ctx, 1, 2)));
  CHECK(c != expected);

  /* Multi-term: floor(1/2*fl_x + 1/3*fl_y + 1/7)
   * lcm(2,3)=6, 1/7 < 1/6 => drop constant */
  ixs_node *fl_y = ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 5)));
  ixs_node *d =
      ixs_floor(ctx, ixs_add(ctx,
                             ixs_add(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)),
                                     ixs_div(ctx, fl_y, ixs_int(ctx, 3))),
                             ixs_rat(ctx, 1, 7)));
  ixs_node *d_exp =
      ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, fl_x, ixs_int(ctx, 2)),
                             ixs_div(ctx, fl_y, ixs_int(ctx, 3))));
  CHECK(d == d_exp);
}

static void test_round_extract_rat_split(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* floor(65/32 + 1/2*floor(x/16))  ->  2 + floor(1/32 + 1/2*floor(x/16))
   * then floor_drop_const fires on 1/32 < 1/2 => 2 + floor(1/2*floor(x/16))
   * i.e. overall result: 2 + floor(floor(x/16) / 2) */
  ixs_node *fl_x16 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 16)));
  ixs_node *a = ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, 65, 32),
                                       ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  ixs_node *expected_a =
      ixs_add(ctx, ixs_int(ctx, 2),
              ixs_floor(ctx, ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  CHECK(a == expected_a);

  /* floor(7/3 + floor(x/5) / 3) -> 2 + floor(1/3 + floor(x/5)/3)
   * 1/3 >= 1/3 so const is NOT dropped; verify integer part extracted. */
  ixs_node *fl_x5 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 5)));
  ixs_node *b = ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, 7, 3),
                                       ixs_div(ctx, fl_x5, ixs_int(ctx, 3))));
  /* Constant is 7/3=2+1/3. After split, integer 2 is extracted.
   * 1/3 is NOT dropped (1/3 >= 1/lcm where lcm=3).
   * Result: 2 + floor(1/3 + floor(x/5)/3) */
  ixs_node *inner =
      ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, 1, 3),
                             ixs_div(ctx, fl_x5, ixs_int(ctx, 3))));
  ixs_node *expected_b = ixs_add(ctx, ixs_int(ctx, 2), inner);
  CHECK(b == expected_b);

  /* floor(1/32 + floor(x/16) / 2) -> floor(floor(x/16) / 2)
   * RAT coeff with fl==0: no split needed, const drops directly. */
  ixs_node *c = ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, 1, 32),
                                       ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  ixs_node *expected_c = ixs_floor(ctx, ixs_div(ctx, fl_x16, ixs_int(ctx, 2)));
  CHECK(c == expected_c);

  /* Negative: floor(-63/32 + floor(x/16)/2).
   * -63/32 splits as fl=-2, rem=1/32.  1/32 < 1/2 => const drops.
   * Result: -2 + floor(floor(x/16)/2). */
  ixs_node *d = ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, -63, 32),
                                       ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  ixs_node *expected_d =
      ixs_add(ctx, ixs_int(ctx, -2),
              ixs_floor(ctx, ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  CHECK(d == expected_d);

  /* Negative: floor(-65/32 + floor(x/16)/2).
   * -65/32 splits as fl=-3, rem=31/32.  31/32 >= 1/2 => const NOT dropped.
   * Result: -3 + floor(31/32 + floor(x/16)/2). */
  ixs_node *e = ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, -65, 32),
                                       ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  ixs_node *inner_e =
      ixs_floor(ctx, ixs_add(ctx, ixs_rat(ctx, 31, 32),
                             ixs_div(ctx, fl_x16, ixs_int(ctx, 2))));
  ixs_node *expected_e = ixs_add(ctx, ixs_int(ctx, -3), inner_e);
  CHECK(e == expected_e);
}

static void test_floor_drop_const_sym(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *D = ixs_sym(ctx, "D");
  ixs_node *dinv = ixs_div(ctx, ixs_int(ctx, 1), D);

  /* floor(x*D^{-1} + (1/32)*D^{-1}):
   * Both terms share D^{-1}. base: x has scaled num=32. const: num=1.
   * g_bases=32, lcm=32, gcd(32,32)=32, 1%32=1>0 => drop constant. */
  ixs_node *xD = ixs_mul(ctx, x, dinv);
  ixs_node *cD = ixs_mul(ctx, ixs_rat(ctx, 1, 32), dinv);
  ixs_node *e1 = ixs_floor(ctx, ixs_add(ctx, xD, cD));
  ixs_node *assume_d = ixs_cmp(ctx, D, IXS_CMP_GE, ixs_int(ctx, 1));
  ixs_node *expected1 = ixs_floor(ctx, xD);
  ixs_node *r1 = ixs_simplify(ctx, e1, &assume_d, 1);
  CHECK(r1 == expected1);

  /* floor(x*D^{-1} + 1*D^{-1}): const scaled=32, gcd(32,32)=32,
   * 32%32=0 => no fire (the 1/D matters). */
  ixs_node *e2 = ixs_floor(ctx, ixs_add(ctx, xD, dinv));
  ixs_node *r2 = ixs_simplify(ctx, e2, &assume_d, 1);
  CHECK(r2 == e2);

  /* floor(32*x*D^{-1} + (5/32)*D^{-1}):
   * base: scaled num = 32*32=1024. const: 5.
   * g_bases=1024, lcm=32, gcd(1024,32)=32, 5%32=5>0 => drop. */
  ixs_node *x32D = ixs_mul(ctx, ixs_mul(ctx, ixs_int(ctx, 32), x), dinv);
  ixs_node *c5D = ixs_mul(ctx, ixs_rat(ctx, 5, 32), dinv);
  ixs_node *e3 = ixs_floor(ctx, ixs_add(ctx, x32D, c5D));
  ixs_node *r3 = ixs_simplify(ctx, e3, &assume_d, 1);
  ixs_node *expected3 = ixs_floor(ctx, x32D);
  CHECK(r3 == expected3);

  /* floor(x*D^{-1} + (17/32)*D^{-1}): 17%32=17>0 => drop. */
  ixs_node *c17D = ixs_mul(ctx, ixs_rat(ctx, 17, 32), dinv);
  ixs_node *e4 = ixs_floor(ctx, ixs_add(ctx, xD, c17D));
  ixs_node *r4 = ixs_simplify(ctx, e4, &assume_d, 1);
  CHECK(r4 == expected1);

  /* Partial: floor(x*D^{-1} + (33/32)*D^{-1}).
   * const=33, gcd=32, 33%32=1>0 => reduce to floor(x/D + 1/D). */
  ixs_node *c33D = ixs_mul(ctx, ixs_rat(ctx, 33, 32), dinv);
  ixs_node *e5 = ixs_floor(ctx, ixs_add(ctx, xD, c33D));
  ixs_node *r5 = ixs_simplify(ctx, e5, &assume_d, 1);
  CHECK(r5 == e2);

  /* Construction-time path (no ixs_simplify, bnds=NULL):
   * floor_drop_const_sym has needs_bounds=false, so it fires from
   * simp_floor directly.  Same pattern as e1 above. */
  {
    ixs_node *ct = ixs_floor(ctx, ixs_add(ctx, xD, cD));
    CHECK(ct == expected1);
  }
}

static void test_add_flatten_neg(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");

  /* (x + y) - (x + y) = 0: negated ADD must flatten */
  ixs_node *s = ixs_add(ctx, x, y);
  CHECK(ixs_sub(ctx, s, s) == ixs_int(ctx, 0));

  /* (2*x + 3*y) - (2*x + 3*y) = 0 */
  ixs_node *s2 = ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 2), x),
                         ixs_mul(ctx, ixs_int(ctx, 3), y));
  CHECK(ixs_sub(ctx, s2, s2) == ixs_int(ctx, 0));

  /* 2*(x + y) - (x + y) = x + y */
  CHECK(ixs_sub(ctx, ixs_mul(ctx, ixs_int(ctx, 2), s), s) == s);
}

static void test_floor_non_integer_min(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *r15 = ixs_rat(ctx, 1, 5);

  /* floor(-Min(1/5, x)) must NOT simplify to -Min(1/5, x).
   * At x=1: floor(-1/5) = -1, but -1/5 = -0.2.  */
  ixs_node *m = ixs_min(ctx, x, r15);
  ixs_node *neg_m = ixs_mul(ctx, ixs_int(ctx, -1), m);
  ixs_node *fl = ixs_floor(ctx, neg_m);
  ixs_node *s = ixs_simplify(ctx, fl, NULL, 0);
  CHECK(ixs_node_tag(s) == IXS_FLOOR);

  /* floor(Min(1/5, x)) also must not drop the floor. */
  ixs_node *fl2 = ixs_floor(ctx, m);
  ixs_node *s2 = ixs_simplify(ctx, fl2, NULL, 0);
  CHECK(ixs_node_tag(s2) == IXS_FLOOR);

  /* floor(-x) should still simplify to -x (x is integer). */
  ixs_node *neg_x = ixs_mul(ctx, ixs_int(ctx, -1), x);
  ixs_node *fl3 = ixs_floor(ctx, neg_x);
  ixs_node *s3 = ixs_simplify(ctx, fl3, NULL, 0);
  CHECK(s3 == neg_x);
}

static void test_modrem_congruence(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *r;

  /* Mod(x, 8) == 3  ⟹  Mod(x, 8) → 3 */
  ixs_node *cong_x_8_3[] = {
      ixs_cmp(ctx, ixs_mod(ctx, x, ixs_int(ctx, 8)), IXS_CMP_EQ,
              ixs_int(ctx, 3)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 8)), cong_x_8_3, 1);
  CHECK(ixs_node_int_val(r) == 3);

  /* Mod(x, 4) → 3 (since 8 % 4 == 0, remainder 3 % 4 == 3) */
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 4)), cong_x_8_3, 1);
  CHECK(ixs_node_int_val(r) == 3);

  /* Mod(x, 2) → 1 (since 8 % 2 == 0, remainder 3 % 2 == 1) */
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 2)), cong_x_8_3, 1);
  CHECK(ixs_node_int_val(r) == 1);

  /* Mod(x, 16) cannot be resolved (8 % 16 != 0) */
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 16)), cong_x_8_3, 1);
  CHECK(ixs_node_tag(r) == IXS_MOD);

  /* Mod(x, 3) cannot be resolved (8 % 3 != 0) */
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 3)), cong_x_8_3, 1);
  CHECK(ixs_node_tag(r) == IXS_MOD);

  /* Divisibility still works: Mod(x, 8) == 0 ⟹ Mod(x, 4) → 0 */
  ixs_node *div_x_8[] = {
      ixs_cmp(ctx, ixs_mod(ctx, x, ixs_int(ctx, 8)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 4)), div_x_8, 1);
  CHECK(ixs_node_int_val(r) == 0);

  /* floor(x/4) when x ≡ 0 (mod 8) still drops floor */
  r = ixs_simplify(ctx, ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4))),
                   div_x_8, 1);
  CHECK(strcmp(pr(r), "1/4*x") == 0);

  /* x ≡ 4 (mod 8): x is divisible by 4 (4%4==0) but NOT by 8 */
  ixs_node *cong_x_8_4[] = {
      ixs_cmp(ctx, ixs_mod(ctx, x, ixs_int(ctx, 8)), IXS_CMP_EQ,
              ixs_int(ctx, 4)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 4)), cong_x_8_4, 1);
  CHECK(ixs_node_int_val(r) == 0);
  r = ixs_simplify(ctx, ixs_mod(ctx, x, ixs_int(ctx, 8)), cong_x_8_4, 1);
  CHECK(ixs_node_int_val(r) == 4);
  r = ixs_simplify(ctx, ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4))),
                   cong_x_8_4, 1);
  CHECK(strcmp(pr(r), "1/4*x") == 0);

  /* floor(x/8) should NOT drop when x ≡ 4 (mod 8) */
  r = ixs_simplify(ctx, ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 8))),
                   cong_x_8_4, 1);
  CHECK(strstr(pr(r), "floor") != NULL);

  /* CRT merge: Mod(y, 4)==1 and Mod(y, 6)==3 ⟹ y ≡ 9 (mod 12) */
  ixs_node *crt_ys[] = {
      ixs_cmp(ctx, ixs_mod(ctx, y, ixs_int(ctx, 4)), IXS_CMP_EQ,
              ixs_int(ctx, 1)),
      ixs_cmp(ctx, ixs_mod(ctx, y, ixs_int(ctx, 6)), IXS_CMP_EQ,
              ixs_int(ctx, 3)),
  };
  r = ixs_simplify(ctx, ixs_mod(ctx, y, ixs_int(ctx, 12)), crt_ys, 2);
  CHECK(ixs_node_int_val(r) == 9);
  r = ixs_simplify(ctx, ixs_mod(ctx, y, ixs_int(ctx, 4)), crt_ys, 2);
  CHECK(ixs_node_int_val(r) == 1);
  r = ixs_simplify(ctx, ixs_mod(ctx, y, ixs_int(ctx, 2)), crt_ys, 2);
  CHECK(ixs_node_int_val(r) == 1);
}

static void test_subs_power_overflow(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* x*x with x=0 -> 0 (constant power folding, base case) */
  ixs_node *x2 = ixs_mul(ctx, x, x);
  ixs_node *r = ixs_subs(ctx, x2, x, ixs_int(ctx, 0));
  CHECK(r && ixs_node_int_val(r) == 0);

  /* x*x with x=3 -> 9 */
  r = ixs_subs(ctx, x2, x, ixs_int(ctx, 3));
  CHECK(r && ixs_node_int_val(r) == 9);

  /* x*x with x=3/2 -> 9/4 (rational base folding) */
  r = ixs_subs(ctx, x2, x, ixs_rat(ctx, 3, 2));
  CHECK(r && ixs_node_tag(r) == IXS_RAT);
  CHECK(ixs_node_rat_num(r) == 9 && ixs_node_rat_den(r) == 4);

  /* x*x with x=2^40: (2^40)^2 overflows int64, must not error */
  int64_t big = (int64_t)1 << 40;
  r = ixs_subs(ctx, x2, x, ixs_int(ctx, big));
  CHECK(r != NULL && !ixs_is_error(r));

  /* x*x*x with x=2^30: (2^30)^3 = 2^90, overflows int64 */
  ixs_node *x3 = ixs_mul(ctx, x2, x);
  r = ixs_subs(ctx, x3, x, ixs_int(ctx, (int64_t)1 << 30));
  CHECK(r != NULL && !ixs_is_error(r));

  /* x*x with x=2^31: fits i64, (2^31)^2 = 2^62 fits too */
  r = ixs_subs(ctx, x2, x, ixs_int(ctx, (int64_t)1 << 31));
  CHECK(r && ixs_node_int_val(r) == ((int64_t)1 << 62));

  /* (3/2)^3 via substitution */
  r = ixs_subs(ctx, x3, x, ixs_rat(ctx, 3, 2));
  CHECK(r && ixs_node_tag(r) == IXS_RAT);
  CHECK(ixs_node_rat_num(r) == 27 && ixs_node_rat_den(r) == 8);
}

/* m*floor(E/m) + Mod(E, m) = E: integer and symbolic moduli. */
static void test_floor_mod_cancel(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");

  /* 4*floor(x/4) + Mod(x, 4) -> x */
  ixs_node *e =
      ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 4),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)))),
              ixs_mod(ctx, x, ixs_int(ctx, 4)));
  CHECK(e == x);

  /* Scaled: 2*Mod(x, 4) + 8*floor(x/4) -> 2*x */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 2), ixs_mod(ctx, x, ixs_int(ctx, 4))),
              ixs_mul(ctx, ixs_int(ctx, 8),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)))));
  CHECK(e == ixs_mul(ctx, ixs_int(ctx, 2), x));

  /* Subtracted pair: -Mod(x, 4) - 4*floor(x/4) -> -x */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, -1), ixs_mod(ctx, x, ixs_int(ctx, 4))),
              ixs_mul(ctx, ixs_int(ctx, -4),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)))));
  CHECK(e == ixs_mul(ctx, ixs_int(ctx, -1), x));

  /* Extra terms: y + 4*floor(x/4) + Mod(x, 4) -> y + x */
  e = ixs_add(ctx, y,
              ixs_add(ctx,
                      ixs_mul(ctx, ixs_int(ctx, 4),
                              ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)))),
                      ixs_mod(ctx, x, ixs_int(ctx, 4))));
  CHECK(e == ixs_add(ctx, x, y));

  /* Nested: 2*floor(floor(x/3)/2) + Mod(floor(x/3), 2) -> floor(x/3) */
  {
    ixs_node *fx3 = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    e = ixs_add(ctx,
                ixs_mul(ctx, ixs_int(ctx, 2),
                        ixs_floor(ctx, ixs_div(ctx, fx3, ixs_int(ctx, 2)))),
                ixs_mod(ctx, fx3, ixs_int(ctx, 2)));
    CHECK(e == fx3);
  }

  /* No false match: 3*floor(x/4) + Mod(x, 4) should NOT cancel */
  e = ixs_add(ctx,
              ixs_mul(ctx, ixs_int(ctx, 3),
                      ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 4)))),
              ixs_mod(ctx, x, ixs_int(ctx, 4)));
  CHECK(ixs_node_tag(e) == IXS_ADD);

  /* No false match: Mod(x, 4) + 4*floor(y/4) - different lhs/arg */
  e = ixs_add(ctx, ixs_mod(ctx, x, ixs_int(ctx, 4)),
              ixs_mul(ctx, ixs_int(ctx, 4),
                      ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 4)))));
  CHECK(ixs_node_tag(e) == IXS_ADD);
}

/* Floor-Mod cancellation with symbolic modulus: m*floor(E/m) + Mod(E, m). */
static void test_floor_mod_cancel_symbolic(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *half_K = ixs_div(ctx, K, ixs_int(ctx, 2));

  /* K/2 * floor(x / (K/2)) + Mod(x, K/2) -> x */
  ixs_node *fl = ixs_floor(ctx, ixs_div(ctx, x, half_K));
  ixs_node *e = ixs_add(ctx, ixs_mul(ctx, half_K, fl), ixs_mod(ctx, x, half_K));
  CHECK(e == x);

  /* With constant offset: K/2*floor((x+5)/(K/2)) + Mod(x+5, K/2) -> x+5 */
  {
    ixs_node *x5 = ixs_add(ctx, x, ixs_int(ctx, 5));
    ixs_node *fl5 = ixs_floor(ctx, ixs_div(ctx, x5, half_K));
    e = ixs_add(ctx, ixs_mul(ctx, half_K, fl5), ixs_mod(ctx, x5, half_K));
    CHECK(e == x5);
  }

  /* Difference of two pairs:
   * K/2*floor((x+A)/(K/2)) + Mod(x+A, K/2)
   * - K/2*floor((x+B)/(K/2)) - Mod(x+B, K/2) -> A - B */
  {
    ixs_node *xA = ixs_add(ctx, x, ixs_int(ctx, 100));
    ixs_node *xB = ixs_add(ctx, x, ixs_int(ctx, 60));
    ixs_node *pair_A = ixs_add(
        ctx, ixs_mul(ctx, half_K, ixs_floor(ctx, ixs_div(ctx, xA, half_K))),
        ixs_mod(ctx, xA, half_K));
    ixs_node *pair_B = ixs_add(
        ctx, ixs_mul(ctx, half_K, ixs_floor(ctx, ixs_div(ctx, xB, half_K))),
        ixs_mod(ctx, xB, half_K));
    e = ixs_sub(ctx, pair_A, pair_B);
    CHECK(e && ixs_node_tag(e) == IXS_INT && ixs_node_int_val(e) == 40);
  }
}

static void test_floor_drop_const_divinfo(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *r;

  ixs_node *div_K_256[] = {
      ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 256)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };

  /* floor(7/8 + K/256) -> K/256 when 256 | K */
  {
    ixs_node *e = ixs_floor(
        ctx, ixs_add(ctx, ixs_div(ctx, ixs_int(ctx, 7), ixs_int(ctx, 8)),
                     ixs_div(ctx, K, ixs_int(ctx, 256))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strcmp(pr(r), "1/256*K") == 0);
  }

  /* floor(floor(K/32)/8 + 7/8) -> K/256 when 256 | K */
  {
    ixs_node *fk32 = ixs_floor(ctx, ixs_div(ctx, K, ixs_int(ctx, 32)));
    ixs_node *e =
        ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, fk32, ixs_int(ctx, 8)),
                               ixs_div(ctx, ixs_int(ctx, 7), ixs_int(ctx, 8))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strcmp(pr(r), "1/256*K") == 0);
  }

  /* floor(1/3 + K/6) -> K/6 when 6 | K */
  {
    ixs_node *div_K_6[] = {
        ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 6)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
    };
    ixs_node *e = ixs_floor(
        ctx, ixs_add(ctx, ixs_div(ctx, ixs_int(ctx, 1), ixs_int(ctx, 3)),
                     ixs_div(ctx, K, ixs_int(ctx, 6))));
    r = ixs_simplify(ctx, e, div_K_6, 1);
    CHECK(strcmp(pr(r), "1/6*K") == 0);
  }

  /* floor(1/2 + K/4) -> K/4 when 4|K: divisibility makes the grid
   * spacing 1 (not 1/4), so 1/2 < 1 and the constant drops. */
  {
    ixs_node *div_K_4[] = {
        ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 4)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
    };
    ixs_node *e = ixs_floor(
        ctx, ixs_add(ctx, ixs_div(ctx, ixs_int(ctx, 1), ixs_int(ctx, 2)),
                     ixs_div(ctx, K, ixs_int(ctx, 4))));
    r = ixs_simplify(ctx, e, div_K_4, 1);
    CHECK(strcmp(pr(r), "1/4*K") == 0);
  }

  /* floor(1/2 + K/3) -> K/3 when 3|K: same reasoning, different
   * modulus. Coefficient 1/3 has denom 3, absorbed by Mod(K,3)==0. */
  {
    ixs_node *div_K_3[] = {
        ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 3)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
    };
    ixs_node *e = ixs_floor(
        ctx, ixs_add(ctx, ixs_div(ctx, ixs_int(ctx, 1), ixs_int(ctx, 2)),
                     ixs_div(ctx, K, ixs_int(ctx, 3))));
    r = ixs_simplify(ctx, e, div_K_3, 1);
    CHECK(strcmp(pr(r), "1/3*K") == 0);
  }

  /* Negative: floor(1/2*x + 1/2*x*(x+1/2)) must NOT drop the floor
   * even with 2|x.  At x=2 the inner is 7/2, floor=3.
   * Regression: is_known_divisible declared x*(x+1/2) divisible by 2
   * because x is, without verifying (x+1/2) is integer-valued. */
  {
    ixs_node *x = ixs_sym(ctx, "x");
    ixs_node *div_x_2[] = {
        ixs_cmp(ctx, ixs_mod(ctx, x, ixs_int(ctx, 2)), IXS_CMP_EQ,
                ixs_int(ctx, 0)),
    };
    ixs_node *xph = ixs_add(ctx, x, ixs_rat(ctx, 1, 2));
    ixs_node *inner = ixs_add(ctx, x, ixs_mul(ctx, x, xph));
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, inner, ixs_int(ctx, 2)));
    r = ixs_simplify(ctx, e, div_x_2, 1);
    CHECK(strstr(pr(r), "floor(") != NULL);
  }
}

static void test_floor_extract_divinfo(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *r;

  ixs_node *div_K_256[] = {
      ixs_cmp(ctx, ixs_mod(ctx, K, ixs_int(ctx, 256)), IXS_CMP_EQ,
              ixs_int(ctx, 0)),
  };

  /* floor(x/3 + K/32) -> 1/32*K + floor(1/3*x) when 32|K.
   * The rational addend K/32 is integer per congruence; extract it. */
  {
    ixs_node *e = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)),
                                         ixs_div(ctx, K, ixs_int(ctx, 32))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "1/32*K") != NULL);
    CHECK(strstr(pr(r), "floor(1/3*x)") != NULL);
    CHECK(strstr(pr(r), "floor(1/3*x + ") == NULL);
  }

  /* Same expression without bounds: floor keeps both terms inside. */
  {
    ixs_node *e = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)),
                                         ixs_div(ctx, K, ixs_int(ctx, 32))));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strstr(pr(r), "floor(") != NULL);
    CHECK(strstr(pr(r), "1/32*K") != NULL);
    CHECK(strstr(pr(r), "1/3*x") != NULL);
  }

  /* floor(x/5 + K/2) -> 1/2*K + floor(1/5*x) when 2|K. */
  {
    ixs_node *e = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 5)),
                                         ixs_div(ctx, K, ixs_int(ctx, 2))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "1/2*K") != NULL);
    CHECK(strstr(pr(r), "floor(1/5*x)") != NULL);
  }

  /* Negative: floor(x/3 + K/257) stays fused -- 257 does not divide
   * K's known modulus 256. */
  {
    ixs_node *e = ixs_floor(ctx, ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)),
                                         ixs_div(ctx, K, ixs_int(ctx, 257))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "floor(") != NULL);
    CHECK(strstr(pr(r), "1/257*K") != NULL);
    CHECK(strstr(pr(r), "1/3*x") != NULL);
  }

  /* ceiling(x/3 + K/32) -> 1/32*K + ceiling(1/3*x) when 32|K.
   * Same path as floor; verify the ceiling branch works. */
  {
    ixs_node *e = ixs_ceil(ctx, ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)),
                                        ixs_div(ctx, K, ixs_int(ctx, 32))));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "1/32*K") != NULL);
    CHECK(strstr(pr(r), "ceiling(1/3*x)") != NULL);
  }

  /* MUL*ADD path: floor((K + x) / 32) -> 1/32*K + floor(1/32*x) when 32|K.
   * round_extract_mul_add distributes 1/32 into the ADD, then
   * round_extract_add (with bounds) extracts the integer 1/32*K term. */
  {
    ixs_node *sum = ixs_add(ctx, K, x);
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, sum, ixs_int(ctx, 32)));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "1/32*K") != NULL);
    CHECK(strstr(pr(r), "floor(1/32*x)") != NULL);
  }

  /* Same MUL*ADD without bounds: both terms stay inside floor. */
  {
    ixs_node *sum = ixs_add(ctx, K, x);
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, sum, ixs_int(ctx, 32)));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "floor(1/32*K + 1/32*x)") == 0);
  }

  /* MUL*ADD ceiling: ceiling((K + x) / 32) extracts K/32 when 32|K. */
  {
    ixs_node *sum = ixs_add(ctx, K, x);
    ixs_node *e = ixs_ceil(ctx, ixs_div(ctx, sum, ixs_int(ctx, 32)));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "1/32*K") != NULL);
    CHECK(strstr(pr(r), "ceiling(1/32*x)") != NULL);
  }

  /* Negative MUL*ADD: floor((K + x) / 257) stays fused -- 257 does
   * not divide K's known modulus 256. */
  {
    ixs_node *sum = ixs_add(ctx, K, x);
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, sum, ixs_int(ctx, 257)));
    r = ixs_simplify(ctx, e, div_K_256, 1);
    CHECK(strstr(pr(r), "floor(") != NULL);
    CHECK(strstr(pr(r), "1/257*K") != NULL);
    CHECK(strstr(pr(r), "1/257*x") != NULL);
  }
}

static void test_opposite_mul_add_cancel(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *pw = ixs_sym(ctx, "PW");
  ixs_node *e, *r;

  /* K*(PW + x) - K*(PW + 3) = K*(x - 3) */
  {
    ixs_node *a = ixs_mul(ctx, K, ixs_add(ctx, pw, x));
    ixs_node *b = ixs_mul(ctx, ixs_int(ctx, -1),
                          ixs_mul(ctx, K, ixs_add(ctx, pw, ixs_int(ctx, 3))));
    e = ixs_add(ctx, a, b);
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "K*(-3 + x)") == 0);
  }

  /* 2*K*(PW + x) - 2*K*(PW + x) = 0 */
  {
    ixs_node *t =
        ixs_mul(ctx, ixs_int(ctx, 2), ixs_mul(ctx, K, ixs_add(ctx, pw, x)));
    e = ixs_add(ctx, t, ixs_mul(ctx, ixs_int(ctx, -1), t));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(r == ixs_int(ctx, 0));
  }

  /* K*(PW + floor(a)) - K*(PW + floor(b)) = K*(floor(a) - floor(b)) */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *fa = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *fb = ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 3)));
    ixs_node *a = ixs_mul(ctx, K, ixs_add(ctx, pw, fa));
    ixs_node *b =
        ixs_mul(ctx, ixs_int(ctx, -1), ixs_mul(ctx, K, ixs_add(ctx, pw, fb)));
    e = ixs_add(ctx, a, b);
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "K*(floor(1/3*x) - floor(1/3*y))") == 0);
  }
}

/* Flatten MUL-over-ADD exposes floor terms for cancel_floor_mod_pairs.
 * K*(floor(A/m) - floor(B/m)) + c*Mod(A,m) - c*Mod(B,m)
 * distributes to K*floor(A/m) - K*floor(B/m) + Mod terms,
 * then floor-Mod identity fires: c*Mod(X,m) + c*m*floor(X/m) = c*X. */
static void test_flatten_mul_add_floor_mod(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *r;

  /* 4*Mod(x+192, K/128) - 4*Mod(x+64, K/128)
   * + K/32 * (floor((x+192)/(K/128)) - floor((x+64)/(K/128)))
   * = 4*(x+192) - 4*(x+64) = 512 */
  {
    ixs_node *m = ixs_div(ctx, K, ixs_int(ctx, 128));
    ixs_node *x64 = ixs_add(ctx, x, ixs_int(ctx, 64));
    ixs_node *x192 = ixs_add(ctx, x, ixs_int(ctx, 192));
    ixs_node *mod1 = ixs_mod(ctx, x64, m);
    ixs_node *mod2 = ixs_mod(ctx, x192, m);
    ixs_node *fl1 = ixs_floor(ctx, ixs_div(ctx, x64, m));
    ixs_node *fl2 = ixs_floor(ctx, ixs_div(ctx, x192, m));
    ixs_node *floor_diff =
        ixs_add(ctx, fl2, ixs_mul(ctx, ixs_int(ctx, -1), fl1));
    ixs_node *k32 = ixs_div(ctx, K, ixs_int(ctx, 32));
    ixs_node *e = ixs_add(ctx,
                          ixs_add(ctx, ixs_mul(ctx, ixs_int(ctx, 4), mod2),
                                  ixs_mul(ctx, ixs_int(ctx, -4), mod1)),
                          ixs_mul(ctx, k32, floor_diff));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(r == ixs_int(ctx, 512));
  }

  /* Same with a single pair: Mod(A, m) + m*floor(A/m) = A,
   * but floor is inside K*floor(A/K) and Mod modulus is K. */
  {
    ixs_node *m = K;
    ixs_node *fl = ixs_floor(ctx, ixs_div(ctx, x, m));
    ixs_node *e = ixs_add(ctx, ixs_mod(ctx, x, m), ixs_mul(ctx, m, fl));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(r == x);
  }

  /* Negative: no Mod present, distribution should NOT fire,
   * keeping the factored form. */
  {
    ixs_node *fa = ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 3)));
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *fb = ixs_floor(ctx, ixs_div(ctx, y, ixs_int(ctx, 3)));
    ixs_node *e =
        ixs_mul(ctx, K, ixs_add(ctx, fa, ixs_mul(ctx, ixs_int(ctx, -1), fb)));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "K*(floor(1/3*x) - floor(1/3*y))") == 0);
  }
}

static void test_round_unwrap_inner(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *K = ixs_sym(ctx, "K");
  ixs_node *r;

  /* floor(floor(x)/3) -> floor(x/3) */
  {
    ixs_node *e =
        ixs_floor(ctx, ixs_div(ctx, ixs_floor(ctx, x), ixs_int(ctx, 3)));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "floor(1/3*x)") == 0);
  }

  /* floor(floor(1/7*x + 1/7*K*y) / K) -> floor(1/7*x/K + 1/7*y)
   * Matches the corpus pattern: inner floor has non-integer coefficients
   * so round_extract_add doesn't split it; divisor is symbolic. */
  {
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *inner =
        ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 7)),
                ixs_div(ctx, ixs_mul(ctx, K, y), ixs_int(ctx, 7)));
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, ixs_floor(ctx, inner), K));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strstr(pr(r), "floor(floor(") == NULL);
  }

  /* Negative: floor(floor(x) * 3) must NOT unwrap (D=1/3, not integer). */
  {
    ixs_node *e = ixs_floor(
        ctx, ixs_mul(ctx, ixs_int(ctx, 3),
                     ixs_floor(ctx, ixs_div(ctx, x, ixs_int(ctx, 7)))));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strstr(pr(r), "floor(1/7*x)") != NULL);
  }

  /* Negative: floor(floor(x) / (-3)) must NOT unwrap (D negative). */
  {
    ixs_node *neg3 = ixs_mul(ctx, ixs_int(ctx, -1), ixs_int(ctx, 3));
    ixs_node *e = ixs_floor(ctx, ixs_div(ctx, ixs_floor(ctx, x), neg3));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strstr(pr(r), "floor(") != NULL);
  }

  /* ceiling(ceiling(x)/5) -> ceiling(x/5) */
  {
    ixs_node *e =
        ixs_ceil(ctx, ixs_div(ctx, ixs_ceil(ctx, x), ixs_int(ctx, 5)));
    r = ixs_simplify(ctx, e, NULL, 0);
    CHECK(strcmp(pr(r), "ceiling(1/5*x)") == 0);
  }
}

/* A|~A = True, A&~A = False, and CMP complement pairs. */
static void test_complement_annihilation(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *zero = ixs_int(ctx, 0);

  /* NOT complement: sym | NOT(sym) = True */
  CHECK(ixs_or(ctx, x, ixs_not(ctx, x)) == ixs_true(ctx));
  CHECK(ixs_or(ctx, ixs_not(ctx, x), x) == ixs_true(ctx));

  /* NOT complement: sym & NOT(sym) = False */
  CHECK(ixs_and(ctx, x, ixs_not(ctx, x)) == ixs_false(ctx));
  CHECK(ixs_and(ctx, ixs_not(ctx, x), x) == ixs_false(ctx));

  /* CMP complement: (x > 0) | (x <= 0) = True */
  {
    ixs_node *gt = ixs_cmp(ctx, x, IXS_CMP_GT, zero);
    ixs_node *le = ixs_cmp(ctx, x, IXS_CMP_LE, zero);
    CHECK(ixs_or(ctx, gt, le) == ixs_true(ctx));
    CHECK(ixs_and(ctx, gt, le) == ixs_false(ctx));
  }

  /* CMP complement: (x == y) | (x != y) = True */
  {
    ixs_node *eq = ixs_cmp(ctx, x, IXS_CMP_EQ, y);
    ixs_node *ne = ixs_cmp(ctx, x, IXS_CMP_NE, y);
    CHECK(ixs_or(ctx, eq, ne) == ixs_true(ctx));
    CHECK(ixs_and(ctx, eq, ne) == ixs_false(ctx));
  }

  /* Piecewise((0, c), (0, ~c)) collapses to 0. */
  {
    ixs_node *c = ixs_cmp(ctx, x, IXS_CMP_GT, zero);
    ixs_node *nc = ixs_not(ctx, c);
    ixs_node *vals[] = {zero, zero};
    ixs_node *conds[] = {c, nc};
    ixs_node *pw = ixs_pw(ctx, 2, vals, conds);
    CHECK(pw == zero);
  }

  /* Negative: (x > 0) | (y <= 0) is NOT True (different operands). */
  {
    ixs_node *a = ixs_cmp(ctx, x, IXS_CMP_GT, zero);
    ixs_node *b = ixs_cmp(ctx, y, IXS_CMP_LE, zero);
    CHECK(ixs_or(ctx, a, b) != ixs_true(ctx));
  }
}

static void test_eq_substitution(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *bm = ixs_sym(ctx, "BLOCK_M");
  ixs_node *x = ixs_sym(ctx, "x");

  /* BLOCK_M == 256 => BLOCK_M + x becomes 256 + x */
  {
    ixs_node *assumptions[] = {ixs_cmp(ctx, bm, IXS_CMP_EQ, ixs_int(ctx, 256))};
    ixs_node *expr = ixs_add(ctx, bm, x);
    ixs_node *result = ixs_simplify(ctx, expr, assumptions, 1);
    ixs_node *expected =
        ixs_simplify(ctx, ixs_add(ctx, ixs_int(ctx, 256), x), NULL, 0);
    CHECK(result == expected);
  }

  /* Derived equality: x >= 5 && x <= 5 => x replaced by 5 */
  {
    ixs_node *assumptions[] = {
        ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 5)),
        ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 5)),
    };
    ixs_node *expr = ixs_add(ctx, x, ixs_int(ctx, 1));
    ixs_node *result = ixs_simplify(ctx, expr, assumptions, 2);
    CHECK(result == ixs_int(ctx, 6));
  }

  /* ceil(M / 256) with M == 256 collapses to 1 */
  {
    ixs_node *assumptions[] = {ixs_cmp(ctx, bm, IXS_CMP_EQ, ixs_int(ctx, 256))};
    ixs_node *expr = ixs_ceil(ctx, ixs_mul(ctx, bm, ixs_rat(ctx, 1, 256)));
    ixs_node *result = ixs_simplify(ctx, expr, assumptions, 1);
    CHECK(result == ixs_int(ctx, 1));
  }

  /* Negative: no equality => symbol stays */
  {
    ixs_node *assumptions[] = {ixs_cmp(ctx, bm, IXS_CMP_GE, ixs_int(ctx, 1))};
    ixs_node *expr = ixs_add(ctx, bm, ixs_int(ctx, 0));
    ixs_node *result = ixs_simplify(ctx, expr, assumptions, 1);
    CHECK(result == bm);
  }

  /* Batch: equality substitution applies to all exprs in batch */
  {
    ixs_node *assumptions[] = {ixs_cmp(ctx, bm, IXS_CMP_EQ, ixs_int(ctx, 256))};
    ixs_node *exprs[] = {
        ixs_add(ctx, bm, ixs_int(ctx, 1)),
        ixs_mul(ctx, bm, ixs_int(ctx, 2)),
    };
    ixs_simplify_batch(ctx, exprs, 2, assumptions, 1);
    CHECK(exprs[0] == ixs_int(ctx, 257));
    CHECK(exprs[1] == ixs_int(ctx, 512));
  }
}

/* Piecewise branches fork bounds; equality substitution should fire
 * independently per branch with each branch's augmented bounds. */
static void test_pw_branch_eq_substitution(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* Piecewise((x + 1, x == 5), (x + 2, True))
   * Branch 1 learns x == 5 => x + 1 becomes 6.
   * Default branch: x stays symbolic => x + 2 unchanged. */
  {
    ixs_node *vals[] = {ixs_add(ctx, x, ixs_int(ctx, 1)),
                        ixs_add(ctx, x, ixs_int(ctx, 2))};
    ixs_node *cds[] = {ixs_cmp(ctx, x, IXS_CMP_EQ, ixs_int(ctx, 5)),
                       ixs_true(ctx)};
    ixs_node *pw = ixs_pw(ctx, 2, vals, cds);
    ixs_node *result = ixs_simplify(ctx, pw, NULL, 0);
    char buf[256];
    ixs_print(result, buf, sizeof(buf));
    CHECK(strcmp(buf, "Piecewise((6, -5 + x == 0), (2 + x, True))") == 0);
  }

  /* Two guarded branches with different equalities and distinct values:
   * Piecewise((x + 10, x == 3), (x + 20, x == 7), (x + 30, True))
   * Branch 1: x == 3 => 13, Branch 2: x == 7 => 27, default: x + 30. */
  {
    ixs_node *vals[] = {ixs_add(ctx, x, ixs_int(ctx, 10)),
                        ixs_add(ctx, x, ixs_int(ctx, 20)),
                        ixs_add(ctx, x, ixs_int(ctx, 30))};
    ixs_node *cds[] = {ixs_cmp(ctx, x, IXS_CMP_EQ, ixs_int(ctx, 3)),
                       ixs_cmp(ctx, x, IXS_CMP_EQ, ixs_int(ctx, 7)),
                       ixs_true(ctx)};
    ixs_node *pw = ixs_pw(ctx, 3, vals, cds);
    ixs_node *result = ixs_simplify(ctx, pw, NULL, 0);
    char buf[256];
    ixs_print(result, buf, sizeof(buf));
    CHECK(strcmp(buf, "Piecewise((13, -3 + x == 0), "
                      "(27, -7 + x == 0), (30 + x, True))") == 0);
  }
}

/* Inside a Piecewise branch whose condition implies x - y > 0,
 * Max(x - y, 1) should collapse to x - y. */
static void test_pw_max_bounds_collapse(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *one = ixs_int(ctx, 1);
  char buf[512];

  /* Piecewise((Max(x - y, 1), y - x < 0), (42, True))
   * with x >= 1, y >= 1.
   * Branch condition y - x < 0  =>  x - y > 0  =>  x - y >= 1.
   * So Max(x - y, 1) collapses to x - y. */
  {
    ixs_node *diff = ixs_add(ctx, x, ixs_mul(ctx, ixs_int(ctx, -1), y));
    ixs_node *neg_diff = ixs_add(ctx, y, ixs_mul(ctx, ixs_int(ctx, -1), x));
    ixs_node *cond = ixs_cmp(ctx, neg_diff, IXS_CMP_LT, ixs_int(ctx, 0));
    ixs_node *vals[] = {ixs_max(ctx, diff, one), ixs_int(ctx, 42)};
    ixs_node *cds[] = {cond, ixs_true(ctx)};
    ixs_node *pw = ixs_pw(ctx, 2, vals, cds);
    ixs_node *assumptions[] = {
        ixs_cmp(ctx, ixs_add(ctx, x, ixs_int(ctx, -1)), IXS_CMP_GE,
                ixs_int(ctx, 0)),
        ixs_cmp(ctx, ixs_add(ctx, y, ixs_int(ctx, -1)), IXS_CMP_GE,
                ixs_int(ctx, 0)),
    };
    ixs_node *result = ixs_simplify(ctx, pw, assumptions, 2);
    ixs_print(result, buf, sizeof(buf));
    CHECK(strstr(buf, "Max(") == NULL);
  }

  /* Standalone Max (no branch guard) — bounds alone don't prove x > y. */
  {
    ixs_node *diff = ixs_add(ctx, x, ixs_mul(ctx, ixs_int(ctx, -1), y));
    ixs_node *maxn = ixs_max(ctx, diff, one);
    ixs_node *assumptions[] = {
        ixs_cmp(ctx, ixs_add(ctx, x, ixs_int(ctx, -1)), IXS_CMP_GE,
                ixs_int(ctx, 0)),
        ixs_cmp(ctx, ixs_add(ctx, y, ixs_int(ctx, -1)), IXS_CMP_GE,
                ixs_int(ctx, 0)),
    };
    ixs_node *result = ixs_simplify(ctx, maxn, assumptions, 2);
    ixs_print(result, buf, sizeof(buf));
    CHECK(strstr(buf, "Max(") != NULL);
  }

  /* LE variant: condition y - x <= 0 => x - y >= 0.
   * Max(x - y, 0) should collapse to x - y. */
  {
    ixs_node *diff = ixs_add(ctx, x, ixs_mul(ctx, ixs_int(ctx, -1), y));
    ixs_node *neg_diff = ixs_add(ctx, y, ixs_mul(ctx, ixs_int(ctx, -1), x));
    ixs_node *cond = ixs_cmp(ctx, neg_diff, IXS_CMP_LE, ixs_int(ctx, 0));
    ixs_node *vals[] = {ixs_max(ctx, diff, ixs_int(ctx, 0)), ixs_int(ctx, 99)};
    ixs_node *cds[] = {cond, ixs_true(ctx)};
    ixs_node *pw = ixs_pw(ctx, 2, vals, cds);
    ixs_node *assumptions[] = {
        ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
        ixs_cmp(ctx, y, IXS_CMP_GE, ixs_int(ctx, 0)),
    };
    ixs_node *result = ixs_simplify(ctx, pw, assumptions, 2);
    ixs_print(result, buf, sizeof(buf));
    CHECK(strstr(buf, "Max(") == NULL);
  }
}

static void test_ceil_collapse_and_unwrap(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *r;

  /* ceil_collapse: ceil(x/32) with 1 <= x <= 32 -> 1
   * x/32 in [1/32, 1], ceil of both endpoints is 1. */
  ixs_node *a_ceil[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 1)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 32)),
  };
  r = ixs_simplify(ctx, ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 32))),
                   a_ceil, 2);
  CHECK(r && ixs_node_int_val(r) == 1);

  /* negative: ceil(x/32) with 1 <= x <= 64 should NOT collapse
   * (ceil(1/32)=1, ceil(64/32)=2). */
  ixs_node *a_wide[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 1)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 64)),
  };
  r = ixs_simplify(ctx, ixs_ceil(ctx, ixs_div(ctx, x, ixs_int(ctx, 32))),
                   a_wide, 2);
  CHECK(r && ixs_node_tag(r) != IXS_INT);

  /* ceil_unwrap_inner: ceil(ceil(A) / K) -> ceil(A / K)
   * Symbolic divisor forces nfactors > 1, bypassing round_pull_in_denom. */
  {
    ixs_node *K = ixs_sym(ctx, "K");
    ixs_node *y = ixs_sym(ctx, "y");
    ixs_node *inner_arg =
        ixs_add(ctx, ixs_div(ctx, x, ixs_int(ctx, 7)),
                ixs_div(ctx, ixs_mul(ctx, K, y), ixs_int(ctx, 7)));
    ixs_node *expr = ixs_ceil(ctx, ixs_div(ctx, ixs_ceil(ctx, inner_arg), K));
    r = ixs_simplify(ctx, expr, NULL, 0);
    CHECK(strstr(pr(r), "ceiling(ceiling(") == NULL);
  }
}

static void test_max_min_const_fold(void) {
  ixs_ctx *ctx = get_ctx();

  /* max_const_fold: Max(3, 7) -> 7 */
  CHECK(ixs_max(ctx, ixs_int(ctx, 3), ixs_int(ctx, 7)) == ixs_int(ctx, 7));

  /* Max(7, 3) -> 7 (reversed order) */
  CHECK(ixs_max(ctx, ixs_int(ctx, 7), ixs_int(ctx, 3)) == ixs_int(ctx, 7));

  /* Max with rationals: Max(1/3, 1/2) -> 1/2 */
  CHECK(ixs_max(ctx, ixs_rat(ctx, 1, 3), ixs_rat(ctx, 1, 2)) ==
        ixs_rat(ctx, 1, 2));

  /* min_const_fold: Min(3, 7) -> 3 */
  CHECK(ixs_min(ctx, ixs_int(ctx, 3), ixs_int(ctx, 7)) == ixs_int(ctx, 3));

  /* Min(7, 3) -> 3 */
  CHECK(ixs_min(ctx, ixs_int(ctx, 7), ixs_int(ctx, 3)) == ixs_int(ctx, 3));

  /* Min with rationals: Min(1/3, 1/2) -> 1/3 */
  CHECK(ixs_min(ctx, ixs_rat(ctx, 1, 3), ixs_rat(ctx, 1, 2)) ==
        ixs_rat(ctx, 1, 3));
}

static void test_min_bounds_collapse(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");

  /* Min(x, y) with 0 <= x <= 5 and 10 <= y <= 20 -> x
   * (x.hi=5 <= y.lo=10, so x is always smaller) */
  ixs_node *assumes[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 5)),
      ixs_cmp(ctx, y, IXS_CMP_GE, ixs_int(ctx, 10)),
      ixs_cmp(ctx, y, IXS_CMP_LE, ixs_int(ctx, 20)),
  };
  ixs_node *r = ixs_simplify(ctx, ixs_min(ctx, x, y), assumes, 4);
  CHECK(r == x);

  /* negative: overlapping ranges should NOT collapse */
  ixs_node *overlap[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 0)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 10)),
      ixs_cmp(ctx, y, IXS_CMP_GE, ixs_int(ctx, 5)),
      ixs_cmp(ctx, y, IXS_CMP_LE, ixs_int(ctx, 20)),
  };
  r = ixs_simplify(ctx, ixs_min(ctx, x, y), overlap, 4);
  CHECK(r && ixs_node_tag(r) != IXS_SYM);
}

static void test_cmp_const_fold(void) {
  ixs_ctx *ctx = get_ctx();

  /* 5 > 3 -> True */
  CHECK(ixs_cmp(ctx, ixs_int(ctx, 5), IXS_CMP_GT, ixs_int(ctx, 3)) ==
        ixs_true(ctx));

  /* 3 > 5 -> False */
  CHECK(ixs_cmp(ctx, ixs_int(ctx, 3), IXS_CMP_GT, ixs_int(ctx, 5)) ==
        ixs_false(ctx));

  /* 5 == 5 -> True */
  CHECK(ixs_cmp(ctx, ixs_int(ctx, 5), IXS_CMP_EQ, ixs_int(ctx, 5)) ==
        ixs_true(ctx));

  /* 5 != 5 -> False */
  CHECK(ixs_cmp(ctx, ixs_int(ctx, 5), IXS_CMP_NE, ixs_int(ctx, 5)) ==
        ixs_false(ctx));

  /* 1/3 < 1/2 -> True */
  CHECK(ixs_cmp(ctx, ixs_rat(ctx, 1, 3), IXS_CMP_LT, ixs_rat(ctx, 1, 2)) ==
        ixs_true(ctx));

  /* 1/2 <= 1/3 -> False */
  CHECK(ixs_cmp(ctx, ixs_rat(ctx, 1, 2), IXS_CMP_LE, ixs_rat(ctx, 1, 3)) ==
        ixs_false(ctx));
}

static void test_cmp_identity(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* x >= x -> True */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_GE, x) == ixs_true(ctx));

  /* x <= x -> True */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_LE, x) == ixs_true(ctx));

  /* x == x -> True */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_EQ, x) == ixs_true(ctx));

  /* x > x -> False */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_GT, x) == ixs_false(ctx));

  /* x < x -> False */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_LT, x) == ixs_false(ctx));

  /* x != x -> False */
  CHECK(ixs_cmp(ctx, x, IXS_CMP_NE, x) == ixs_false(ctx));
}

static void test_cmp_bounds_resolve(void) {
  ixs_ctx *ctx = get_ctx();
  ixs_node *x = ixs_sym(ctx, "x");

  /* x >= 5 with 10 <= x <= 20 -> True */
  ixs_node *assumes[] = {
      ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 10)),
      ixs_cmp(ctx, x, IXS_CMP_LE, ixs_int(ctx, 20)),
  };
  ixs_node *ge5 = ixs_cmp(ctx, x, IXS_CMP_GE, ixs_int(ctx, 5));
  CHECK(ixs_simplify(ctx, ge5, assumes, 2) == ixs_true(ctx));

  /* x < 5 with 10 <= x <= 20 -> False */
  ixs_node *lt5 = ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 5));
  CHECK(ixs_simplify(ctx, lt5, assumes, 2) == ixs_false(ctx));

  /* negative: x > 15 with 10 <= x <= 20 is indeterminate */
  ixs_node *gt15 = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 15));
  ixs_node *r = ixs_simplify(ctx, gt15, assumes, 2);
  CHECK(r != ixs_true(ctx) && r != ixs_false(ctx));
}

int main(void) {
  test_add_canonicalize();
  test_mul_canonicalize();
  test_hash_consing();
  test_floor_rules();
  test_mod_rules();
  test_boolean();
  test_simplify_with_bounds();
  test_eq_substitution();
  test_pw_branch_eq_substitution();
  test_floor_bounds_collapse();
  test_mod_bounds_tighten();
  test_mod_extract_constant();
  test_floor_drop_small_rational();
  test_substitution();
  test_subs_multi();
  test_sentinel_propagation();
  test_nested_floor_ceil();
  test_same_node();
  test_print_roundtrip();
  test_divisibility_assumptions();
  test_large_expressions();
  test_bounds_many_vars();
  test_mod_floor_regression();
  test_mod_recognition();
  test_floor_mod_divisor();
  test_pw_fold_in_add();
  test_piecewise_branch_bounds();
  test_product_zero_branch_collapse();
  test_pw_max_bounds_collapse();
  test_floor_symbolic_denom();
  test_simplify_batch();
  test_print_c();
  test_floor_drop_fractional_const();
  test_round_extract_rat_split();
  test_floor_drop_const_sym();
  test_floor_non_integer_min();
  test_add_flatten_neg();
  test_modrem_congruence();
  test_subs_power_overflow();
  test_floor_mod_cancel();
  test_floor_mod_cancel_symbolic();
  test_floor_drop_const_divinfo();
  test_floor_extract_divinfo();
  test_opposite_mul_add_cancel();
  test_flatten_mul_add_floor_mod();
  test_round_unwrap_inner();
  test_complement_annihilation();
  test_ceil_collapse_and_unwrap();
  test_max_min_const_fold();
  test_min_bounds_collapse();
  test_cmp_const_fold();
  test_cmp_identity();
  test_cmp_bounds_resolve();

  printf("test_simplify: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
