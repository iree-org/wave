/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <ixsimpl.h>
#include <string.h>

#include "test_check.h"

static char buf[4096];

static const char *pr(ixs_node *n) {
  ixs_print(n, buf, sizeof(buf));
  return buf;
}

static void test_expand_leaves(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *n5 = ixs_int(ctx, 5);

  CHECK(ixs_expand(ctx, x) == x);
  CHECK(ixs_expand(ctx, n5) == n5);
  CHECK(ixs_expand(ctx, NULL) == NULL);
  ixs_ctx_destroy(ctx);
}

static void test_expand_add_noop(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *y = ixs_sym(ctx, "y");
  ixs_node *sum = ixs_add(ctx, x, y);

  CHECK(ixs_expand(ctx, sum) == sum);
  ixs_ctx_destroy(ctx);
}

static void test_expand_const_times_add(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");

  /* 2*(a+b) -> 2*a + 2*b */
  ixs_node *expr = ixs_mul(ctx, ixs_int(ctx, 2), ixs_add(ctx, a, b));
  ixs_node *r = ixs_expand(ctx, expr);
  CHECK(strcmp(pr(r), "2*a + 2*b") == 0);
  ixs_ctx_destroy(ctx);
}

static void test_expand_two_sums(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *c = ixs_sym(ctx, "c");
  ixs_node *d = ixs_sym(ctx, "d");

  /* (a+b)*(c+d) -> canonical: a*d + a*c + b*c + b*d (SYM factors sorted) */
  ixs_node *expr = ixs_mul(ctx, ixs_add(ctx, a, b), ixs_add(ctx, c, d));
  ixs_node *r = ixs_expand(ctx, expr);
  const char *s = pr(r);
  CHECK(strstr(s, "a*d") != NULL);
  CHECK(strstr(s, "a*c") != NULL);
  CHECK(strstr(s, "b*c") != NULL);
  CHECK(strstr(s, "b*d") != NULL);
  ixs_ctx_destroy(ctx);
}

static void test_expand_sym_times_add(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");

  /* x*(a+b) -> a*x + b*x */
  ixs_node *expr = ixs_mul(ctx, x, ixs_add(ctx, a, b));
  ixs_node *r = ixs_expand(ctx, expr);
  const char *s = pr(r);
  CHECK(strstr(s, "a*x") != NULL);
  CHECK(strstr(s, "b*x") != NULL);
  CHECK(strstr(s, "+") != NULL);
  ixs_ctx_destroy(ctx);
}

static void test_expand_nested_add(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");

  /* 2*(a+b) + 3*(a+b) in an ADD should expand to 5*a + 5*b */
  ixs_node *e1 = ixs_mul(ctx, ixs_int(ctx, 2), ixs_add(ctx, a, b));
  ixs_node *e2 = ixs_mul(ctx, ixs_int(ctx, 3), ixs_add(ctx, a, b));
  ixs_node *expr = ixs_add(ctx, e1, e2);
  ixs_node *r = ixs_expand(ctx, expr);
  CHECK(strcmp(pr(r), "5*a + 5*b") == 0);
  ixs_ctx_destroy(ctx);
}

static void test_expand_inside_floor(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");

  /* floor(2*(a+b)) -> 2*a + 2*b (integer-valued, floor drops) */
  ixs_node *inner = ixs_mul(ctx, ixs_int(ctx, 2), ixs_add(ctx, a, b));
  ixs_node *expr = ixs_floor(ctx, inner);
  ixs_node *r = ixs_expand(ctx, expr);
  CHECK(strcmp(pr(r), "2*a + 2*b") == 0);
  ixs_ctx_destroy(ctx);
}

static void test_expand_already_expanded(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *c = ixs_sym(ctx, "c");

  /* a*b + c is already expanded */
  ixs_node *expr = ixs_add(ctx, ixs_mul(ctx, a, b), c);
  CHECK(ixs_expand(ctx, expr) == expr);
  ixs_ctx_destroy(ctx);
}

static void test_expand_three_factors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *c = ixs_sym(ctx, "c");

  /* (a+b)*c*2 -> 2*a*c + 2*b*c */
  ixs_node *expr =
      ixs_mul(ctx, ixs_mul(ctx, ixs_add(ctx, a, b), c), ixs_int(ctx, 2));
  ixs_node *r = ixs_expand(ctx, expr);
  const char *s = pr(r);
  CHECK(strstr(s, "a*c") != NULL);
  CHECK(strstr(s, "b*c") != NULL);
  CHECK(strstr(s, "+") != NULL);
  ixs_ctx_destroy(ctx);
}

static void test_expand_piecewise(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *x = ixs_sym(ctx, "x");

  /* Piecewise((2*(a+b), x > 0), (0, True)) -> Piecewise((2*a+2*b, ...)) */
  ixs_node *vals[2];
  ixs_node *conds[2];
  vals[0] = ixs_mul(ctx, ixs_int(ctx, 2), ixs_add(ctx, a, b));
  conds[0] = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  vals[1] = ixs_int(ctx, 0);
  conds[1] = ixs_true(ctx);
  ixs_node *pw = ixs_pw(ctx, 2, vals, conds);
  ixs_node *r = ixs_expand(ctx, pw);
  const char *s = pr(r);
  CHECK(strstr(s, "2*a + 2*b") != NULL);
  CHECK(strstr(s, "Piecewise") != NULL);
  ixs_ctx_destroy(ctx);
}

static void test_expand_sentinel(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *err = ixs_parse(ctx, "???", 3);
  CHECK(ixs_is_error(err));
  CHECK(ixs_expand(ctx, err) == err);
  ixs_ctx_destroy(ctx);
}

int main(void) {
  test_expand_leaves();
  test_expand_add_noop();
  test_expand_const_times_add();
  test_expand_two_sums();
  test_expand_sym_times_add();
  test_expand_nested_add();
  test_expand_inside_floor();
  test_expand_already_expanded();
  test_expand_three_factors();
  test_expand_piecewise();
  test_expand_sentinel();

  printf("test_expand: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
