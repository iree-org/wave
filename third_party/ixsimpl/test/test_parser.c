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

static void test_integers(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  n = ixs_parse(ctx, "42", 2);
  CHECK(n && !ixs_is_error(n));
  CHECK(ixs_node_tag(n) == IXS_INT && ixs_node_int_val(n) == 42);

  n = ixs_parse(ctx, "0", 1);
  CHECK(n && ixs_node_int_val(n) == 0);

  ixs_ctx_destroy(ctx);
}

static void test_symbols(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  n = ixs_parse(ctx, "$T0", 3);
  CHECK(n && !ixs_is_error(n));
  CHECK(ixs_node_tag(n) == IXS_SYM);
  CHECK(strcmp(pr(n), "$T0") == 0);

  n = ixs_parse(ctx, "_M_div_32", 9);
  CHECK(n && strcmp(pr(n), "_M_div_32") == 0);

  ixs_ctx_destroy(ctx);
}

static void test_arithmetic(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  /* 3 + 4 → 7 */
  n = ixs_parse(ctx, "3 + 4", 5);
  CHECK(n && ixs_node_tag(n) == IXS_INT && ixs_node_int_val(n) == 7);

  /* 3 * 4 → 12 */
  n = ixs_parse(ctx, "3 * 4", 5);
  CHECK(n && ixs_node_int_val(n) == 12);

  /* 7 / 2 → 7/2 (rational) */
  n = ixs_parse(ctx, "7/2", 3);
  CHECK(n && ixs_node_tag(n) == IXS_RAT);

  /* x + x → 2*x */
  n = ixs_parse(ctx, "x + x", 5);
  CHECK(n && !ixs_is_error(n));

  /* 3*x + 2*x → 5*x */
  n = ixs_parse(ctx, "3*x + 2*x", 9);
  CHECK(n && !ixs_is_error(n));
  CHECK(strcmp(pr(n), "5*x") == 0);

  ixs_ctx_destroy(ctx);
}

static void test_floor_ceil(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  /* floor(7/2) → 3 */
  n = ixs_parse(ctx, "floor(7/2)", 10);
  CHECK(n && ixs_node_tag(n) == IXS_INT && ixs_node_int_val(n) == 3);

  /* ceiling(7/2) → 4 */
  n = ixs_parse(ctx, "ceiling(7/2)", 12);
  CHECK(n && ixs_node_tag(n) == IXS_INT && ixs_node_int_val(n) == 4);

  /* floor(x) → x (x is integer-valued) */
  n = ixs_parse(ctx, "floor(x)", 8);
  CHECK(n && ixs_node_tag(n) == IXS_SYM);

  /* floor(floor(x)) → x */
  n = ixs_parse(ctx, "floor(floor(x))", 15);
  CHECK(n && ixs_node_tag(n) == IXS_SYM);

  ixs_ctx_destroy(ctx);
}

static void test_mod(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  /* Mod(17, 5) → 2 */
  n = ixs_parse(ctx, "Mod(17, 5)", 10);
  CHECK(n && ixs_node_int_val(n) == 2);

  /* Mod(floor(x), 1) → 0 (only integer-valued args fold) */
  n = ixs_parse(ctx, "Mod(floor(x), 1)", 16);
  CHECK(n && ixs_node_int_val(n) == 0);

  ixs_ctx_destroy(ctx);
}

static void test_max_min_xor(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  n = ixs_parse(ctx, "Max(3, 7)", 9);
  CHECK(n && ixs_node_int_val(n) == 7);

  n = ixs_parse(ctx, "Min(3, 7)", 9);
  CHECK(n && ixs_node_int_val(n) == 3);

  n = ixs_parse(ctx, "xor(5, 3)", 9);
  CHECK(n && ixs_node_int_val(n) == 6);

  n = ixs_parse(ctx, "xor(x, x)", 9);
  CHECK(n && ixs_node_int_val(n) == 0);

  ixs_ctx_destroy(ctx);
}

static void test_piecewise(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  /* Single True branch → value */
  n = ixs_parse(ctx, "Piecewise((42, True))", 21);
  CHECK(n && ixs_node_int_val(n) == 42);

  /* False branch dropped */
  n = ixs_parse(ctx, "Piecewise((1, False), (2, True))", 32);
  CHECK(n && ixs_node_int_val(n) == 2);

  ixs_ctx_destroy(ctx);
}

static void test_comparisons(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  n = ixs_parse(ctx, "Piecewise((1, 3 > 2), (0, True))", 32);
  CHECK(n && ixs_node_int_val(n) == 1);

  n = ixs_parse(ctx, "Piecewise((1, 1 > 2), (0, True))", 32);
  CHECK(n && ixs_node_int_val(n) == 0);

  ixs_ctx_destroy(ctx);
}

static void test_errors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  /* Division by zero */
  n = ixs_parse(ctx, "1/0", 3);
  CHECK(n && ixs_is_domain_error(n));
  CHECK(ixs_ctx_nerrors(ctx) > 0);
  ixs_ctx_clear_errors(ctx);

  /* Mod by zero */
  n = ixs_parse(ctx, "Mod(x, 0)", 9);
  CHECK(n && ixs_is_domain_error(n));
  ixs_ctx_clear_errors(ctx);

  /* Parse error: trailing chars */
  n = ixs_parse(ctx, "x y", 3);
  CHECK(n && ixs_is_parse_error(n));
  ixs_ctx_clear_errors(ctx);

  ixs_ctx_destroy(ctx);
}

static void test_complex_expr(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  const char *expr = "128*floor($T0/64) + 4*floor(Mod($T0, 64)/16)";
  n = ixs_parse(ctx, expr, strlen(expr));
  CHECK(n && !ixs_is_error(n));

  ixs_ctx_destroy(ctx);
}

static void test_negation(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;

  n = ixs_parse(ctx, "-x", 2);
  CHECK(n && !ixs_is_error(n));
  CHECK(strcmp(pr(n), "-x") == 0);

  n = ixs_parse(ctx, "-(x + y)", 8);
  CHECK(n && !ixs_is_error(n));

  ixs_ctx_destroy(ctx);
}

int main(void) {
  test_integers();
  test_symbols();
  test_arithmetic();
  test_floor_ceil();
  test_mod();
  test_max_min_xor();
  test_piecewise();
  test_comparisons();
  test_errors();
  test_complex_expr();
  test_negation();

  printf("test_parser: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
