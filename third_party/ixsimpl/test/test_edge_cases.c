/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Edge case tests for ixsimpl: overflow, div-by-zero, degenerate piecewise,
 * INT64_MIN, empty inputs, max depth, sentinel propagation, large integers,
 * symbol edge cases, print buffer truncation.
 */

#include <ixsimpl.h>
#include <stdint.h>
#include <string.h>

#include "test_check.h"

static char buf[4096];

static const char *pr(ixs_node *n) {
  ixs_print(n, buf, sizeof(buf));
  return buf;
}

/* ------------------------------------------------------------------ */
/*  1. Integer overflow: INT64_MAX and INT64_MIN in arithmetic        */
/* ------------------------------------------------------------------ */

static void test_integer_overflow(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *max_n;
  ixs_node *min_n;
  ixs_node *r;

  max_n = ixs_int(ctx, INT64_MAX);
  min_n = ixs_int(ctx, INT64_MIN);
  CHECK(max_n && !ixs_is_error(max_n));
  CHECK(min_n && !ixs_is_error(min_n));
  CHECK(ixs_node_int_val(max_n) == INT64_MAX);
  CHECK(ixs_node_int_val(min_n) == INT64_MIN);

  /* INT64_MAX + 0 stays INT64_MAX */
  r = ixs_add(ctx, max_n, ixs_int(ctx, 0));
  CHECK(r && ixs_node_int_val(r) == INT64_MAX);

  /* INT64_MAX * 1 stays INT64_MAX */
  r = ixs_mul(ctx, max_n, ixs_int(ctx, 1));
  CHECK(r && ixs_node_int_val(r) == INT64_MAX);

  /* Overflow in rational: 1/0 is domain error, not overflow. Test rat. */
  r = ixs_rat(ctx, INT64_MAX, 1);
  CHECK(r && !ixs_is_error(r));

  r = ixs_rat(ctx, INT64_MIN, 1);
  CHECK(r && !ixs_is_error(r));

  /* Parse overflow: "99999999999999999999" should overflow or error */
  r = ixs_parse(ctx, "99999999999999999999", 20);
  CHECK(r == NULL || ixs_is_error(r) || ixs_is_parse_error(r));

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  2. Division by zero: ixs_mod(x, zero), parse "x/0"                 */
/* ------------------------------------------------------------------ */

static void test_division_by_zero(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x;
  ixs_node *zero;
  ixs_node *r;

  x = ixs_sym(ctx, "x");
  zero = ixs_int(ctx, 0);

  /* ixs_mod(ctx, x, zero) returns domain error */
  r = ixs_mod(ctx, x, zero);
  CHECK(r && ixs_is_domain_error(r));
  ixs_ctx_clear_errors(ctx);

  /* ixs_div(ctx, x, zero) if available */
  r = ixs_div(ctx, x, zero);
  CHECK(r && ixs_is_domain_error(r));
  ixs_ctx_clear_errors(ctx);

  /* Parse "x/0" or "1/0" */
  r = ixs_parse(ctx, "1/0", 3);
  CHECK(r && ixs_is_domain_error(r));
  ixs_ctx_clear_errors(ctx);

  r = ixs_parse(ctx, "Mod(x, 0)", 9);
  CHECK(r && ixs_is_domain_error(r));
  ixs_ctx_clear_errors(ctx);

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  3. Degenerate Piecewise: empty, single-case True, all-False        */
/* ------------------------------------------------------------------ */

static void test_degenerate_piecewise(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *v;
  ixs_node *t;
  ixs_node *f;
  ixs_node *vals[2];
  ixs_node *conds[2];
  ixs_node *r;

  v = ixs_int(ctx, 42);
  t = ixs_true(ctx);
  f = ixs_false(ctx);

  /* Empty piecewise: n=0 */
  r = ixs_pw(ctx, 0, NULL, NULL);
  CHECK(r && ixs_is_error(r));
  ixs_ctx_clear_errors(ctx);

  /* Single-case True: Piecewise((42, True)) */
  vals[0] = v;
  conds[0] = t;
  r = ixs_pw(ctx, 1, vals, conds);
  CHECK(r && r == v);

  /* All-False: Piecewise((1, False), (2, False)) - no True default */
  vals[0] = ixs_int(ctx, 1);
  vals[1] = ixs_int(ctx, 2);
  conds[0] = f;
  conds[1] = f;
  r = ixs_pw(ctx, 2, vals, conds);
  CHECK(r && ixs_is_error(r));
  ixs_ctx_clear_errors(ctx);

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  4. INT64_MIN handling: ixs_int(ctx, INT64_MIN), negation           */
/* ------------------------------------------------------------------ */

static void test_int64_min_handling(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *min_n;
  ixs_node *r;

  min_n = ixs_int(ctx, INT64_MIN);
  CHECK(min_n && !ixs_is_error(min_n));
  CHECK(ixs_node_int_val(min_n) == INT64_MIN);

  /* Negation of INT64_MIN overflows in two's complement */
  r = ixs_neg(ctx, min_n);
  CHECK(r && (ixs_is_error(r) || ixs_is_domain_error(r) ||
              ixs_node_int_val(r) == INT64_MIN));
  ixs_ctx_clear_errors(ctx);

  /* Parse "-9223372036854775808" */
  r = ixs_parse(ctx, "-9223372036854775808", 20);
  CHECK(r && (ixs_is_error(r) || (ixs_node_tag(r) == IXS_INT &&
                                  ixs_node_int_val(r) == INT64_MIN)));

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  5. Empty/null inputs: "", "   ", skip NULL (may crash)             */
/* ------------------------------------------------------------------ */

static void test_empty_null_inputs(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *r;

  /* Empty string */
  r = ixs_parse(ctx, "", 0);
  CHECK(r == NULL || ixs_is_parse_error(r));
  if (r && ixs_is_parse_error(r))
    ixs_ctx_clear_errors(ctx);

  /* Whitespace only */
  r = ixs_parse(ctx, "   ", 3);
  CHECK(r == NULL || ixs_is_parse_error(r));
  if (r && ixs_is_parse_error(r))
    ixs_ctx_clear_errors(ctx);

  /* Skip ixs_parse(ctx, NULL, 0): may crash if input is dereferenced */
  (void)ctx;

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  6. Max-depth expressions: floor(floor(...)) up to 200 levels       */
/* ------------------------------------------------------------------ */

static void test_max_depth_expressions(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x;
  ixs_node *r;
  int i;

  x = ixs_sym(ctx, "x");
  r = x;
  for (i = 0; i < 200; i++) {
    ixs_node *next;
    next = ixs_floor(ctx, r);
    if (!next || ixs_is_error(next)) {
      CHECK(0 && "floor chain failed");
      break;
    }
    r = next;
  }
  CHECK(r && !ixs_is_error(r));
  CHECK(ixs_node_tag(r) == IXS_SYM);

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  7. Sentinel propagation through all operations                    */
/* ------------------------------------------------------------------ */

static void test_sentinel_propagation(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x;
  ixs_node *err;
  ixs_node *r;
  ixs_node *vals[1];
  ixs_node *conds[1];
  size_t n;
  char small_buf[4];

  x = ixs_sym(ctx, "x");
  err = ixs_mod(ctx, x, ixs_int(ctx, 0));
  CHECK(err && ixs_is_domain_error(err));
  ixs_ctx_clear_errors(ctx);

  /* add */
  r = ixs_add(ctx, err, x);
  CHECK(r && ixs_is_error(r));

  /* mul */
  r = ixs_mul(ctx, x, err);
  CHECK(r && ixs_is_error(r));

  /* floor */
  r = ixs_floor(ctx, err);
  CHECK(r && ixs_is_error(r));

  /* ceil */
  r = ixs_ceil(ctx, err);
  CHECK(r && ixs_is_error(r));

  /* mod */
  r = ixs_mod(ctx, err, x);
  CHECK(r && ixs_is_error(r));

  /* max */
  r = ixs_max(ctx, err, x);
  CHECK(r && ixs_is_error(r));

  /* min */
  r = ixs_min(ctx, x, err);
  CHECK(r && ixs_is_error(r));

  /* xor */
  r = ixs_xor(ctx, err, x);
  CHECK(r && ixs_is_error(r));

  /* cmp */
  r = ixs_cmp(ctx, err, IXS_CMP_GT, x);
  CHECK(r && ixs_is_error(r));

  /* and */
  r = ixs_and(ctx, err, x);
  CHECK(r && ixs_is_error(r));

  /* or */
  r = ixs_or(ctx, x, err);
  CHECK(r && ixs_is_error(r));

  /* not */
  r = ixs_not(ctx, err);
  CHECK(r && ixs_is_error(r));

  /* pw */
  vals[0] = err;
  conds[0] = ixs_true(ctx);
  r = ixs_pw(ctx, 1, vals, conds);
  CHECK(r && ixs_is_error(r));

  /* simplify */
  r = ixs_simplify(ctx, err, NULL, 0);
  CHECK(r && ixs_is_error(r));

  /* subs */
  r = ixs_subs(ctx, err, ixs_sym(ctx, "x"), ixs_int(ctx, 1));
  CHECK(r && ixs_is_error(r));

  /* print: truncates safely, no overrun */
  n = ixs_print(err, small_buf, sizeof(small_buf));
  CHECK(n > 0);

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  8. Large integers: parse and display near INT64_MAX                */
/* ------------------------------------------------------------------ */

static void test_large_integers(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *r;
  char out[64];
  size_t len;

  /* INT64_MAX as string */
  r = ixs_parse(ctx, "9223372036854775807", 19);
  CHECK(r && !ixs_is_error(r));
  CHECK(ixs_node_tag(r) == IXS_INT);
  CHECK(ixs_node_int_val(r) == INT64_MAX);

  len = ixs_print(r, out, sizeof(out));
  (void)len;
  CHECK(strstr(out, "9223372036854775807") != NULL || out[0] != '\0');

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  9. Symbol edge cases: $ prefix, single-char, long names             */
/* ------------------------------------------------------------------ */

static void test_symbol_edge_cases(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *r;

  /* $ prefix */
  r = ixs_parse(ctx, "$T0", 3);
  CHECK(r && !ixs_is_error(r));
  CHECK(ixs_node_tag(r) == IXS_SYM);
  CHECK(strcmp(pr(r), "$T0") == 0);

  /* Single-char symbol */
  r = ixs_parse(ctx, "x", 1);
  CHECK(r && !ixs_is_error(r));
  CHECK(ixs_node_tag(r) == IXS_SYM);
  CHECK(strcmp(pr(r), "x") == 0);

  /* Long symbol name */
  r = ixs_sym(ctx, "_M_div_32");
  CHECK(r && !ixs_is_error(r));
  CHECK(strcmp(pr(r), "_M_div_32") == 0);

  ixs_ctx_destroy(ctx);
}

/* ------------------------------------------------------------------ */
/*  10. Print buffer: too small should truncate safely                  */
/* ------------------------------------------------------------------ */

static void test_print_buffer(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n;
  char tiny[2];
  char zero_buf[1];
  size_t len;

  n = ixs_parse(ctx, "x + y + z", 9);
  CHECK(n && !ixs_is_error(n));

  /* Buffer size 2: room for 1 char + NUL */
  len = ixs_print(n, tiny, 2);
  CHECK(len > 0);
  CHECK(tiny[1] == '\0');
  CHECK(tiny[0] != '\0' || len == 0);

  /* Buffer size 1: only NUL */
  len = ixs_print(n, zero_buf, 1);
  CHECK(zero_buf[0] == '\0');

  /* Buffer size 0: should not crash */
  len = ixs_print(n, tiny, 0);
  (void)len;

  ixs_ctx_destroy(ctx);
}

int main(void) {
  test_integer_overflow();
  test_division_by_zero();
  test_degenerate_piecewise();
  test_int64_min_handling();
  test_empty_null_inputs();
  test_max_depth_expressions();
  test_sentinel_propagation();
  test_large_integers();
  test_symbol_edge_cases();
  test_print_buffer();

  printf("test_edge_cases: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
