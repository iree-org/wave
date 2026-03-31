/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include <ixsimpl.h>
#include <string.h>

#include "test_check.h"

static void test_rat_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *r = ixs_rat(ctx, 3, 7);
  CHECK(ixs_node_tag(r) == IXS_RAT);
  CHECK(ixs_node_rat_num(r) == 3);
  CHECK(ixs_node_rat_den(r) == 7);

  ixs_node *neg = ixs_rat(ctx, -5, 3);
  CHECK(ixs_node_rat_num(neg) == -5);
  CHECK(ixs_node_rat_den(neg) == 3);

  ixs_ctx_destroy(ctx);
  printf("  rat_accessors: OK\n");
}

static void test_sym_name(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *s = ixs_sym(ctx, "hello");
  CHECK(ixs_node_tag(s) == IXS_SYM);
  CHECK(strcmp(ixs_node_sym_name(s), "hello") == 0);

  ixs_node *s2 = ixs_sym(ctx, "x");
  CHECK(strcmp(ixs_node_sym_name(s2), "x") == 0);

  ixs_ctx_destroy(ctx);
  printf("  sym_name: OK\n");
}

static void test_add_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *expr = ixs_add(ctx, ixs_int(ctx, 3), ixs_add(ctx, a, b));

  CHECK(ixs_node_tag(expr) == IXS_ADD);
  CHECK(ixs_node_tag(ixs_node_add_coeff(expr)) == IXS_INT);
  CHECK(ixs_node_int_val(ixs_node_add_coeff(expr)) == 3);

  uint32_t n = ixs_node_add_nterms(expr);
  CHECK(n == 2);

  int found_a = 0, found_b = 0;
  uint32_t i;
  for (i = 0; i < n; i++) {
    ixs_node *term = ixs_node_add_term(expr, i);
    ixs_node *coeff = ixs_node_add_term_coeff(expr, i);
    CHECK(ixs_node_tag(coeff) == IXS_INT);
    CHECK(ixs_node_int_val(coeff) == 1);
    if (ixs_same_node(term, a))
      found_a = 1;
    else if (ixs_same_node(term, b))
      found_b = 1;
  }
  CHECK(found_a && found_b);

  ixs_ctx_destroy(ctx);
  printf("  add_accessors: OK\n");
}

static void test_mul_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *expr = ixs_mul(ctx, ixs_int(ctx, 5), x);

  CHECK(ixs_node_tag(expr) == IXS_MUL);
  CHECK(ixs_node_int_val(ixs_node_mul_coeff(expr)) == 5);
  CHECK(ixs_node_mul_nfactors(expr) == 1);
  CHECK(ixs_same_node(ixs_node_mul_factor_base(expr, 0), x));
  CHECK(ixs_node_mul_factor_exp(expr, 0) == 1);

  ixs_ctx_destroy(ctx);
  printf("  mul_accessors: OK\n");
}

static void test_unary_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  /* x/2 is not integer-valued, so floor/ceil remain as FLOOR/CEIL nodes. */
  ixs_node *arg = ixs_div(ctx, x, ixs_int(ctx, 2));

  ixs_node *fl = ixs_floor(ctx, arg);
  CHECK(ixs_node_tag(fl) == IXS_FLOOR);
  CHECK(ixs_same_node(ixs_node_unary_arg(fl), arg));

  ixs_node *ce = ixs_ceil(ctx, arg);
  CHECK(ixs_node_tag(ce) == IXS_CEIL);
  CHECK(ixs_same_node(ixs_node_unary_arg(ce), arg));

  ixs_node *xr = ixs_xor(ctx, ixs_sym(ctx, "a"), ixs_sym(ctx, "b"));
  ixs_node *nt = ixs_not(ctx, xr);
  CHECK(ixs_node_tag(nt) == IXS_NOT);
  CHECK(ixs_same_node(ixs_node_unary_arg(nt), xr));

  ixs_ctx_destroy(ctx);
  printf("  unary_accessors: OK\n");
}

static void test_binary_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");

  ixs_node *mod = ixs_mod(ctx, a, b);
  CHECK(ixs_node_tag(mod) == IXS_MOD);
  CHECK(ixs_same_node(ixs_node_binary_lhs(mod), a));
  CHECK(ixs_same_node(ixs_node_binary_rhs(mod), b));

  ixs_node *mx = ixs_max(ctx, a, b);
  CHECK(ixs_node_tag(mx) == IXS_MAX);
  CHECK(ixs_same_node(ixs_node_binary_lhs(mx), a));
  CHECK(ixs_same_node(ixs_node_binary_rhs(mx), b));

  ixs_node *mn = ixs_min(ctx, a, b);
  CHECK(ixs_node_tag(mn) == IXS_MIN);
  CHECK(ixs_same_node(ixs_node_binary_lhs(mn), a));
  CHECK(ixs_same_node(ixs_node_binary_rhs(mn), b));

  ixs_node *xr = ixs_xor(ctx, a, b);
  CHECK(ixs_node_tag(xr) == IXS_XOR);
  CHECK(ixs_same_node(ixs_node_binary_lhs(xr), a));
  CHECK(ixs_same_node(ixs_node_binary_rhs(xr), b));

  ixs_node *diff = ixs_sub(ctx, a, b);
  ixs_node *cmp = ixs_cmp(ctx, diff, IXS_CMP_LT, ixs_int(ctx, 0));
  CHECK(ixs_node_tag(cmp) == IXS_CMP);
  CHECK(ixs_same_node(ixs_node_binary_lhs(cmp), diff));
  CHECK(ixs_same_node(ixs_node_binary_rhs(cmp), ixs_int(ctx, 0)));
  CHECK(ixs_node_cmp_op(cmp) == IXS_CMP_LT);

  ixs_ctx_destroy(ctx);
  printf("  binary_accessors: OK\n");
}

static void test_pw_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *v0 = ixs_int(ctx, 10);
  ixs_node *c0 = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *v1 = ixs_int(ctx, 20);
  ixs_node *c1 = ixs_true(ctx);
  ixs_node *vals[] = {v0, v1};
  ixs_node *conds[] = {c0, c1};
  ixs_node *pw = ixs_pw(ctx, 2, vals, conds);

  CHECK(ixs_node_tag(pw) == IXS_PIECEWISE);
  CHECK(ixs_node_pw_ncases(pw) == 2);
  CHECK(ixs_same_node(ixs_node_pw_value(pw, 0), v0));
  CHECK(ixs_same_node(ixs_node_pw_cond(pw, 0), c0));
  CHECK(ixs_same_node(ixs_node_pw_value(pw, 1), v1));
  CHECK(ixs_same_node(ixs_node_pw_cond(pw, 1), c1));

  ixs_ctx_destroy(ctx);
  printf("  pw_accessors: OK\n");
}

static void test_logic_accessors(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *c1 = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *c2 = ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 10));

  ixs_node *and_node = ixs_and(ctx, c1, c2);
  CHECK(ixs_node_tag(and_node) == IXS_AND);
  CHECK(ixs_node_logic_nargs(and_node) == 2);
  CHECK(ixs_same_node(ixs_node_logic_arg(and_node, 0), c1) ||
        ixs_same_node(ixs_node_logic_arg(and_node, 1), c1));
  CHECK(ixs_same_node(ixs_node_logic_arg(and_node, 0), c2) ||
        ixs_same_node(ixs_node_logic_arg(and_node, 1), c2));

  ixs_node *or_node = ixs_or(ctx, c1, c2);
  CHECK(ixs_node_tag(or_node) == IXS_OR);
  CHECK(ixs_node_logic_nargs(or_node) == 2);

  ixs_ctx_destroy(ctx);
  printf("  logic_accessors: OK\n");
}

/* ---- Generic child access tests ---- */

static void test_nchildren_leaves(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  CHECK(ixs_node_nchildren(ixs_int(ctx, 0)) == 0);
  CHECK(ixs_node_nchildren(ixs_rat(ctx, 1, 3)) == 0);
  CHECK(ixs_node_nchildren(ixs_sym(ctx, "x")) == 0);
  CHECK(ixs_node_nchildren(ixs_true(ctx)) == 0);
  CHECK(ixs_node_nchildren(ixs_false(ctx)) == 0);
  ixs_ctx_destroy(ctx);
  printf("  nchildren_leaves: OK\n");
}

static void test_nchildren_child_add(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *expr = ixs_add(ctx, ixs_int(ctx, 3), ixs_add(ctx, a, b));

  CHECK(ixs_node_tag(expr) == IXS_ADD);
  /* coeff + 2 terms * 2 (coeff,term) each = 5 children */
  CHECK(ixs_node_nchildren(expr) == 5);
  /* child 0 = coeff */
  CHECK(ixs_same_node(ixs_node_child(expr, 0), ixs_node_add_coeff(expr)));
  /* children match type-specific accessors */
  CHECK(
      ixs_same_node(ixs_node_child(expr, 1), ixs_node_add_term_coeff(expr, 0)));
  CHECK(ixs_same_node(ixs_node_child(expr, 2), ixs_node_add_term(expr, 0)));
  CHECK(
      ixs_same_node(ixs_node_child(expr, 3), ixs_node_add_term_coeff(expr, 1)));
  CHECK(ixs_same_node(ixs_node_child(expr, 4), ixs_node_add_term(expr, 1)));
  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_add: OK\n");
}

static void test_nchildren_child_mul(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *expr = ixs_mul(ctx, ixs_int(ctx, 5), x);

  CHECK(ixs_node_tag(expr) == IXS_MUL);
  CHECK(ixs_node_nchildren(expr) == 2);
  CHECK(ixs_same_node(ixs_node_child(expr, 0), ixs_node_mul_coeff(expr)));
  CHECK(ixs_same_node(ixs_node_child(expr, 1),
                      ixs_node_mul_factor_base(expr, 0)));
  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_mul: OK\n");
}

static void test_nchildren_child_binary(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *mod = ixs_mod(ctx, a, b);

  CHECK(ixs_node_nchildren(mod) == 2);
  CHECK(ixs_same_node(ixs_node_child(mod, 0), a));
  CHECK(ixs_same_node(ixs_node_child(mod, 1), b));
  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_binary: OK\n");
}

static void test_nchildren_child_unary(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *arg = ixs_div(ctx, x, ixs_int(ctx, 2));

  ixs_node *fl = ixs_floor(ctx, arg);
  CHECK(ixs_node_nchildren(fl) == 1);
  CHECK(ixs_same_node(ixs_node_child(fl, 0), arg));

  ixs_node *xr = ixs_xor(ctx, ixs_sym(ctx, "a"), ixs_sym(ctx, "b"));
  ixs_node *nt = ixs_not(ctx, xr);
  CHECK(ixs_node_nchildren(nt) == 1);
  CHECK(ixs_same_node(ixs_node_child(nt, 0), xr));

  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_unary: OK\n");
}

static void test_nchildren_child_pw(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *v0 = ixs_int(ctx, 10);
  ixs_node *c0 = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *v1 = ixs_int(ctx, 20);
  ixs_node *c1 = ixs_true(ctx);
  ixs_node *vals[] = {v0, v1};
  ixs_node *conds[] = {c0, c1};
  ixs_node *pw = ixs_pw(ctx, 2, vals, conds);

  CHECK(ixs_node_nchildren(pw) == 4);
  CHECK(ixs_same_node(ixs_node_child(pw, 0), v0));
  CHECK(ixs_same_node(ixs_node_child(pw, 1), c0));
  CHECK(ixs_same_node(ixs_node_child(pw, 2), v1));
  CHECK(ixs_same_node(ixs_node_child(pw, 3), c1));

  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_pw: OK\n");
}

static void test_nchildren_child_logic(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *c1 = ixs_cmp(ctx, x, IXS_CMP_GT, ixs_int(ctx, 0));
  ixs_node *c2 = ixs_cmp(ctx, x, IXS_CMP_LT, ixs_int(ctx, 10));

  ixs_node *and_node = ixs_and(ctx, c1, c2);
  CHECK(ixs_node_nchildren(and_node) == 2);
  CHECK(ixs_same_node(ixs_node_child(and_node, 0),
                      ixs_node_logic_arg(and_node, 0)));
  CHECK(ixs_same_node(ixs_node_child(and_node, 1),
                      ixs_node_logic_arg(and_node, 1)));

  ixs_ctx_destroy(ctx);
  printf("  nchildren_child_logic: OK\n");
}

/* ---- Walk tests ---- */

typedef struct {
  ixs_tag *tags;
  uint32_t count;
  uint32_t cap;
} tag_log;

static ixs_walk_action log_tags(ixs_node *node, void *ud) {
  tag_log *log = (tag_log *)ud;
  if (log->count < log->cap)
    log->tags[log->count] = ixs_node_tag(node);
  log->count++;
  return IXS_WALK_CONTINUE;
}

static void test_walk_pre_order(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  /* 5*x: MUL(coeff=5, factors=[{x,1}]) */
  ixs_node *expr = ixs_mul(ctx, ixs_int(ctx, 5), x);

  ixs_tag buf[16];
  tag_log log = {buf, 0, 16};
  ixs_node *res = ixs_walk_pre(ctx, expr, log_tags, &log);

  CHECK(res == expr);
  CHECK(log.count == 3);
  CHECK(buf[0] == IXS_MUL);

  ixs_ctx_destroy(ctx);
  printf("  walk_pre_order: OK\n");
}

static void test_walk_post_order(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *x = ixs_sym(ctx, "x");
  ixs_node *expr = ixs_mul(ctx, ixs_int(ctx, 5), x);

  ixs_tag buf[16];
  tag_log log = {buf, 0, 16};
  ixs_node *res = ixs_walk_post(ctx, expr, log_tags, &log);

  CHECK(res == expr);
  CHECK(log.count == 3);
  CHECK(buf[log.count - 1] == IXS_MUL);

  ixs_ctx_destroy(ctx);
  printf("  walk_post_order: OK\n");
}

static ixs_walk_action stop_on_sym(ixs_node *node, void *ud) {
  (void)ud;
  if (ixs_node_tag(node) == IXS_SYM)
    return IXS_WALK_STOP;
  return IXS_WALK_CONTINUE;
}

static void test_walk_stop(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  ixs_node *expr = ixs_add(ctx, a, b);

  ixs_node *res = ixs_walk_pre(ctx, expr, stop_on_sym, NULL);
  CHECK(res != NULL);
  CHECK(res != expr);
  CHECK(ixs_node_tag(res) == IXS_SYM);

  ixs_ctx_destroy(ctx);
  printf("  walk_stop: OK\n");
}

static ixs_walk_action skip_add(ixs_node *node, void *ud) {
  tag_log *log = (tag_log *)ud;
  if (log->count < log->cap)
    log->tags[log->count] = ixs_node_tag(node);
  log->count++;
  if (ixs_node_tag(node) == IXS_ADD)
    return IXS_WALK_SKIP;
  return IXS_WALK_CONTINUE;
}

static void test_walk_skip(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *a = ixs_sym(ctx, "a");
  ixs_node *b = ixs_sym(ctx, "b");
  /* mod(a + b, 3): MOD(lhs=ADD(a,b), rhs=3) */
  ixs_node *sum = ixs_add(ctx, a, b);
  ixs_node *expr = ixs_mod(ctx, sum, ixs_int(ctx, 3));

  ixs_tag buf[32];
  tag_log log = {buf, 0, 32};
  ixs_node *res = ixs_walk_pre(ctx, expr, skip_add, &log);
  CHECK(res == expr);

  uint32_t i;
  int found_sym = 0;
  for (i = 0; i < log.count; i++) {
    if (buf[i] == IXS_SYM)
      found_sym = 1;
  }
  CHECK(!found_sym);

  ixs_ctx_destroy(ctx);
  printf("  walk_skip: OK\n");
}

static void test_walk_null_root(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *res = ixs_walk_pre(ctx, NULL, log_tags, NULL);
  CHECK(res == NULL);
  res = ixs_walk_post(ctx, NULL, log_tags, NULL);
  CHECK(res == NULL);
  ixs_ctx_destroy(ctx);
  printf("  walk_null_root: OK\n");
}

static void test_walk_leaf(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *n = ixs_int(ctx, 42);

  ixs_tag buf[4];
  tag_log log = {buf, 0, 4};
  ixs_node *res = ixs_walk_pre(ctx, n, log_tags, &log);
  CHECK(res == n);
  CHECK(log.count == 1);
  CHECK(buf[0] == IXS_INT);

  ixs_ctx_destroy(ctx);
  printf("  walk_leaf: OK\n");
}

static void test_walk_sentinel(void) {
  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *err = ixs_div(ctx, ixs_int(ctx, 1), ixs_int(ctx, 0));
  CHECK(ixs_is_error(err));

  ixs_tag buf[4];
  tag_log log = {buf, 0, 4};
  ixs_node *res = ixs_walk_pre(ctx, err, log_tags, &log);
  CHECK(res == err);
  CHECK(log.count == 1);

  ixs_ctx_destroy(ctx);
  printf("  walk_sentinel: OK\n");
}

int main(void) {
  printf("test_introspect:\n");
  test_rat_accessors();
  test_sym_name();
  test_add_accessors();
  test_mul_accessors();
  test_unary_accessors();
  test_binary_accessors();
  test_pw_accessors();
  test_logic_accessors();
  test_nchildren_leaves();
  test_nchildren_child_add();
  test_nchildren_child_mul();
  test_nchildren_child_binary();
  test_nchildren_child_unary();
  test_nchildren_child_pw();
  test_nchildren_child_logic();
  test_walk_pre_order();
  test_walk_post_order();
  test_walk_stop();
  test_walk_skip();
  test_walk_null_root();
  test_walk_leaf();
  test_walk_sentinel();
  printf("test_introspect: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
