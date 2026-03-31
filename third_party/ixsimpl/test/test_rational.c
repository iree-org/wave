/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "rational.h"
#include <assert.h>
#include <limits.h>

#include "test_check.h"

#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807LL - 1)
#endif
#ifndef INT64_MAX
#define INT64_MAX 9223372036854775807LL
#endif

static void test_gcd(void) {
  CHECK(ixs_gcd(12, 8) == 4);
  CHECK(ixs_gcd(0, 5) == 5);
  CHECK(ixs_gcd(5, 0) == 5);
  CHECK(ixs_gcd(0, 0) == 0);
  CHECK(ixs_gcd(7, 13) == 1);
  CHECK(ixs_gcd(-12, 8) == 4);
  CHECK(ixs_gcd(12, -8) == 4);
  CHECK(ixs_gcd(-12, -8) == 4);
  CHECK(ixs_gcd(1, 1) == 1);
  CHECK(ixs_gcd(INT64_MIN, 2) != 0);
  /* dual-INT64_MIN: GCD magnitude is 2^63, must not UB on cast */
  CHECK(ixs_gcd(INT64_MIN, INT64_MIN) > 0);
  CHECK(ixs_gcd(INT64_MIN, 0) > 0);
  CHECK(ixs_gcd(0, INT64_MIN) > 0);
}

static void test_normalize(void) {
  int64_t p, q;
  CHECK(ixs_rat_normalize(6, 4, &p, &q) && p == 3 && q == 2);
  CHECK(ixs_rat_normalize(0, 5, &p, &q) && p == 0 && q == 1);
  CHECK(ixs_rat_normalize(-6, 4, &p, &q) && p == -3 && q == 2);
  CHECK(ixs_rat_normalize(6, -4, &p, &q) && p == -3 && q == 2);
  CHECK(ixs_rat_normalize(-6, -4, &p, &q) && p == 3 && q == 2);
  CHECK(!ixs_rat_normalize(1, 0, &p, &q));
}

static void test_add(void) {
  int64_t p, q;
  CHECK(ixs_rat_add(1, 2, 1, 3, &p, &q) && p == 5 && q == 6);
  CHECK(ixs_rat_add(1, 1, -1, 1, &p, &q) && p == 0 && q == 1);
  CHECK(ixs_rat_add(3, 4, 1, 4, &p, &q) && p == 1 && q == 1);
}

static void test_sub(void) {
  int64_t p, q;
  CHECK(ixs_rat_sub(1, 1, 1, 2, &p, &q) && p == 1 && q == 2);
  CHECK(ixs_rat_sub(1, 3, 1, 3, &p, &q) && p == 0 && q == 1);
}

static void test_mul(void) {
  int64_t p, q;
  CHECK(ixs_rat_mul(2, 3, 3, 4, &p, &q) && p == 1 && q == 2);
  CHECK(ixs_rat_mul(0, 1, 5, 7, &p, &q) && p == 0 && q == 1);
}

static void test_div(void) {
  int64_t p, q;
  CHECK(ixs_rat_div(1, 2, 3, 4, &p, &q) && p == 2 && q == 3);
  CHECK(!ixs_rat_div(1, 1, 0, 1, &p, &q));
}

static void test_floor(void) {
  CHECK(ixs_rat_floor(7, 2) == 3);
  CHECK(ixs_rat_floor(-7, 2) == -4);
  CHECK(ixs_rat_floor(6, 3) == 2);
  CHECK(ixs_rat_floor(0, 1) == 0);
  CHECK(ixs_rat_floor(-1, 1) == -1);
  CHECK(ixs_rat_floor(1, 1) == 1);
}

static void test_ceil(void) {
  CHECK(ixs_rat_ceil(7, 2) == 4);
  CHECK(ixs_rat_ceil(-7, 2) == -3);
  CHECK(ixs_rat_ceil(6, 3) == 2);
  CHECK(ixs_rat_ceil(0, 1) == 0);
  CHECK(ixs_rat_ceil(1, 3) == 1);
}

static void test_mod(void) {
  int64_t p, q;
  CHECK(ixs_rat_mod(17, 1, 5, 1, &p, &q) && p == 2 && q == 1);
  CHECK(ixs_rat_mod(-7, 1, 2, 1, &p, &q) && p == 1 && q == 1);
  CHECK(ixs_rat_mod(0, 1, 3, 1, &p, &q) && p == 0 && q == 1);
  CHECK(ixs_rat_mod(7, 2, 3, 1, &p, &q)); /* 3.5 mod 3 = 0.5 */
}

static void test_cmp(void) {
  CHECK(ixs_rat_cmp(1, 2, 1, 3) > 0);
  CHECK(ixs_rat_cmp(1, 3, 1, 2) < 0);
  CHECK(ixs_rat_cmp(2, 4, 1, 2) == 0);
  CHECK(ixs_rat_cmp(-1, 1, 1, 1) < 0);
  /* Large values that overflow 64-bit cross-multiply */
  CHECK(ixs_rat_cmp(INT64_MAX, 1, INT64_MAX - 1, 1) > 0);
  CHECK(ixs_rat_cmp(INT64_MAX, INT64_MAX, 1, 1) == 0);
  CHECK(ixs_rat_cmp(INT64_MAX, 2, INT64_MAX, 3) > 0);
  CHECK(ixs_rat_cmp(INT64_MIN + 1, 1, INT64_MIN + 2, 1) < 0);
  CHECK(ixs_rat_cmp(INT64_MAX, INT64_MAX - 1, INT64_MAX - 1, INT64_MAX - 2) <
        0);
}

static void test_safe_ops(void) {
  int64_t r;
  CHECK(ixs_safe_add(INT64_MAX, 0, &r) && r == INT64_MAX);
  CHECK(!ixs_safe_add(INT64_MAX, 1, &r));
  CHECK(ixs_safe_sub(0, INT64_MIN + 1, &r));
  CHECK(!ixs_safe_neg(INT64_MIN, &r));
  CHECK(ixs_safe_mul(1000000000LL, 1000000000LL, &r) &&
        r == 1000000000000000000LL);
}

int main(void) {
  test_gcd();
  test_normalize();
  test_add();
  test_sub();
  test_mul();
  test_div();
  test_floor();
  test_ceil();
  test_mod();
  test_cmp();
  test_safe_ops();

  printf("test_rational: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
