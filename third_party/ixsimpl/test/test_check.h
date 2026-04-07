/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TEST_CHECK_H
#define TEST_CHECK_H

#include <stdio.h>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    tests_run++;                                                               \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
    } else {                                                                   \
      tests_passed++;                                                          \
    }                                                                          \
  } while (0)

#endif /* TEST_CHECK_H */
