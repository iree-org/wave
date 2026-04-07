/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Unit tests for arena save/restore and grow.
 */

#include "arena.h"

#include "test_check.h"

/* ------------------------------------------------------------------ */
/*  save/restore basics                                                */
/* ------------------------------------------------------------------ */

static void test_save_restore_basic(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  /* Pre-allocate so the chunk exists before save. */
  void *anchor = ixs_arena_alloc(&a, 16, sizeof(void *));
  CHECK(anchor != NULL);

  ixs_arena_mark m0 = ixs_arena_save(&a);

  void *p1 = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(p1 != NULL);

  void *p2 = ixs_arena_alloc(&a, 128, sizeof(void *));
  CHECK(p2 != NULL);

  ixs_arena_restore(&a, m0);

  /* After restore, the next alloc should reuse the same space. */
  void *p3 = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(p3 == p1);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  save/restore on empty arena                                        */
/* ------------------------------------------------------------------ */

static void test_save_restore_empty(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  ixs_arena_mark m = ixs_arena_save(&a);
  CHECK(m.chunk == NULL);
  CHECK(m.used == 0);

  void *p = ixs_arena_alloc(&a, 100, 1);
  CHECK(p != NULL);

  ixs_arena_restore(&a, m);
  CHECK(a.current == NULL);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  LIFO nesting                                                       */
/* ------------------------------------------------------------------ */

static void test_save_restore_nested(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  void *base = ixs_arena_alloc(&a, 32, sizeof(void *));
  CHECK(base != NULL);

  ixs_arena_mark m_outer = ixs_arena_save(&a);

  void *outer = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(outer != NULL);

  ixs_arena_mark m_inner = ixs_arena_save(&a);

  void *inner = ixs_arena_alloc(&a, 128, sizeof(void *));
  CHECK(inner != NULL);

  ixs_arena_restore(&a, m_inner);

  /* Inner region freed, outer still intact. Next alloc reuses inner's space. */
  void *after_inner = ixs_arena_alloc(&a, 128, sizeof(void *));
  CHECK(after_inner == inner);

  ixs_arena_restore(&a, m_outer);

  /* Outer region freed too. Next alloc reuses outer's space. */
  void *after_outer = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(after_outer == outer);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  restore frees intermediate chunks                                  */
/* ------------------------------------------------------------------ */

static void test_save_restore_cross_chunk(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  /* Fill most of the first chunk. */
  void *p0 = ixs_arena_alloc(&a, 4000, 1);
  CHECK(p0 != NULL);

  ixs_arena_mark m = ixs_arena_save(&a);
  ixs_arena_chunk *saved_chunk = m.chunk;

  /* Force a new chunk by allocating more than remaining capacity. */
  void *big = ixs_arena_alloc(&a, 4096, 1);
  CHECK(big != NULL);
  CHECK(a.current != saved_chunk);

  ixs_arena_restore(&a, m);
  CHECK(a.current == saved_chunk);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow: in-place fast path                                           */
/* ------------------------------------------------------------------ */

static void test_grow_fast_path(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  int *arr = ixs_arena_alloc(&a, 4 * sizeof(int), sizeof(void *));
  CHECK(arr != NULL);
  arr[0] = 10;
  arr[1] = 20;
  arr[2] = 30;
  arr[3] = 40;

  int *grown =
      ixs_arena_grow(&a, arr, 4 * sizeof(int), 8 * sizeof(int), sizeof(void *));
  CHECK(grown == arr);
  CHECK(grown[0] == 10);
  CHECK(grown[3] == 40);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow: slow path (intervening alloc defeats in-place)               */
/* ------------------------------------------------------------------ */

static void test_grow_slow_path(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  int *arr = ixs_arena_alloc(&a, 4 * sizeof(int), sizeof(void *));
  CHECK(arr != NULL);
  arr[0] = 100;
  arr[1] = 200;
  arr[2] = 300;
  arr[3] = 400;

  /* Intervening alloc moves the tip past arr. */
  void *blocker = ixs_arena_alloc(&a, 8, 1);
  CHECK(blocker != NULL);
  (void)blocker;

  int *grown =
      ixs_arena_grow(&a, arr, 4 * sizeof(int), 8 * sizeof(int), sizeof(void *));
  CHECK(grown != NULL);
  CHECK(grown != arr);
  CHECK(grown[0] == 100);
  CHECK(grown[3] == 400);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow: NULL ptr delegates to alloc                                  */
/* ------------------------------------------------------------------ */

static void test_grow_null_ptr(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  void *p = ixs_arena_grow(&a, NULL, 0, 64, sizeof(void *));
  CHECK(p != NULL);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow: shrink rejected                                              */
/* ------------------------------------------------------------------ */

static void test_grow_no_shrink(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  void *p = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(p != NULL);

  void *q = ixs_arena_grow(&a, p, 64, 32, sizeof(void *));
  CHECK(q == NULL);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow + save/restore combined                                       */
/* ------------------------------------------------------------------ */

static void test_grow_with_save_restore(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  /* Pre-allocate so the chunk exists before save. */
  void *anchor = ixs_arena_alloc(&a, 16, sizeof(void *));
  CHECK(anchor != NULL);

  ixs_arena_mark m = ixs_arena_save(&a);

  int *arr = ixs_arena_alloc(&a, 4 * sizeof(int), sizeof(void *));
  CHECK(arr != NULL);
  arr[0] = 42;

  int *grown = ixs_arena_grow(&a, arr, 4 * sizeof(int), 16 * sizeof(int),
                              sizeof(void *));
  CHECK(grown == arr);
  CHECK(grown[0] == 42);

  ixs_arena_restore(&a, m);

  /* After restore, space is reclaimed. */
  int *reused = ixs_arena_alloc(&a, 4 * sizeof(int), sizeof(void *));
  CHECK(reused == arr);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  deeply nested save/restore (5 levels)                              */
/* ------------------------------------------------------------------ */

static void test_save_restore_deep_nesting(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  /* Pre-allocate so the chunk exists before the first save. */
  void *anchor = ixs_arena_alloc(&a, 16, sizeof(void *));
  CHECK(anchor != NULL);

  ixs_arena_mark marks[5];
  void *ptrs[5];
  int i;

  for (i = 0; i < 5; i++) {
    marks[i] = ixs_arena_save(&a);
    ptrs[i] = ixs_arena_alloc(&a, 64, sizeof(void *));
    CHECK(ptrs[i] != NULL);
  }

  for (i = 4; i >= 0; i--) {
    ixs_arena_restore(&a, marks[i]);
    void *p = ixs_arena_alloc(&a, 64, sizeof(void *));
    CHECK(p == ptrs[i]);
  }

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */
/*  grow same size is no-op                                            */
/* ------------------------------------------------------------------ */

static void test_grow_same_size(void) {
  ixs_arena a;
  ixs_arena_init(&a, 4096);

  void *p = ixs_arena_alloc(&a, 64, sizeof(void *));
  CHECK(p != NULL);

  void *q = ixs_arena_grow(&a, p, 64, 64, sizeof(void *));
  CHECK(q == p);

  ixs_arena_destroy(&a);
}

/* ------------------------------------------------------------------ */

int main(void) {
  test_save_restore_basic();
  test_save_restore_empty();
  test_save_restore_nested();
  test_save_restore_cross_chunk();
  test_grow_fast_path();
  test_grow_slow_path();
  test_grow_null_ptr();
  test_grow_no_shrink();
  test_grow_with_save_restore();
  test_save_restore_deep_nesting();
  test_grow_same_size();

  printf("test_arena: %d/%d passed\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
