/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_ARENA_H
#define IXS_ARENA_H

#include "internal.h"

#include <stddef.h>

#define IXS_ARENA_DEFAULT_SIZE 4096

typedef struct ixs_arena_chunk {
  char *base;
  size_t used;
  size_t capacity;
  struct ixs_arena_chunk *next;
} ixs_arena_chunk;

typedef struct {
  ixs_arena_chunk *current;
  size_t min_chunk;
} ixs_arena;

typedef struct {
  ixs_arena_chunk *chunk;
  size_t used;
} ixs_arena_mark;

IXS_STATIC void ixs_arena_init(ixs_arena *a, size_t initial_size);
IXS_STATIC void ixs_arena_destroy(ixs_arena *a);

/* Returns NULL on OOM or overflow. align must be a power of 2, at most 16. */
IXS_STATIC void *ixs_arena_alloc(ixs_arena *a, size_t size, size_t align);

/* Copy len bytes of s into the arena, null-terminate. NULL on OOM/overflow. */
IXS_STATIC char *ixs_arena_strdup(ixs_arena *a, const char *s, size_t len);

/* Snapshot current position. Pairs with ixs_arena_restore (LIFO). */
IXS_STATIC ixs_arena_mark ixs_arena_save(ixs_arena *a);

/* Rewind to mark, freeing any chunks allocated after it.
 * mark must have come from ixs_arena_save on the same arena.
 * Marks are invalidated by any ixs_arena_restore that rewinds past them. */
IXS_STATIC void ixs_arena_restore(ixs_arena *a, ixs_arena_mark m);

/*
 * Grow an existing allocation from old_size to new_size bytes.
 * Fast path extends in-place if ptr is at the tip of the current chunk.
 * Slow path allocs new block and copies; old space is wasted (fine for
 * scratch arena, not for main arena).
 * new_size must be >= old_size.  NULL ptr delegates to ixs_arena_alloc.
 * Returns NULL on OOM or if new_size < old_size.
 */
IXS_STATIC void *ixs_arena_grow(ixs_arena *a, void *ptr, size_t old_size,
                                size_t new_size, size_t align);

#endif /* IXS_ARENA_H */
