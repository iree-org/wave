/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Corpus benchmark: time parse + simplify of all expressions from
 * test/corpus.txt. Uses clock() for portable C99 timing. Runs 3 iterations,
 * reports best.
 */

#include <ixsimpl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE 8192
#define MAX_EXPRESSIONS 4096
#define MAX_ASSUMPTIONS 256
#define N_ITERATIONS 3

static const char *corpus_path = "test/corpus.txt";
static const char *assumptions_path = "test/corpus_assumptions.txt";

static const char *extract_expr(char *line) {
  const char *p = strstr(line, "s: ");
  if (!p)
    return NULL;
  p += 3;
  while (*p == ' ')
    p++;
  return p;
}

static ixs_node *parse_assumption(ixs_ctx *ctx, char *line) {
  static const struct {
    const char *op;
    size_t len;
    ixs_cmp_op cmp;
  } ops[] = {{">=", 2, IXS_CMP_GE}, {"<=", 2, IXS_CMP_LE},
             {"==", 2, IXS_CMP_EQ}, {"!=", 2, IXS_CMP_NE},
             {">", 1, IXS_CMP_GT},  {"<", 1, IXS_CMP_LT}};

  char *start = line;
  while (*start && (unsigned char)*start <= ' ')
    start++;
  if (!*start || *start == '#')
    return NULL;

  char *op_pos = NULL;
  size_t op_len = 0;
  ixs_cmp_op cmp_op = IXS_CMP_EQ;

  for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
    char *p = strstr(start, ops[i].op);
    if (p) {
      op_pos = p;
      op_len = ops[i].len;
      cmp_op = ops[i].cmp;
      break;
    }
  }
  if (!op_pos)
    return NULL;

  *op_pos = '\0';
  char *right = op_pos + op_len;
  while (*right == ' ')
    right++;

  char *left_end = op_pos;
  while (left_end > start && (unsigned char)left_end[-1] <= ' ')
    left_end--;
  *left_end = '\0';

  ixs_node *a = ixs_parse(ctx, start, (size_t)(left_end - start));
  ixs_node *b = ixs_parse(ctx, right, strlen(right));
  if (!a || !b || ixs_is_error(a) || ixs_is_error(b))
    return NULL;
  return ixs_cmp(ctx, a, cmp_op, b);
}

static size_t load_assumptions(ixs_ctx *ctx, ixs_node **assumptions,
                               size_t max_n) {
  FILE *f = fopen(assumptions_path, "r");
  if (!f)
    return 0;

  size_t n = 0;
  char line[MAX_LINE];
  while (n < max_n && fgets(line, sizeof line, f)) {
    ixs_node *a = parse_assumption(ctx, line);
    if (a)
      assumptions[n++] = a;
  }
  fclose(f);
  return n;
}

/* Load expressions from corpus. Returns count, fills exprs[]. */
static size_t load_corpus(const char **exprs, size_t max_n, char *storage,
                          size_t storage_size) {
  FILE *f = fopen(corpus_path, "r");
  if (!f)
    return 0;

  size_t n = 0;
  size_t used = 0;
  char line[MAX_LINE];

  while (n < max_n && fgets(line, sizeof line, f)) {
    const char *e = extract_expr(line);
    if (!e || !*e)
      continue;

    size_t len = strlen(e);
    if (len > 0 && e[len - 1] == '\n')
      len--;
    if (used + len + 1 > storage_size)
      break;

    memcpy(storage + used, e, len);
    storage[used + len] = '\0';
    exprs[n++] = storage + used;
    used += len + 1;
  }
  fclose(f);
  return n;
}

int main(void) {
  char *storage = malloc(MAX_EXPRESSIONS * MAX_LINE);
  if (!storage) {
    fprintf(stderr, "bench_corpus: out of memory\n");
    return 1;
  }

  const char *exprs[MAX_EXPRESSIONS];
  size_t n_exprs =
      load_corpus(exprs, MAX_EXPRESSIONS, storage, MAX_EXPRESSIONS * MAX_LINE);
  if (n_exprs == 0) {
    fprintf(stderr, "bench_corpus: no expressions in %s\n", corpus_path);
    free(storage);
    return 1;
  }

  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *assumptions[MAX_ASSUMPTIONS];
  size_t n_assumptions = load_assumptions(ctx, assumptions, MAX_ASSUMPTIONS);

  ixs_node **parsed = malloc(n_exprs * sizeof(ixs_node *));
  if (!parsed) {
    fprintf(stderr, "bench_corpus: out of memory\n");
    ixs_ctx_destroy(ctx);
    free(storage);
    return 1;
  }

  double best_parse_ms = 1e9;
  double best_simplify_ms = 1e9;

  for (int iter = 0; iter < N_ITERATIONS; iter++) {
    ixs_ctx_clear_errors(ctx);

    clock_t t0 = clock();
    for (size_t i = 0; i < n_exprs; i++) {
      parsed[i] = ixs_parse(ctx, exprs[i], strlen(exprs[i]));
      if (!parsed[i] || ixs_is_error(parsed[i]))
        parsed[i] = NULL;
    }
    clock_t t1 = clock();

    clock_t t2 = clock();
    for (size_t i = 0; i < n_exprs; i++) {
      if (parsed[i])
        ixs_simplify(ctx, parsed[i], assumptions, n_assumptions);
    }
    clock_t t3 = clock();

    double parse_ms = 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC;
    double simplify_ms = 1000.0 * (double)(t3 - t2) / CLOCKS_PER_SEC;

    if (parse_ms < best_parse_ms)
      best_parse_ms = parse_ms;
    if (simplify_ms < best_simplify_ms)
      best_simplify_ms = simplify_ms;
  }

  double total_ms = best_parse_ms + best_simplify_ms;
  double expr_per_sec = (n_exprs > 0) ? (1000.0 * n_exprs / total_ms) : 0;
  double avg_us = (n_exprs > 0) ? (1000.0 * total_ms / n_exprs) : 0;

  printf("bench_corpus: %zu expressions", n_exprs);
  if (n_assumptions > 0)
    printf(", %zu assumptions", n_assumptions);
  printf("\n");
  printf("  parse:    %.2f ms total, %.3f ms/expr\n", best_parse_ms,
         n_exprs > 0 ? best_parse_ms / n_exprs : 0);
  printf("  simplify: %.2f ms total, %.3f ms/expr\n", best_simplify_ms,
         n_exprs > 0 ? best_simplify_ms / n_exprs : 0);
  printf("  total:    %.2f ms, %.1f expr/s, %.2f us/expr (best of %d)\n",
         total_ms, expr_per_sec, avg_us, N_ITERATIONS);

  free(parsed);
  ixs_ctx_destroy(ctx);
  free(storage);
  return 0;
}
