/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Corpus test: parse and simplify all expressions from test/corpus.txt.
 * Reads assumptions from test/corpus_assumptions.txt if present.
 * Compares output to test/corpus_expected.txt if present (no failure if
 * missing).
 */

#include <ixsimpl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 8192
#define MAX_ASSUMPTIONS 256

static const char *corpus_path = "test/corpus.txt";
static const char *assumptions_path = "test/corpus_assumptions.txt";
static const char *expected_path = "test/corpus_expected.txt";

static void parse_path_args(int argc, char **argv) {
  int i;
  for (i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--corpus=", 9) == 0)
      corpus_path = argv[i] + 9;
    else if (strncmp(argv[i], "--assumptions=", 14) == 0)
      assumptions_path = argv[i] + 14;
    else if (strncmp(argv[i], "--expected=", 11) == 0)
      expected_path = argv[i] + 11;
  }
}

/* Extract expression from "simplify time: X.XXXXs: <expr>". Returns pointer
 * into line, or NULL if format invalid. */
static const char *extract_expr(char *line) {
  const char *p = strstr(line, "s: ");
  if (!p)
    return NULL;
  p += 3; /* skip "s: " */
  while (*p == ' ')
    p++;
  return p;
}

/* Parse one assumption line like "$T0 >= 0" or "M >= 1" into ixs_cmp node.
 * Returns NULL on parse failure or if line is blank/comment. */
static ixs_node *parse_assumption(ixs_ctx *ctx, char *line) {
  /* Trim leading whitespace */
  char *start = line;
  while (*start && (unsigned char)*start <= ' ')
    start++;
  if (!*start || *start == '#')
    return NULL;

  /* Find operator (check 2-char first) */
  static const struct {
    const char *op;
    size_t len;
    ixs_cmp_op cmp;
  } ops[] = {{">=", 2, IXS_CMP_GE}, {"<=", 2, IXS_CMP_LE},
             {"==", 2, IXS_CMP_EQ}, {"!=", 2, IXS_CMP_NE},
             {">", 1, IXS_CMP_GT},  {"<", 1, IXS_CMP_LT}};

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
  char *left = start;
  char *right = op_pos + op_len;
  while (*right == ' ')
    right++;

  /* Trim trailing whitespace from left */
  char *left_end = op_pos;
  while (left_end > left && (unsigned char)left_end[-1] <= ' ')
    left_end--;
  *left_end = '\0';

  ixs_node *a = ixs_parse(ctx, left, (size_t)(left_end - left));
  ixs_node *b = ixs_parse(ctx, right, strlen(right));
  if (!a || !b || ixs_is_error(a) || ixs_is_error(b))
    return NULL;
  return ixs_cmp(ctx, a, cmp_op, b);
}

/* Load assumptions from file. Returns count, fills assumptions[].
 * Caller must not exceed MAX_ASSUMPTIONS. */
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

int main(int argc, char **argv) {
  FILE *corpus;
  FILE *gen_expected = NULL;
  int generate = 0;
  int i;
  parse_path_args(argc, argv);
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--generate") == 0)
      generate = 1;
  }
  corpus = fopen(corpus_path, "r");
  if (!corpus) {
    fprintf(stderr, "test_corpus: cannot open %s\n", corpus_path);
    return 1;
  }
  if (generate) {
    gen_expected = fopen(expected_path, "w");
    if (!gen_expected) {
      fprintf(stderr, "test_corpus: cannot create %s\n", expected_path);
      fclose(corpus);
      return 1;
    }
  }

  ixs_ctx *ctx = ixs_ctx_create();
  ixs_node *assumptions[MAX_ASSUMPTIONS];
  size_t n_assumptions = load_assumptions(ctx, assumptions, MAX_ASSUMPTIONS);

  size_t n_lines = 0;
  size_t n_parsed = 0;
  size_t n_simplified = 0;
  size_t n_errors = 0;
  size_t n_mismatches = 0;
  char line[MAX_LINE];
  char out_buf[MAX_LINE];
  char **expected_lines = NULL;
  size_t expected_cap = 0;
  size_t expected_len = 0;

  /* Optionally load expected output for comparison (skip in generate mode) */
  if (!gen_expected) {
    FILE *expf = fopen(expected_path, "r");
    if (expf) {
      expected_cap = 4096;
      expected_lines = malloc(expected_cap * sizeof(char *));
      if (expected_lines) {
        while (expected_len < expected_cap && fgets(line, sizeof line, expf)) {
          size_t len = strlen(line);
          if (len > 0 && line[len - 1] == '\n')
            line[--len] = '\0';
          {
            size_t len2 = strlen(line) + 1;
            char *copy = malloc(len2);
            if (copy) {
              memcpy(copy, line, len2);
              expected_lines[expected_len++] = copy;
            }
          }
        }
        fclose(expf);
      } else {
        fclose(expf);
      }
    }
  }

  size_t expr_index = 0;
  while (fgets(line, sizeof line, corpus)) {
    n_lines++;
    const char *expr = extract_expr(line);
    if (!expr || !*expr)
      continue;

    ixs_node *parsed = ixs_parse(ctx, expr, strlen(expr));
    if (!parsed || ixs_is_error(parsed)) {
      n_errors++;
      if (ixs_ctx_nerrors(ctx) > 0)
        fprintf(stderr, "parse error line %zu: %s\n", n_lines,
                ixs_ctx_error(ctx, 0));
      ixs_ctx_clear_errors(ctx);
      continue;
    }
    n_parsed++;

    ixs_node *simplified =
        ixs_simplify(ctx, parsed, assumptions, n_assumptions);
    if (!simplified || ixs_is_error(simplified)) {
      n_errors++;
      if (ixs_ctx_nerrors(ctx) > 0)
        fprintf(stderr, "simplify error line %zu: %s\n", n_lines,
                ixs_ctx_error(ctx, 0));
      ixs_ctx_clear_errors(ctx);
      continue;
    }
    n_simplified++;

    {
      size_t n = ixs_print(simplified, out_buf, sizeof out_buf);
      out_buf[n] = '\0';
    }

    if (gen_expected) {
      fprintf(gen_expected, "%s\n", out_buf);
    } else if (expected_lines && expr_index < expected_len) {
      if (strcmp(out_buf, expected_lines[expr_index]) != 0) {
        fprintf(stderr, "mismatch line %zu: got '%s' expected '%s'\n", n_lines,
                out_buf, expected_lines[expr_index]);
        n_mismatches++;
      }
    }
    expr_index++;
  }
  fclose(corpus);
  if (gen_expected) {
    fclose(gen_expected);
    printf("test_corpus: generated %s with %zu entries\n", expected_path,
           n_simplified);
    ixs_ctx_destroy(ctx);
    return (n_errors == 0 && n_parsed > 0) ? 0 : 1;
  }

  if (expected_lines && expr_index != expected_len) {
    fprintf(stderr,
            "expected line count mismatch: %zu expressions vs %zu expected\n",
            expr_index, expected_len);
    n_mismatches++;
  }

  /* Free expected lines */
  if (expected_lines) {
    for (size_t i = 0; i < expected_len; i++)
      free(expected_lines[i]);
    free(expected_lines);
  }

  printf("test_corpus: %zu lines, %zu parsed, %zu simplified, %zu errors, "
         "%zu mismatches\n",
         n_lines, n_parsed, n_simplified, n_errors, n_mismatches);
  if (n_assumptions > 0)
    printf("  (using %zu assumptions from %s)\n", n_assumptions,
           assumptions_path);

  ixs_ctx_destroy(ctx);
  return (n_errors == 0 && n_mismatches == 0 && n_parsed > 0 &&
          n_parsed == n_simplified)
             ? 0
             : 1;
}
