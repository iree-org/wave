/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "parser.h"
#include "simplify.h"
#include <ctype.h>
#include <limits.h>
#include <string.h>

#define PARSER_MAX_DEPTH 256

typedef struct {
  ixs_ctx *ctx;
  const char *input;
  size_t len;
  size_t pos;
  int depth;
  int max_depth;
} parser;

/* --- Lexer helpers --- */

static void skip_ws(parser *p) {
  while (p->pos < p->len &&
         (p->input[p->pos] == ' ' || p->input[p->pos] == '\t' ||
          p->input[p->pos] == '\n' || p->input[p->pos] == '\r'))
    p->pos++;
}

static bool at_end(parser *p) {
  skip_ws(p);
  return p->pos >= p->len;
}

static char peek(parser *p) {
  skip_ws(p);
  if (p->pos >= p->len)
    return '\0';
  return p->input[p->pos];
}

static bool match_char(parser *p, char c) {
  skip_ws(p);
  if (p->pos < p->len && p->input[p->pos] == c) {
    p->pos++;
    return true;
  }
  return false;
}

static bool match_str(parser *p, const char *s) {
  size_t slen = strlen(s);
  skip_ws(p);
  if (p->pos + slen <= p->len && memcmp(p->input + p->pos, s, slen) == 0) {
    /* Check that the next char isn't alphanumeric (keyword boundary). */
    if (slen > 0 && isalpha((unsigned char)s[slen - 1])) {
      if (p->pos + slen < p->len &&
          (isalnum((unsigned char)p->input[p->pos + slen]) ||
           p->input[p->pos + slen] == '_' || p->input[p->pos + slen] == '$'))
        return false;
    }
    p->pos += slen;
    return true;
  }
  return false;
}

static ixs_node *parse_error(parser *p, const char *msg) {
  ixs_ctx_push_error(p->ctx, "parse error at offset %zu: %s", p->pos, msg);
  return p->ctx->sentinel_parse_error;
}

static bool depth_push(parser *p) {
  if (p->depth >= p->max_depth) {
    ixs_ctx_push_error(p->ctx,
                       "parse error: recursion depth limit (%d) exceeded",
                       p->max_depth);
    return false;
  }
  p->depth++;
  return true;
}

static void depth_pop(parser *p) { p->depth--; }

/* --- Forward declarations --- */

static ixs_node *parse_expr(parser *p);
static ixs_node *parse_cond(parser *p);

/* --- Grammar implementation --- */

static ixs_node *parse_int(parser *p) {
  skip_ws(p);
  size_t start = p->pos;
  if (p->pos >= p->len || !isdigit((unsigned char)p->input[p->pos]))
    return NULL;

  int64_t val = 0;
  while (p->pos < p->len && isdigit((unsigned char)p->input[p->pos])) {
    int d = p->input[p->pos] - '0';
    if (val > (INT64_MAX - d) / 10) {
      /* Overflow */
      while (p->pos < p->len && isdigit((unsigned char)p->input[p->pos]))
        p->pos++;
      ixs_ctx_push_error(p->ctx, "integer literal overflow at offset %zu",
                         start);
      return p->ctx->sentinel_error;
    }
    val = val * 10 + d;
    p->pos++;
  }
  return ixs_node_int(p->ctx, val);
}

static ixs_node *parse_symbol(parser *p) {
  skip_ws(p);
  size_t start = p->pos;
  if (p->pos >= p->len)
    return NULL;

  char c = p->input[p->pos];
  if (!isalpha((unsigned char)c) && c != '_' && c != '$')
    return NULL;

  while (p->pos < p->len) {
    c = p->input[p->pos];
    if (isalnum((unsigned char)c) || c == '_' || c == '$')
      p->pos++;
    else
      break;
  }

  return ixs_node_sym(p->ctx, p->input + start, p->pos - start);
}

static ixs_node *parse_atom(parser *p);

static ixs_node *parse_func_1(parser *p, const char *name) {
  (void)name;
  if (!match_char(p, '('))
    return parse_error(p, "expected '(' after function name");
  ixs_node *arg = parse_expr(p);
  if (!arg)
    return NULL;
  if (ixs_node_is_sentinel(arg))
    return arg;
  if (!match_char(p, ')'))
    return parse_error(p, "expected ')' after function argument");
  return arg;
}

typedef ixs_node *(*binary_ctor)(ixs_ctx *, ixs_node *, ixs_node *);

static ixs_node *parse_func_2(parser *p, const char *name, binary_ctor ctor) {
  if (!match_char(p, '('))
    return parse_error(p, "expected '(' after function name");
  ixs_node *a = parse_expr(p);
  if (!a)
    return NULL;
  if (!match_char(p, ','))
    return parse_error(p, "expected ',' in function call");
  ixs_node *b = parse_expr(p);
  if (!b)
    return NULL;
  if (!match_char(p, ')'))
    return parse_error(p, "expected ')' after function arguments");
  (void)name;
  return ctor(p->ctx, a, b);
}

static ixs_node *parse_piecewise_impl(parser *p) {
  size_t cap = 16;
  ixs_node **values =
      ixs_arena_alloc(&p->ctx->scratch, cap * sizeof(*values), sizeof(void *));
  ixs_node **conds =
      ixs_arena_alloc(&p->ctx->scratch, cap * sizeof(*conds), sizeof(void *));
  if (!values || !conds)
    return NULL;
  uint32_t n = 0;

  if (!match_char(p, '('))
    return parse_error(p, "expected '(' after Piecewise");

  while (!at_end(p) && peek(p) != ')') {
    if (n > 0 && !match_char(p, ','))
      return parse_error(p, "expected ',' between Piecewise cases");

    if (!match_char(p, '('))
      return parse_error(p, "expected '(' for Piecewise case");

    ixs_node *val = parse_expr(p);
    if (!val)
      return NULL;

    if (!match_char(p, ','))
      return parse_error(p, "expected ',' in Piecewise case");

    ixs_node *cond = parse_cond(p);
    if (!cond)
      return NULL;

    if (!match_char(p, ')'))
      return parse_error(p, "expected ')' after Piecewise case");

    if (n >= cap) {
      size_t old_cap = cap;
      size_t new_cap = old_cap * 2;
      if (new_cap <= old_cap || new_cap > (size_t)-1 / sizeof(*values))
        return NULL;
      values =
          ixs_arena_grow(&p->ctx->scratch, values, old_cap * sizeof(*values),
                         new_cap * sizeof(*values), sizeof(void *));
      conds = ixs_arena_grow(&p->ctx->scratch, conds, old_cap * sizeof(*conds),
                             new_cap * sizeof(*conds), sizeof(void *));
      if (!values || !conds)
        return NULL;
      cap = new_cap;
    }
    values[n] = val;
    conds[n] = cond;
    n++;
  }

  if (!match_char(p, ')'))
    return parse_error(p, "expected ')' after Piecewise");

  if (n == 0)
    return parse_error(p, "empty Piecewise");

  return simp_pw(p->ctx, n, values, conds);
}

static ixs_node *parse_piecewise(parser *p) {
  ixs_arena_mark m = ixs_arena_save(&p->ctx->scratch);
  ixs_node *result = parse_piecewise_impl(p);
  ixs_arena_restore(&p->ctx->scratch, m);
  return result;
}

static ixs_node *parse_atom(parser *p) {
  ixs_node *result;

  if (!depth_push(p))
    return p->ctx->sentinel_parse_error;

  skip_ws(p);

  /* Parenthesized expression */
  if (peek(p) == '(') {
    match_char(p, '(');
    result = parse_expr(p);
    if (!result) {
      depth_pop(p);
      return NULL;
    }
    if (!match_char(p, ')')) {
      depth_pop(p);
      return parse_error(p, "expected ')'");
    }
    depth_pop(p);
    return result;
  }

  /* Integer literal */
  if (p->pos < p->len && isdigit((unsigned char)p->input[p->pos])) {
    result = parse_int(p);
    depth_pop(p);
    return result;
  }

  /* Keywords / functions */
  if (match_str(p, "floor")) {
    result = parse_func_1(p, "floor");
    depth_pop(p);
    return result ? simp_floor(p->ctx, result) : NULL;
  }
  if (match_str(p, "ceiling")) {
    result = parse_func_1(p, "ceiling");
    depth_pop(p);
    return result ? simp_ceil(p->ctx, result) : NULL;
  }
  if (match_str(p, "Mod")) {
    result = parse_func_2(p, "Mod", simp_mod);
    depth_pop(p);
    return result;
  }
  if (match_str(p, "Max")) {
    result = parse_func_2(p, "Max", simp_max);
    depth_pop(p);
    return result;
  }
  if (match_str(p, "Min")) {
    result = parse_func_2(p, "Min", simp_min);
    depth_pop(p);
    return result;
  }
  if (match_str(p, "xor")) {
    result = parse_func_2(p, "xor", simp_xor);
    depth_pop(p);
    return result;
  }
  if (match_str(p, "Piecewise")) {
    result = parse_piecewise(p);
    depth_pop(p);
    return result;
  }
  if (match_str(p, "True")) {
    depth_pop(p);
    return p->ctx->node_true;
  }
  if (match_str(p, "False")) {
    depth_pop(p);
    return p->ctx->node_false;
  }

  /* Symbol */
  if (p->pos < p->len) {
    char c = p->input[p->pos];
    if (isalpha((unsigned char)c) || c == '_' || c == '$') {
      result = parse_symbol(p);
      depth_pop(p);
      return result;
    }
  }

  depth_pop(p);
  return parse_error(p, "unexpected token");
}

static ixs_node *parse_unary(parser *p) {
  skip_ws(p);
  if (peek(p) == '-') {
    match_char(p, '-');
    ixs_node *a = parse_unary(p);
    if (!a)
      return NULL;
    return simp_neg(p->ctx, a);
  }
  return parse_atom(p);
}

static ixs_node *parse_term(parser *p) {
  ixs_node *left = parse_unary(p);
  if (!left)
    return NULL;

  for (;;) {
    skip_ws(p);
    if (peek(p) == '*') {
      /* Check for ** (power) — not in our grammar, skip. */
      if (p->pos + 1 < p->len && p->input[p->pos + 1] == '*') {
        break; /* Stop, don't consume ** */
      }
      match_char(p, '*');
      ixs_node *right = parse_unary(p);
      if (!right)
        return NULL;
      left = simp_mul(p->ctx, left, right);
      if (!left)
        return NULL;
    } else if (peek(p) == '/') {
      match_char(p, '/');
      ixs_node *right = parse_unary(p);
      if (!right)
        return NULL;
      left = simp_div(p->ctx, left, right);
      if (!left)
        return NULL;
    } else {
      break;
    }
  }
  return left;
}

static ixs_node *parse_expr(parser *p) {
  ixs_node *left = parse_term(p);
  if (!left)
    return NULL;

  for (;;) {
    skip_ws(p);
    if (peek(p) == '+') {
      match_char(p, '+');
      ixs_node *right = parse_term(p);
      if (!right)
        return NULL;
      left = simp_add(p->ctx, left, right);
      if (!left)
        return NULL;
    } else if (peek(p) == '-') {
      match_char(p, '-');
      ixs_node *right = parse_term(p);
      if (!right)
        return NULL;
      left = simp_sub(p->ctx, left, right);
      if (!left)
        return NULL;
    } else {
      break;
    }
  }
  return left;
}

/* --- Condition parsing --- */

static ixs_node *parse_cmp_expr(parser *p);

static ixs_node *parse_cmp_expr(parser *p) {
  skip_ws(p);

  /* ~expr */
  if (peek(p) == '~') {
    match_char(p, '~');
    ixs_node *a = parse_cmp_expr(p);
    if (!a)
      return NULL;
    return simp_not(p->ctx, a);
  }

  /* True / False */
  if (match_str(p, "True"))
    return p->ctx->node_true;
  if (match_str(p, "False"))
    return p->ctx->node_false;

  /* (cond) */
  if (peek(p) == '(') {
    match_char(p, '(');
    ixs_node *c = parse_cond(p);
    if (!c)
      return NULL;
    if (!match_char(p, ')'))
      return parse_error(p, "expected ')' in condition");
    return c;
  }

  /* expr [cmp_op expr] */
  ixs_node *left = parse_expr(p);
  if (!left)
    return NULL;

  skip_ws(p);
  ixs_cmp_op op;
  bool have_cmp = false;

  if (p->pos + 1 < p->len && p->input[p->pos] == '>' &&
      p->input[p->pos + 1] == '=') {
    p->pos += 2;
    op = IXS_CMP_GE;
    have_cmp = true;
  } else if (p->pos + 1 < p->len && p->input[p->pos] == '<' &&
             p->input[p->pos + 1] == '=') {
    p->pos += 2;
    op = IXS_CMP_LE;
    have_cmp = true;
  } else if (p->pos + 1 < p->len && p->input[p->pos] == '=' &&
             p->input[p->pos + 1] == '=') {
    p->pos += 2;
    op = IXS_CMP_EQ;
    have_cmp = true;
  } else if (p->pos + 1 < p->len && p->input[p->pos] == '!' &&
             p->input[p->pos + 1] == '=') {
    p->pos += 2;
    op = IXS_CMP_NE;
    have_cmp = true;
  } else if (p->pos < p->len && p->input[p->pos] == '>') {
    p->pos++;
    op = IXS_CMP_GT;
    have_cmp = true;
  } else if (p->pos < p->len && p->input[p->pos] == '<') {
    p->pos++;
    op = IXS_CMP_LT;
    have_cmp = true;
  }

  if (have_cmp) {
    ixs_node *right = parse_expr(p);
    if (!right)
      return NULL;
    return simp_cmp(p->ctx, left, op, right);
  }

  /* Bare expression in condition context → e != 0 */
  return simp_cmp(p->ctx, left, IXS_CMP_NE, ixs_node_int(p->ctx, 0));
}

static ixs_node *parse_cond(parser *p) {
  ixs_node *left = parse_cmp_expr(p);
  if (!left)
    return NULL;

  for (;;) {
    skip_ws(p);
    if (peek(p) == '&') {
      match_char(p, '&');
      ixs_node *right = parse_cmp_expr(p);
      if (!right)
        return NULL;
      left = simp_and(p->ctx, left, right);
      if (!left)
        return NULL;
    } else if (peek(p) == '|') {
      match_char(p, '|');
      ixs_node *right = parse_cmp_expr(p);
      if (!right)
        return NULL;
      left = simp_or(p->ctx, left, right);
      if (!left)
        return NULL;
    } else {
      break;
    }
  }
  return left;
}

/* --- Public entry point --- */

IXS_STATIC ixs_node *ixs_parse_impl(ixs_ctx *ctx, const char *input,
                                    size_t len) {
  parser p;
  p.ctx = ctx;
  p.input = input;
  p.len = len;
  p.pos = 0;
  p.depth = 0;
  p.max_depth = PARSER_MAX_DEPTH;

  ixs_node *result = parse_expr(&p);
  if (!result)
    return NULL;

  skip_ws(&p);
  if (p.pos < p.len)
    return parse_error(&p, "trailing characters");

  return result;
}
