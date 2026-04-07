/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "expand.h"
#include "node.h"

#define EXPAND_MAX_DEPTH 256
#define EXPAND_MAX_EXP 64

/*
 * Multiply a * b, distributing over ADD on either side.
 * Both operands must already be expanded (no MUL-over-ADD internally).
 * Recursion depth is bounded by the ADD nesting depth of a and b,
 * which is at most 2 for already-expanded inputs.
 */
static ixs_node *mul_expand(ixs_ctx *ctx, ixs_node *a, ixs_node *b) {
  if (!a || !b)
    return NULL;
  if (ixs_node_is_sentinel(a))
    return a;
  if (ixs_node_is_sentinel(b))
    return b;

  if (a->tag == IXS_ADD) {
    ixs_node *result = mul_expand(ctx, a->u.add.coeff, b);
    for (uint32_t i = 0; i < a->u.add.nterms; i++) {
      ixs_node *tc = a->u.add.terms[i].coeff;
      ixs_node *tt = a->u.add.terms[i].term;
      ixs_node *tb = mul_expand(ctx, tt, b);
      result = ixs_add(ctx, result, mul_expand(ctx, tc, tb));
    }
    return result;
  }
  if (b->tag == IXS_ADD)
    return mul_expand(ctx, b, a);

  return ixs_mul(ctx, a, b);
}

static ixs_node *do_expand(ixs_ctx *ctx, ixs_node *node, int depth) {
  if (!node)
    return NULL;
  if (ixs_node_is_sentinel(node))
    return node;
  if (depth >= EXPAND_MAX_DEPTH) {
    ixs_ctx_push_error(ctx, "expand: recursion depth limit (%d) exceeded",
                       EXPAND_MAX_DEPTH);
    return ctx->sentinel_error;
  }

  switch (node->tag) {
  case IXS_INT:
  case IXS_RAT:
  case IXS_SYM:
  case IXS_TRUE:
  case IXS_FALSE:
  case IXS_ERROR:
  case IXS_PARSE_ERROR:
    return node;

  case IXS_ADD: {
    ixs_node *result = node->u.add.coeff;
    for (uint32_t i = 0; i < node->u.add.nterms; i++) {
      ixs_node *tc = node->u.add.terms[i].coeff;
      ixs_node *tt = do_expand(ctx, node->u.add.terms[i].term, depth + 1);
      result = ixs_add(ctx, result, mul_expand(ctx, tc, tt));
    }
    return result;
  }

  case IXS_MUL: {
    ixs_node *result = node->u.mul.coeff;
    for (uint32_t i = 0; i < node->u.mul.nfactors; i++) {
      ixs_node *base = do_expand(ctx, node->u.mul.factors[i].base, depth + 1);
      int32_t exp = node->u.mul.factors[i].exp;
      /* -INT32_MIN overflows; clamp to INT32_MAX */
      int32_t mag = (exp > 0) ? exp : (exp == INT32_MIN) ? INT32_MAX : -exp;
      if (mag > EXPAND_MAX_EXP) {
        ixs_ctx_push_error(ctx, "expand: exponent magnitude (%d) exceeds limit",
                           mag);
        return ctx->sentinel_error;
      }
      if (exp > 0) {
        for (int32_t e = 0; e < exp; e++)
          result = mul_expand(ctx, result, base);
      } else {
        ixs_node *pow = base;
        for (int32_t e = 1; e < mag; e++)
          pow = ixs_mul(ctx, pow, base);
        result = ixs_div(ctx, result, pow);
      }
    }
    return result;
  }

  case IXS_FLOOR:
    return ixs_floor(ctx, do_expand(ctx, node->u.unary.arg, depth + 1));
  case IXS_CEIL:
    return ixs_ceil(ctx, do_expand(ctx, node->u.unary.arg, depth + 1));

  case IXS_MOD:
    return ixs_mod(ctx, do_expand(ctx, node->u.binary.lhs, depth + 1),
                   do_expand(ctx, node->u.binary.rhs, depth + 1));
  case IXS_MAX:
    return ixs_max(ctx, do_expand(ctx, node->u.binary.lhs, depth + 1),
                   do_expand(ctx, node->u.binary.rhs, depth + 1));
  case IXS_MIN:
    return ixs_min(ctx, do_expand(ctx, node->u.binary.lhs, depth + 1),
                   do_expand(ctx, node->u.binary.rhs, depth + 1));
  case IXS_XOR:
    return ixs_xor(ctx, do_expand(ctx, node->u.binary.lhs, depth + 1),
                   do_expand(ctx, node->u.binary.rhs, depth + 1));
  case IXS_CMP:
    return ixs_cmp(ctx, do_expand(ctx, node->u.binary.lhs, depth + 1),
                   node->u.binary.cmp_op,
                   do_expand(ctx, node->u.binary.rhs, depth + 1));

  case IXS_PIECEWISE: {
    uint32_t nc = node->u.pw.ncases;
    ixs_arena_mark sm = ixs_arena_save(&ctx->scratch);
    ixs_node **vals =
        ixs_arena_alloc(&ctx->scratch, nc * sizeof(*vals), sizeof(void *));
    ixs_node **conds =
        ixs_arena_alloc(&ctx->scratch, nc * sizeof(*conds), sizeof(void *));
    if (!vals || !conds) {
      ixs_arena_restore(&ctx->scratch, sm);
      return NULL;
    }
    for (uint32_t i = 0; i < nc; i++) {
      vals[i] = do_expand(ctx, node->u.pw.cases[i].value, depth + 1);
      conds[i] = do_expand(ctx, node->u.pw.cases[i].cond, depth + 1);
    }
    ixs_node *result = ixs_pw(ctx, nc, vals, conds);
    ixs_arena_restore(&ctx->scratch, sm);
    return result;
  }

  case IXS_AND:
  case IXS_OR: {
    uint32_t na = node->u.logic.nargs;
    if (na < 2)
      return node;
    ixs_node *result = do_expand(ctx, node->u.logic.args[0], depth + 1);
    for (uint32_t i = 1; i < na; i++) {
      ixs_node *arg = do_expand(ctx, node->u.logic.args[i], depth + 1);
      result = (node->tag == IXS_AND) ? ixs_and(ctx, result, arg)
                                      : ixs_or(ctx, result, arg);
    }
    return result;
  }

  case IXS_NOT:
    return ixs_not(ctx, do_expand(ctx, node->u.unary_bool.arg, depth + 1));

  default:
    return node;
  }
}

IXS_STATIC ixs_node *expand_impl(ixs_ctx *ctx, ixs_node *expr) {
  return do_expand(ctx, expr, 0);
}
