/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "node.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Hashing                                                           */
/* ------------------------------------------------------------------ */

static uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9u;
  h ^= h >> 16;
  return h;
}

static uint32_t hash_i64(int64_t v) {
  uint64_t u = (uint64_t)v;
  return (uint32_t)(u ^ (u >> 32));
}

static uint32_t hash_str(const char *s, size_t len) {
  uint32_t h = 5381;
  size_t i;
  for (i = 0; i < len; i++)
    h = ((h << 5) + h) ^ (unsigned char)s[i];
  return h;
}

static uint32_t compute_hash(const ixs_node *n) {
  uint32_t h = (uint32_t)n->tag * 2654435761u;
  switch (n->tag) {
  case IXS_INT:
    h = hash_mix(h, hash_i64(n->u.ival));
    break;
  case IXS_RAT:
    h = hash_mix(h, hash_i64(n->u.rat.p));
    h = hash_mix(h, hash_i64(n->u.rat.q));
    break;
  case IXS_SYM:
    h = hash_mix(h, hash_str(n->u.name, strlen(n->u.name)));
    break;
  case IXS_ADD: {
    h = hash_mix(h, n->u.add.coeff->hash);
    uint32_t i;
    for (i = 0; i < n->u.add.nterms; i++) {
      h = hash_mix(h, n->u.add.terms[i].term->hash);
      h = hash_mix(h, n->u.add.terms[i].coeff->hash);
    }
    break;
  }
  case IXS_MUL: {
    h = hash_mix(h, n->u.mul.coeff->hash);
    uint32_t i;
    for (i = 0; i < n->u.mul.nfactors; i++) {
      h = hash_mix(h, n->u.mul.factors[i].base->hash);
      h = hash_mix(h, (uint32_t)n->u.mul.factors[i].exp);
    }
    break;
  }
  case IXS_FLOOR:
  case IXS_CEIL:
    h = hash_mix(h, n->u.unary.arg->hash);
    break;
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
  case IXS_XOR:
    h = hash_mix(h, n->u.binary.lhs->hash);
    h = hash_mix(h, n->u.binary.rhs->hash);
    break;
  case IXS_CMP:
    h = hash_mix(h, n->u.binary.lhs->hash);
    h = hash_mix(h, n->u.binary.rhs->hash);
    h = hash_mix(h, (uint32_t)n->u.binary.cmp_op);
    break;
  case IXS_PIECEWISE: {
    uint32_t i;
    for (i = 0; i < n->u.pw.ncases; i++) {
      h = hash_mix(h, n->u.pw.cases[i].value->hash);
      h = hash_mix(h, n->u.pw.cases[i].cond->hash);
    }
    break;
  }
  case IXS_AND:
  case IXS_OR: {
    uint32_t i;
    for (i = 0; i < n->u.logic.nargs; i++)
      h = hash_mix(h, n->u.logic.args[i]->hash);
    break;
  }
  case IXS_NOT:
    h = hash_mix(h, n->u.unary_bool.arg->hash);
    break;
  case IXS_TRUE:
    h = hash_mix(h, 1);
    break;
  case IXS_FALSE:
    h = hash_mix(h, 0);
    break;
  case IXS_ERROR:
    h = hash_mix(h, 0xDEAD);
    break;
  case IXS_PARSE_ERROR:
    h = hash_mix(h, 0xBEEF);
    break;
  }
  return h;
}

/* ------------------------------------------------------------------ */
/*  Node equality (structural)                                        */
/* ------------------------------------------------------------------ */

IXS_STATIC bool ixs_node_equal(const ixs_node *a, const ixs_node *b) {
  uint32_t i;
  if (a == b)
    return true;
  if (a->tag != b->tag)
    return false;
  switch (a->tag) {
  case IXS_INT:
    return a->u.ival == b->u.ival;
  case IXS_RAT:
    return a->u.rat.p == b->u.rat.p && a->u.rat.q == b->u.rat.q;
  case IXS_SYM:
    return strcmp(a->u.name, b->u.name) == 0;
  case IXS_ADD:
    if (a->u.add.coeff != b->u.add.coeff)
      return false;
    if (a->u.add.nterms != b->u.add.nterms)
      return false;
    for (i = 0; i < a->u.add.nterms; i++) {
      if (a->u.add.terms[i].term != b->u.add.terms[i].term)
        return false;
      if (a->u.add.terms[i].coeff != b->u.add.terms[i].coeff)
        return false;
    }
    return true;
  case IXS_MUL:
    if (a->u.mul.coeff != b->u.mul.coeff)
      return false;
    if (a->u.mul.nfactors != b->u.mul.nfactors)
      return false;
    for (i = 0; i < a->u.mul.nfactors; i++) {
      if (a->u.mul.factors[i].base != b->u.mul.factors[i].base)
        return false;
      if (a->u.mul.factors[i].exp != b->u.mul.factors[i].exp)
        return false;
    }
    return true;
  case IXS_FLOOR:
  case IXS_CEIL:
    return a->u.unary.arg == b->u.unary.arg;
  case IXS_CMP:
    return a->u.binary.lhs == b->u.binary.lhs &&
           a->u.binary.rhs == b->u.binary.rhs &&
           a->u.binary.cmp_op == b->u.binary.cmp_op;
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
  case IXS_XOR:
    return a->u.binary.lhs == b->u.binary.lhs &&
           a->u.binary.rhs == b->u.binary.rhs;
  case IXS_PIECEWISE:
    if (a->u.pw.ncases != b->u.pw.ncases)
      return false;
    for (i = 0; i < a->u.pw.ncases; i++) {
      if (a->u.pw.cases[i].value != b->u.pw.cases[i].value)
        return false;
      if (a->u.pw.cases[i].cond != b->u.pw.cases[i].cond)
        return false;
    }
    return true;
  case IXS_AND:
  case IXS_OR:
    if (a->u.logic.nargs != b->u.logic.nargs)
      return false;
    for (i = 0; i < a->u.logic.nargs; i++)
      if (a->u.logic.args[i] != b->u.logic.args[i])
        return false;
    return true;
  case IXS_NOT:
    return a->u.unary_bool.arg == b->u.unary_bool.arg;
  case IXS_TRUE:
  case IXS_FALSE:
  case IXS_ERROR:
  case IXS_PARSE_ERROR:
    return true;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/*  Node comparison (total order)                                     */
/* ------------------------------------------------------------------ */

/*
 * Total order on nodes for canonical sorting.  Fast path: hash-consed
 * nodes with identical structure are pointer-equal, caught immediately.
 * Next: compare by precomputed structural hash (O(1), deterministic).
 * Recursive fallback fires only on the rare 32-bit hash collision.
 */
IXS_STATIC int ixs_node_cmp(const ixs_node *a, const ixs_node *b) {
  uint32_t i;
  int c;
  if (a == b)
    return 0;
  if ((int)a->tag != (int)b->tag)
    return (int)a->tag < (int)b->tag ? -1 : 1;

  switch (a->tag) {
  case IXS_INT:
    if (a->u.ival < b->u.ival)
      return -1;
    if (a->u.ival > b->u.ival)
      return 1;
    return 0;
  case IXS_RAT:
    return ixs_rat_cmp(a->u.rat.p, a->u.rat.q, b->u.rat.p, b->u.rat.q);
  case IXS_SYM:
    return strcmp(a->u.name, b->u.name);
  case IXS_ADD:
    c = ixs_node_cmp(a->u.add.coeff, b->u.add.coeff);
    if (c)
      return c;
    if (a->u.add.nterms != b->u.add.nterms)
      return a->u.add.nterms < b->u.add.nterms ? -1 : 1;
    for (i = 0; i < a->u.add.nterms; i++) {
      c = ixs_node_cmp(a->u.add.terms[i].term, b->u.add.terms[i].term);
      if (c)
        return c;
      c = ixs_node_cmp(a->u.add.terms[i].coeff, b->u.add.terms[i].coeff);
      if (c)
        return c;
    }
    return 0;
  case IXS_MUL:
    c = ixs_node_cmp(a->u.mul.coeff, b->u.mul.coeff);
    if (c)
      return c;
    if (a->u.mul.nfactors != b->u.mul.nfactors)
      return a->u.mul.nfactors < b->u.mul.nfactors ? -1 : 1;
    for (i = 0; i < a->u.mul.nfactors; i++) {
      c = ixs_node_cmp(a->u.mul.factors[i].base, b->u.mul.factors[i].base);
      if (c)
        return c;
      if (a->u.mul.factors[i].exp != b->u.mul.factors[i].exp)
        return a->u.mul.factors[i].exp < b->u.mul.factors[i].exp ? -1 : 1;
    }
    return 0;
  case IXS_FLOOR:
  case IXS_CEIL:
    return ixs_node_cmp(a->u.unary.arg, b->u.unary.arg);
  case IXS_CMP:
    if (a->u.binary.cmp_op != b->u.binary.cmp_op)
      return (int)a->u.binary.cmp_op < (int)b->u.binary.cmp_op ? -1 : 1;
    /* fallthrough */
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
  case IXS_XOR:
    c = ixs_node_cmp(a->u.binary.lhs, b->u.binary.lhs);
    if (c)
      return c;
    return ixs_node_cmp(a->u.binary.rhs, b->u.binary.rhs);
  case IXS_PIECEWISE:
    if (a->u.pw.ncases != b->u.pw.ncases)
      return a->u.pw.ncases < b->u.pw.ncases ? -1 : 1;
    for (i = 0; i < a->u.pw.ncases; i++) {
      c = ixs_node_cmp(a->u.pw.cases[i].value, b->u.pw.cases[i].value);
      if (c)
        return c;
      c = ixs_node_cmp(a->u.pw.cases[i].cond, b->u.pw.cases[i].cond);
      if (c)
        return c;
    }
    return 0;
  case IXS_AND:
  case IXS_OR:
    if (a->u.logic.nargs != b->u.logic.nargs)
      return a->u.logic.nargs < b->u.logic.nargs ? -1 : 1;
    for (i = 0; i < a->u.logic.nargs; i++) {
      c = ixs_node_cmp(a->u.logic.args[i], b->u.logic.args[i]);
      if (c)
        return c;
    }
    return 0;
  case IXS_NOT:
    return ixs_node_cmp(a->u.unary_bool.arg, b->u.unary_bool.arg);
  case IXS_TRUE:
  case IXS_FALSE:
  case IXS_ERROR:
  case IXS_PARSE_ERROR:
    return 0;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/*  Hash-consing table                                                */
/* ------------------------------------------------------------------ */

IXS_STATIC bool ixs_htab_init(ixs_ctx *ctx) {
  ctx->htab_cap = IXS_HTAB_INIT_CAP;
  ctx->htab_used = 0;
  ctx->htab = calloc(ctx->htab_cap, sizeof(ixs_node *));
  return ctx->htab != NULL;
}

IXS_STATIC void ixs_htab_destroy(ixs_ctx *ctx) {
  free(ctx->htab);
  ctx->htab = NULL;
}

static bool htab_rehash(ixs_ctx *ctx) {
  size_t new_cap = ctx->htab_cap * 2;
  if (new_cap < ctx->htab_cap)
    return false;

  ixs_node **new_buckets = calloc(new_cap, sizeof(ixs_node *));
  if (!new_buckets)
    return false;

  size_t mask = new_cap - 1;
  size_t i;
  for (i = 0; i < ctx->htab_cap; i++) {
    ixs_node *n = ctx->htab[i];
    if (!n)
      continue;
    size_t idx = n->hash & mask;
    while (new_buckets[idx])
      idx = (idx + 1) & mask;
    new_buckets[idx] = n;
  }

  free(ctx->htab);
  ctx->htab = new_buckets;
  ctx->htab_cap = new_cap;
  return true;
}

/* Single probe loop shared by lookup and intern.  Returns the index of
 * either the matching slot or the first empty slot.  Sets *found to the
 * matching node, or NULL if the slot is empty. */
static size_t htab_find_slot(ixs_ctx *ctx, const ixs_node *probe,
                             ixs_node **found) {
  size_t mask = ctx->htab_cap - 1;
  size_t idx = probe->hash & mask;
  for (;;) {
    ixs_node *slot = ctx->htab[idx];
    if (!slot) {
      *found = NULL;
      return idx;
    }
    if (slot->hash == probe->hash && ixs_node_equal(slot, probe)) {
      *found = slot;
      return idx;
    }
    idx = (idx + 1) & mask;
  }
}

static ixs_node *htab_lookup(ixs_ctx *ctx, const ixs_node *probe) {
  ixs_node *found;
  htab_find_slot(ctx, probe, &found);
  return found;
}

IXS_STATIC ixs_node *ixs_htab_intern(ixs_ctx *ctx, ixs_node *node) {
  ixs_node *found;
  size_t idx = htab_find_slot(ctx, node, &found);
  if (found)
    return found;
  ctx->htab[idx] = node;
  ctx->htab_used++;
  if (ctx->htab_used * IXS_HTAB_LOAD_DEN > ctx->htab_cap * IXS_HTAB_LOAD_NUM) {
    if (!htab_rehash(ctx))
      return NULL;
  }
  return node;
}

/* ------------------------------------------------------------------ */
/*  Arena allocation helpers                                          */
/* ------------------------------------------------------------------ */

static ixs_node *alloc_node(ixs_ctx *ctx) {
  ixs_node *n = ixs_arena_alloc(&ctx->arena, sizeof(ixs_node), sizeof(void *));
  if (n)
    memset(n, 0, sizeof(*n));
  return n;
}

/* ------------------------------------------------------------------ */
/*  Raw node constructors                                             */
/* ------------------------------------------------------------------ */

IXS_STATIC ixs_node *ixs_node_int(ixs_ctx *ctx, int64_t val) {
  ixs_node tmp;
  ixs_node *found, *n;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_INT;
  tmp.u.ival = val;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_rat(ixs_ctx *ctx, int64_t p, int64_t q) {
  ixs_node tmp;
  ixs_node *found, *n;
  if (q == 1)
    return ixs_node_int(ctx, p);

  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_RAT;
  tmp.u.rat.p = p;
  tmp.u.rat.q = q;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

/* Cannot use htab_lookup: name may not be NUL-terminated, and
 * ixs_node_equal uses strcmp, so we probe with memcmp directly. */
IXS_STATIC ixs_node *ixs_node_sym(ixs_ctx *ctx, const char *name, size_t len) {
  uint32_t sym_hash;
  ixs_node *n;
  char *interned;

  /* Compute hash from the bounded slice — name may not be NUL-terminated. */
  sym_hash = (uint32_t)IXS_SYM * 2654435761u;
  sym_hash = hash_mix(sym_hash, hash_str(name, len));

  {
    size_t mask = ctx->htab_cap - 1;
    size_t idx = sym_hash & mask;
    for (;;) {
      ixs_node *slot = ctx->htab[idx];
      if (!slot)
        break;
      if (slot->hash == sym_hash && slot->tag == IXS_SYM &&
          strlen(slot->u.name) == len && memcmp(slot->u.name, name, len) == 0)
        return slot;
      idx = (idx + 1) & mask;
    }
  }

  interned = ixs_arena_strdup(&ctx->arena, name, len);
  if (!interned)
    return NULL;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  n->tag = IXS_SYM;
  n->u.name = interned;
  n->hash = sym_hash;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_add(ixs_ctx *ctx, ixs_node *coeff,
                                  uint32_t nterms, ixs_addterm *terms) {
  ixs_node tmp;
  ixs_node *found, *n;
  ixs_addterm *a;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_ADD;
  tmp.u.add.coeff = coeff;
  tmp.u.add.nterms = nterms;
  tmp.u.add.terms = terms;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  a = NULL;
  if (nterms > 0) {
    size_t sz = (size_t)nterms * sizeof(ixs_addterm);
    if (sz / sizeof(ixs_addterm) != nterms)
      return NULL;
    a = ixs_arena_alloc(&ctx->arena, sz, sizeof(void *));
    if (!a)
      return NULL;
    memcpy(a, terms, sz);
  }

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  tmp.u.add.terms = a;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_mul(ixs_ctx *ctx, ixs_node *coeff,
                                  uint32_t nfactors, ixs_mulfactor *factors) {
  ixs_node tmp;
  ixs_node *found, *n;
  ixs_mulfactor *f;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_MUL;
  tmp.u.mul.coeff = coeff;
  tmp.u.mul.nfactors = nfactors;
  tmp.u.mul.factors = factors;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  f = NULL;
  if (nfactors > 0) {
    size_t sz = (size_t)nfactors * sizeof(ixs_mulfactor);
    if (sz / sizeof(ixs_mulfactor) != nfactors)
      return NULL;
    f = ixs_arena_alloc(&ctx->arena, sz, sizeof(void *));
    if (!f)
      return NULL;
    memcpy(f, factors, sz);
  }

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  tmp.u.mul.factors = f;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_floor(ixs_ctx *ctx, ixs_node *arg) {
  ixs_node tmp;
  ixs_node *found, *n;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_FLOOR;
  tmp.u.unary.arg = arg;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_ceil(ixs_ctx *ctx, ixs_node *arg) {
  ixs_node tmp;
  ixs_node *found, *n;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_CEIL;
  tmp.u.unary.arg = arg;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_binary(ixs_ctx *ctx, ixs_tag tag, ixs_node *lhs,
                                     ixs_node *rhs, ixs_cmp_op op) {
  ixs_node tmp;
  ixs_node *found, *n;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = tag;
  tmp.u.binary.lhs = lhs;
  tmp.u.binary.rhs = rhs;
  tmp.u.binary.cmp_op = op;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_pw(ixs_ctx *ctx, uint32_t ncases,
                                 ixs_pwcase *cases) {
  ixs_node tmp;
  ixs_node *found, *n;
  ixs_pwcase *c;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_PIECEWISE;
  tmp.u.pw.ncases = ncases;
  tmp.u.pw.cases = cases;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  c = NULL;
  if (ncases > 0) {
    size_t sz = (size_t)ncases * sizeof(ixs_pwcase);
    if (sz / sizeof(ixs_pwcase) != ncases)
      return NULL;
    c = ixs_arena_alloc(&ctx->arena, sz, sizeof(void *));
    if (!c)
      return NULL;
    memcpy(c, cases, sz);
  }

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  tmp.u.pw.cases = c;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_logic(ixs_ctx *ctx, ixs_tag tag, uint32_t nargs,
                                    ixs_node **args) {
  ixs_node tmp;
  ixs_node *found, *n;
  ixs_node **a;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = tag;
  tmp.u.logic.nargs = nargs;
  tmp.u.logic.args = args;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  a = NULL;
  if (nargs > 0) {
    size_t sz = (size_t)nargs * sizeof(ixs_node *);
    if (sz / sizeof(ixs_node *) != nargs)
      return NULL;
    a = ixs_arena_alloc(&ctx->arena, sz, sizeof(void *));
    if (!a)
      return NULL;
    memcpy(a, args, sz);
  }

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  tmp.u.logic.args = a;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

IXS_STATIC ixs_node *ixs_node_not(ixs_ctx *ctx, ixs_node *arg) {
  ixs_node tmp;
  ixs_node *found, *n;
  memset(&tmp, 0, sizeof(tmp));
  tmp.tag = IXS_NOT;
  tmp.u.unary_bool.arg = arg;
  tmp.hash = compute_hash(&tmp);

  found = htab_lookup(ctx, &tmp);
  if (found)
    return found;

  n = alloc_node(ctx);
  if (!n)
    return NULL;
  *n = tmp;
  return ixs_htab_intern(ctx, n);
}

/* ------------------------------------------------------------------ */
/*  Utilities                                                         */
/* ------------------------------------------------------------------ */

IXS_STATIC bool ixs_node_is_const(const ixs_node *n) {
  return n->tag == IXS_INT || n->tag == IXS_RAT;
}

IXS_STATIC bool ixs_node_is_zero(const ixs_node *n) {
  return n->tag == IXS_INT && n->u.ival == 0;
}

IXS_STATIC bool ixs_node_is_one(const ixs_node *n) {
  return n->tag == IXS_INT && n->u.ival == 1;
}

IXS_STATIC void ixs_node_get_rat(const ixs_node *n, int64_t *p, int64_t *q) {
  if (n->tag == IXS_INT) {
    *p = n->u.ival;
    *q = 1;
  } else if (n->tag == IXS_RAT) {
    *p = n->u.rat.p;
    *q = n->u.rat.q;
  } else {
    *p = 0;
    *q = 1;
  }
}

IXS_STATIC bool ixs_node_is_sentinel(const ixs_node *n) {
  return n->tag == IXS_ERROR || n->tag == IXS_PARSE_ERROR;
}

/* ------------------------------------------------------------------ */
/*  Error list                                                        */
/* ------------------------------------------------------------------ */

IXS_STATIC void ixs_ctx_push_error(ixs_ctx *ctx, const char *fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  char *msg = ixs_arena_strdup(&ctx->arena, buf, strlen(buf));
  if (!msg)
    return;

  if (ctx->nerrors >= ctx->errors_cap) {
    size_t new_cap = ctx->errors_cap ? ctx->errors_cap * 2 : 16;
    if (new_cap <= ctx->errors_cap ||
        new_cap > (size_t)-1 / sizeof(const char *))
      return;
    const char **new_arr =
        ixs_arena_grow(&ctx->arena, (void *)ctx->errors,
                       ctx->errors_cap * sizeof(const char *),
                       new_cap * sizeof(const char *), sizeof(void *));
    if (!new_arr)
      return;
    ctx->errors = new_arr;
    ctx->errors_cap = new_cap;
  }
  ctx->errors[ctx->nerrors++] = msg;
}

/* ------------------------------------------------------------------ */
/*  NULL / sentinel propagation                                       */
/* ------------------------------------------------------------------ */

IXS_STATIC ixs_node *ixs_propagate1(ixs_node *a) {
  if (!a)
    return NULL;
  if (ixs_node_is_sentinel(a))
    return a;
  return NULL; /* clean */
}

/*
 * If either arg is a sentinel (ERROR or PARSE_ERROR), return it so
 * the caller can short-circuit.  NULL (OOM) returns NULL — callers
 * propagate that by their own NULL return.  Both clean → NULL.
 */
IXS_STATIC ixs_node *ixs_propagate2(ixs_node *a, ixs_node *b) {
  if (!a || !b)
    return NULL; /* OOM propagation — caller returns NULL */
  if (a->tag == IXS_PARSE_ERROR || b->tag == IXS_PARSE_ERROR)
    return a->tag == IXS_PARSE_ERROR ? a : b;
  if (a->tag == IXS_ERROR)
    return a;
  if (b->tag == IXS_ERROR)
    return b;
  return NULL; /* both clean */
}

/* ------------------------------------------------------------------ */
/*  Integer-valued predicate                                          */
/* ------------------------------------------------------------------ */

IXS_STATIC bool ixs_node_is_integer_valued(const ixs_node *n) {
  if (!n)
    return false;
  switch (n->tag) {
  case IXS_INT:
  case IXS_FLOOR:
  case IXS_CEIL:
  case IXS_SYM:
  case IXS_XOR:
    return true;
  case IXS_ADD: {
    uint32_t i;
    int64_t cp, cq;
    ixs_node_get_rat(n->u.add.coeff, &cp, &cq);
    if (cq != 1)
      return false;
    for (i = 0; i < n->u.add.nterms; i++) {
      ixs_node_get_rat(n->u.add.terms[i].coeff, &cp, &cq);
      if (cq != 1)
        return false;
      if (!ixs_node_is_integer_valued(n->u.add.terms[i].term))
        return false;
    }
    return true;
  }
  case IXS_MUL: {
    uint32_t i;
    int64_t cp, cq;
    ixs_node_get_rat(n->u.mul.coeff, &cp, &cq);
    if (cq != 1)
      return false;
    for (i = 0; i < n->u.mul.nfactors; i++) {
      if (n->u.mul.factors[i].exp < 0)
        return false;
      if (!ixs_node_is_integer_valued(n->u.mul.factors[i].base))
        return false;
    }
    return true;
  }
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
    return ixs_node_is_integer_valued(n->u.binary.lhs) &&
           ixs_node_is_integer_valued(n->u.binary.rhs);
  case IXS_PIECEWISE: {
    uint32_t i;
    for (i = 0; i < n->u.pw.ncases; i++) {
      if (!ixs_node_is_integer_valued(n->u.pw.cases[i].value))
        return false;
    }
    return n->u.pw.ncases > 0;
  }
  default:
    return false;
  }
}

/* ------------------------------------------------------------------ */
/*  Type-specific accessors                                           */
/* ------------------------------------------------------------------ */

int64_t ixs_node_rat_num(ixs_node *node) {
  assert(node && node->tag == IXS_RAT);
  return node->u.rat.p;
}

int64_t ixs_node_rat_den(ixs_node *node) {
  assert(node && node->tag == IXS_RAT);
  return node->u.rat.q;
}

const char *ixs_node_sym_name(ixs_node *node) {
  assert(node && node->tag == IXS_SYM);
  return node->u.name;
}

ixs_node *ixs_node_add_coeff(ixs_node *node) {
  assert(node && node->tag == IXS_ADD);
  return node->u.add.coeff;
}

uint32_t ixs_node_add_nterms(ixs_node *node) {
  assert(node && node->tag == IXS_ADD);
  return node->u.add.nterms;
}

ixs_node *ixs_node_add_term(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_ADD && i < node->u.add.nterms);
  return node->u.add.terms[i].term;
}

ixs_node *ixs_node_add_term_coeff(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_ADD && i < node->u.add.nterms);
  return node->u.add.terms[i].coeff;
}

ixs_node *ixs_node_mul_coeff(ixs_node *node) {
  assert(node && node->tag == IXS_MUL);
  return node->u.mul.coeff;
}

uint32_t ixs_node_mul_nfactors(ixs_node *node) {
  assert(node && node->tag == IXS_MUL);
  return node->u.mul.nfactors;
}

ixs_node *ixs_node_mul_factor_base(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_MUL && i < node->u.mul.nfactors);
  return node->u.mul.factors[i].base;
}

int32_t ixs_node_mul_factor_exp(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_MUL && i < node->u.mul.nfactors);
  return node->u.mul.factors[i].exp;
}

ixs_node *ixs_node_unary_arg(ixs_node *node) {
  assert(node && (node->tag == IXS_FLOOR || node->tag == IXS_CEIL ||
                  node->tag == IXS_NOT));
  if (node->tag == IXS_NOT)
    return node->u.unary_bool.arg;
  return node->u.unary.arg;
}

ixs_node *ixs_node_binary_lhs(ixs_node *node) {
  assert(node && (node->tag == IXS_MOD || node->tag == IXS_MAX ||
                  node->tag == IXS_MIN || node->tag == IXS_XOR ||
                  node->tag == IXS_CMP));
  return node->u.binary.lhs;
}

ixs_node *ixs_node_binary_rhs(ixs_node *node) {
  assert(node && (node->tag == IXS_MOD || node->tag == IXS_MAX ||
                  node->tag == IXS_MIN || node->tag == IXS_XOR ||
                  node->tag == IXS_CMP));
  return node->u.binary.rhs;
}

ixs_cmp_op ixs_node_cmp_op(ixs_node *node) {
  assert(node && node->tag == IXS_CMP);
  return node->u.binary.cmp_op;
}

uint32_t ixs_node_pw_ncases(ixs_node *node) {
  assert(node && node->tag == IXS_PIECEWISE);
  return node->u.pw.ncases;
}

ixs_node *ixs_node_pw_value(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_PIECEWISE && i < node->u.pw.ncases);
  return node->u.pw.cases[i].value;
}

ixs_node *ixs_node_pw_cond(ixs_node *node, uint32_t i) {
  assert(node && node->tag == IXS_PIECEWISE && i < node->u.pw.ncases);
  return node->u.pw.cases[i].cond;
}

uint32_t ixs_node_logic_nargs(ixs_node *node) {
  assert(node && (node->tag == IXS_AND || node->tag == IXS_OR));
  return node->u.logic.nargs;
}

ixs_node *ixs_node_logic_arg(ixs_node *node, uint32_t i) {
  assert(node && (node->tag == IXS_AND || node->tag == IXS_OR) &&
         i < node->u.logic.nargs);
  return node->u.logic.args[i];
}

/* ------------------------------------------------------------------ */
/*  Generic child access                                              */
/* ------------------------------------------------------------------ */

uint32_t ixs_node_nchildren(ixs_node *node) {
  assert(node);
  switch (node->tag) {
  case IXS_ADD:
    return 1 + 2 * node->u.add.nterms;
  case IXS_MUL:
    return 1 + node->u.mul.nfactors;
  case IXS_FLOOR:
  case IXS_CEIL:
  case IXS_NOT:
    return 1;
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
  case IXS_XOR:
  case IXS_CMP:
    return 2;
  case IXS_PIECEWISE:
    return 2 * node->u.pw.ncases;
  case IXS_AND:
  case IXS_OR:
    return node->u.logic.nargs;
  default:
    return 0;
  }
}

ixs_node *ixs_node_child(ixs_node *node, uint32_t i) {
  assert(node && i < ixs_node_nchildren(node));
  switch (node->tag) {
  case IXS_ADD:
    if (i == 0)
      return node->u.add.coeff;
    {
      uint32_t j = i - 1;
      if (j % 2 == 0)
        return node->u.add.terms[j / 2].coeff;
      return node->u.add.terms[j / 2].term;
    }
  case IXS_MUL:
    if (i == 0)
      return node->u.mul.coeff;
    return node->u.mul.factors[i - 1].base;
  case IXS_FLOOR:
  case IXS_CEIL:
    return node->u.unary.arg;
  case IXS_NOT:
    return node->u.unary_bool.arg;
  case IXS_MOD:
  case IXS_MAX:
  case IXS_MIN:
  case IXS_XOR:
  case IXS_CMP:
    return i == 0 ? node->u.binary.lhs : node->u.binary.rhs;
  case IXS_PIECEWISE:
    if (i % 2 == 0)
      return node->u.pw.cases[i / 2].value;
    return node->u.pw.cases[i / 2].cond;
  case IXS_AND:
  case IXS_OR:
    return node->u.logic.args[i];
  default:
    return NULL;
  }
}
