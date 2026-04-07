/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "print.h"
#include <stdio.h>
#include <string.h>

/*
 * snprintf-style accumulator: tracks position and remaining space.
 */
typedef struct {
  char *buf;
  size_t size;
  size_t pos; /* total chars written (may exceed size) */
} printbuf;

static void pb_init(printbuf *pb, char *buf, size_t size) {
  pb->buf = buf;
  pb->size = size;
  pb->pos = 0;
}

static void pb_write(printbuf *pb, const char *s, size_t len) {
  if (pb->buf && pb->pos < pb->size) {
    size_t avail = pb->size - pb->pos;
    size_t n = len < avail ? len : avail;
    memcpy(pb->buf + pb->pos, s, n);
  }
  pb->pos += len;
}

static void pb_str(printbuf *pb, const char *s) { pb_write(pb, s, strlen(s)); }

static void pb_char(printbuf *pb, char c) { pb_write(pb, &c, 1); }

static void pb_i64(printbuf *pb, int64_t v) {
  char tmp[32];
  int n = snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
  pb_write(pb, tmp, (size_t)n);
}

static void pb_finish(printbuf *pb) {
  if (pb->buf && pb->size > 0) {
    size_t end = pb->pos < pb->size ? pb->pos : pb->size - 1;
    pb->buf[end] = '\0';
  }
}

/* Precedence levels for minimal parenthesization. */
typedef enum {
  PREC_ATOM = 0,
  PREC_MUL = 1,
  PREC_ADD = 2,
  PREC_CMP = 3,
  PREC_NOT = 4,
  PREC_AND = 5,
  PREC_OR = 6,
  PREC_TOP = 7
} prec_t;

static prec_t node_prec(ixs_node *n) {
  switch (n->tag) {
  case IXS_ADD:
    return PREC_ADD;
  case IXS_MUL:
    return PREC_MUL;
  case IXS_CMP:
    return PREC_CMP;
  case IXS_AND:
    return PREC_AND;
  case IXS_OR:
    return PREC_OR;
  case IXS_NOT:
    return PREC_NOT;
  default:
    return PREC_ATOM;
  }
}

static void print_node(printbuf *pb, ixs_node *n, prec_t parent_prec);

static void print_wrapped(printbuf *pb, ixs_node *n, prec_t parent_prec) {
  prec_t my = node_prec(n);
  if (my > parent_prec) {
    pb_char(pb, '(');
    print_node(pb, n, PREC_TOP);
    pb_char(pb, ')');
  } else {
    print_node(pb, n, parent_prec);
  }
}

static void print_add(printbuf *pb, ixs_node *n) {
  uint32_t i;
  int64_t cp, cq;
  bool first = true;

  ixs_node_get_rat(n->u.add.coeff, &cp, &cq);
  if (!ixs_rat_is_zero(cp)) {
    print_node(pb, n->u.add.coeff, PREC_ADD);
    first = false;
  }

  for (i = 0; i < n->u.add.nterms; i++) {
    int64_t tp, tq;
    ixs_node_get_rat(n->u.add.terms[i].coeff, &tp, &tq);

    if (first) {
      if (tp == -1 && tq == 1) {
        pb_str(pb, "-");
        print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
      } else if (tp == 1 && tq == 1) {
        print_wrapped(pb, n->u.add.terms[i].term, PREC_ADD);
      } else if (ixs_rat_is_neg(tp)) {
        pb_str(pb, "-");
        int64_t np = 0, nq = 0;
        ixs_rat_neg(tp, tq, &np, &nq);
        if (np == 1 && nq == 1) {
          print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
        } else {
          ixs_node tmp;
          memset(&tmp, 0, sizeof(tmp));
          if (nq == 1) {
            tmp.tag = IXS_INT;
            tmp.u.ival = np;
          } else {
            tmp.tag = IXS_RAT;
            tmp.u.rat.p = np;
            tmp.u.rat.q = nq;
          }
          print_node(pb, &tmp, PREC_MUL);
          pb_char(pb, '*');
          print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
        }
      } else {
        if (!(tp == 1 && tq == 1)) {
          print_node(pb, n->u.add.terms[i].coeff, PREC_MUL);
          pb_char(pb, '*');
        }
        print_wrapped(pb, n->u.add.terms[i].term, PREC_ADD);
      }
      first = false;
    } else {
      if (ixs_rat_is_neg(tp)) {
        pb_str(pb, " - ");
        int64_t np = 0, nq = 0;
        ixs_rat_neg(tp, tq, &np, &nq);
        if (np == 1 && nq == 1) {
          print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
        } else {
          ixs_node tmp;
          memset(&tmp, 0, sizeof(tmp));
          if (nq == 1) {
            tmp.tag = IXS_INT;
            tmp.u.ival = np;
          } else {
            tmp.tag = IXS_RAT;
            tmp.u.rat.p = np;
            tmp.u.rat.q = nq;
          }
          print_node(pb, &tmp, PREC_MUL);
          pb_char(pb, '*');
          print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
        }
      } else {
        pb_str(pb, " + ");
        if (!(tp == 1 && tq == 1)) {
          print_node(pb, n->u.add.terms[i].coeff, PREC_MUL);
          pb_char(pb, '*');
        }
        print_wrapped(pb, n->u.add.terms[i].term, PREC_MUL);
      }
    }
  }
  if (first)
    pb_str(pb, "0");
}

static void print_node(printbuf *pb, ixs_node *n, prec_t parent_prec) {
  uint32_t i;
  (void)parent_prec;

  if (!n) {
    pb_str(pb, "<null>");
    return;
  }

  switch (n->tag) {
  case IXS_INT:
    pb_i64(pb, n->u.ival);
    break;

  case IXS_RAT:
    pb_i64(pb, n->u.rat.p);
    pb_char(pb, '/');
    pb_i64(pb, n->u.rat.q);
    break;

  case IXS_SYM:
    pb_str(pb, n->u.name);
    break;

  case IXS_ADD:
    print_add(pb, n);
    break;

  case IXS_MUL: {
    int64_t cp, cq;
    ixs_node_get_rat(n->u.mul.coeff, &cp, &cq);
    bool need_sep = false;

    if (cp == -1 && cq == 1) {
      pb_str(pb, "-");
    } else if (!(cp == 1 && cq == 1)) {
      print_node(pb, n->u.mul.coeff, PREC_MUL);
      need_sep = true;
    }

    for (i = 0; i < n->u.mul.nfactors; i++) {
      if (need_sep)
        pb_char(pb, '*');
      if (n->u.mul.factors[i].exp == 1) {
        print_wrapped(pb, n->u.mul.factors[i].base, PREC_MUL);
      } else if (n->u.mul.factors[i].exp == -1) {
        pb_str(pb, "1/");
        print_wrapped(pb, n->u.mul.factors[i].base, PREC_MUL);
      } else {
        print_wrapped(pb, n->u.mul.factors[i].base, PREC_ATOM);
        pb_str(pb, "**");
        pb_i64(pb, n->u.mul.factors[i].exp);
      }
      need_sep = true;
    }
    break;
  }

  case IXS_FLOOR:
    pb_str(pb, "floor(");
    print_node(pb, n->u.unary.arg, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_CEIL:
    pb_str(pb, "ceiling(");
    print_node(pb, n->u.unary.arg, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_MOD:
    pb_str(pb, "Mod(");
    print_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_MAX:
    pb_str(pb, "Max(");
    print_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_MIN:
    pb_str(pb, "Min(");
    print_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_XOR:
    pb_str(pb, "xor(");
    print_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;

  case IXS_CMP: {
    const char *opstr = "??";
    switch (n->u.binary.cmp_op) {
    case IXS_CMP_GT:
      opstr = " > ";
      break;
    case IXS_CMP_GE:
      opstr = " >= ";
      break;
    case IXS_CMP_LT:
      opstr = " < ";
      break;
    case IXS_CMP_LE:
      opstr = " <= ";
      break;
    case IXS_CMP_EQ:
      opstr = " == ";
      break;
    case IXS_CMP_NE:
      opstr = " != ";
      break;
    }
    print_wrapped(pb, n->u.binary.lhs, PREC_CMP);
    pb_str(pb, opstr);
    print_wrapped(pb, n->u.binary.rhs, PREC_CMP);
    break;
  }

  case IXS_PIECEWISE:
    pb_str(pb, "Piecewise(");
    for (i = 0; i < n->u.pw.ncases; i++) {
      if (i > 0)
        pb_str(pb, ", ");
      pb_char(pb, '(');
      print_node(pb, n->u.pw.cases[i].value, PREC_TOP);
      pb_str(pb, ", ");
      print_node(pb, n->u.pw.cases[i].cond, PREC_TOP);
      pb_char(pb, ')');
    }
    pb_char(pb, ')');
    break;

  case IXS_AND:
    for (i = 0; i < n->u.logic.nargs; i++) {
      if (i > 0)
        pb_str(pb, " & ");
      print_wrapped(pb, n->u.logic.args[i], PREC_AND);
    }
    break;

  case IXS_OR:
    for (i = 0; i < n->u.logic.nargs; i++) {
      if (i > 0)
        pb_str(pb, " | ");
      print_wrapped(pb, n->u.logic.args[i], PREC_OR);
    }
    break;

  case IXS_NOT:
    pb_str(pb, "~");
    print_wrapped(pb, n->u.unary_bool.arg, PREC_NOT);
    break;

  case IXS_TRUE:
    pb_str(pb, "True");
    break;

  case IXS_FALSE:
    pb_str(pb, "False");
    break;

  case IXS_ERROR:
  case IXS_PARSE_ERROR:
    pb_str(pb, "<error>");
    break;
  }
}

IXS_STATIC size_t ixs_print_impl(ixs_node *expr, char *buf, size_t bufsize) {
  printbuf pb;
  if (!expr) {
    if (buf && bufsize > 0)
      buf[0] = '\0';
    return 0;
  }
  pb_init(&pb, buf, bufsize);
  print_node(&pb, expr, PREC_TOP);
  pb_finish(&pb);
  return pb.pos;
}

/* ------------------------------------------------------------------ */
/*  C output mode                                                     */
/* ------------------------------------------------------------------ */

static void print_c_node(printbuf *pb, ixs_node *n, prec_t parent_prec);

static void print_c_wrapped(printbuf *pb, ixs_node *n, prec_t parent_prec) {
  prec_t my = node_prec(n);
  if (my > parent_prec) {
    pb_char(pb, '(');
    print_c_node(pb, n, PREC_TOP);
    pb_char(pb, ')');
  } else {
    print_c_node(pb, n, parent_prec);
  }
}

static void print_c_node(printbuf *pb, ixs_node *n, prec_t parent_prec) {
  (void)parent_prec;

  if (!n) {
    pb_str(pb, "/*null*/0");
    return;
  }

  switch (n->tag) {
  case IXS_INT:
    pb_i64(pb, n->u.ival);
    break;
  case IXS_RAT: {
    char tmp[64];
    int len = snprintf(tmp, sizeof(tmp), "(%lld.0/%lld.0)",
                       (long long)n->u.rat.p, (long long)n->u.rat.q);
    pb_write(pb, tmp, (size_t)len);
    break;
  }
  case IXS_SYM:
    pb_str(pb, n->u.name);
    break;
  case IXS_FLOOR:
    pb_str(pb, "ixs_floor_i(");
    print_c_node(pb, n->u.unary.arg, PREC_TOP);
    pb_char(pb, ')');
    break;
  case IXS_CEIL:
    pb_str(pb, "ixs_ceil_i(");
    print_c_node(pb, n->u.unary.arg, PREC_TOP);
    pb_char(pb, ')');
    break;
  case IXS_MOD:
    pb_str(pb, "ixs_mod_i(");
    print_c_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_c_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;
  case IXS_MAX:
    pb_str(pb, "ixs_max_i(");
    print_c_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_c_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;
  case IXS_MIN:
    pb_str(pb, "ixs_min_i(");
    print_c_node(pb, n->u.binary.lhs, PREC_TOP);
    pb_str(pb, ", ");
    print_c_node(pb, n->u.binary.rhs, PREC_TOP);
    pb_char(pb, ')');
    break;
  case IXS_XOR:
    pb_char(pb, '(');
    print_c_wrapped(pb, n->u.binary.lhs, PREC_ATOM);
    pb_str(pb, " ^ ");
    print_c_wrapped(pb, n->u.binary.rhs, PREC_ATOM);
    pb_char(pb, ')');
    break;
  default:
    /* Fall back to SymPy format for complex nodes. */
    print_node(pb, n, parent_prec);
    break;
  }
}

IXS_STATIC size_t ixs_print_c_impl(ixs_node *expr, char *buf, size_t bufsize) {
  printbuf pb;
  if (!expr) {
    if (buf && bufsize > 0)
      buf[0] = '\0';
    return 0;
  }
  pb_init(&pb, buf, bufsize);
  print_c_node(&pb, expr, PREC_TOP);
  pb_finish(&pb);
  return pb.pos;
}
