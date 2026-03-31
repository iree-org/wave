/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "node.h"

static ixs_node *walk_pre(ixs_node *node, ixs_visit_fn fn, void *ud) {
  ixs_walk_action act = fn(node, ud);
  if (act == IXS_WALK_STOP)
    return node;
  if (act == IXS_WALK_SKIP)
    return NULL;

  uint32_t n = ixs_node_nchildren(node);
  uint32_t i;
  for (i = 0; i < n; i++) {
    ixs_node *stopped = walk_pre(ixs_node_child(node, i), fn, ud);
    if (stopped)
      return stopped;
  }
  return NULL;
}

static ixs_node *walk_post(ixs_node *node, ixs_visit_fn fn, void *ud) {
  uint32_t n = ixs_node_nchildren(node);
  uint32_t i;
  for (i = 0; i < n; i++) {
    ixs_node *stopped = walk_post(ixs_node_child(node, i), fn, ud);
    if (stopped)
      return stopped;
  }

  ixs_walk_action act = fn(node, ud);
  if (act == IXS_WALK_STOP)
    return node;
  return NULL;
}

ixs_node *ixs_walk_pre(ixs_ctx *ctx, ixs_node *root, ixs_visit_fn fn,
                       void *userdata) {
  (void)ctx;
  if (!root)
    return NULL;
  ixs_node *stopped = walk_pre(root, fn, userdata);
  return stopped ? stopped : root;
}

ixs_node *ixs_walk_post(ixs_ctx *ctx, ixs_node *root, ixs_visit_fn fn,
                        void *userdata) {
  (void)ctx;
  if (!root)
    return NULL;
  ixs_node *stopped = walk_post(root, fn, userdata);
  return stopped ? stopped : root;
}
