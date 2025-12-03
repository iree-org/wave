"""
Utilities to persist non-core torch.fx.Node attributes across process boundaries.

- torch.fx defines custom pickling for Graph/Node that round-trips a canonical
  subset of fields (op, target, args, kwargs, name) and the metadata dict
  (node.meta). Arbitrary attributes we attach to node instances (e.g., node.type)
  during our compilation are not preserved.
- We leverage node.meta as the supported extension point: before pickling, we
  snapshot additional attributes into node.meta.
  After unpickling, we restore them back onto the node.

Alternative approaches considered:
- Monkey-patching fx's Node/Graph picklers to include custom fields is fragile.
  Using the meta field avoids patching FX internals and scales as we
  add more attributes without further code changes here.
"""

import torch.fx as fx


SNAPSHOT_KEY = "__wave_snapshot__"

CORE_NODE_FIELDS = {
    "op",
    "target",
    "args",
    "kwargs",
    "name",
    "meta",
    "users",
    "graph",
    "_erased",
    "_prev",
    "_next",
}


def snapshot_node_fields(node: fx.Node) -> None:
    """Snapshot non-core fx.Node attributes into node.meta[SNAPSHOT_KEY].

    - Stores a dict of attribute_name -> value directly under SNAPSHOT_KEY.
    - Skips fx fields and internals.
    - Always snapshots `type` if present. It is not guaranteed to live in
      node.__dict__, so iterating over vars(node) will not catch it so we
      handle it explicitly here.
    """
    stored: dict[str, bytes] = {}
    for key, value in vars(node).items():
        if key in CORE_NODE_FIELDS:
            continue
        stored[key] = value
    if hasattr(node, "type") and "type" not in stored:
        stored["type"] = getattr(node, "type")
    node.meta[SNAPSHOT_KEY] = stored


def restore_node_fields(node: fx.Node) -> None:
    """Restore attributes previously stored under node.meta[SNAPSHOT_KEY].

    Applies the snapshotted attributes back onto the node instance. Canonical
    FX fields are never overwritten.
    """
    stored = node.meta.get(SNAPSHOT_KEY)
    if not isinstance(stored, dict):
        raise TypeError(
            f"Expected dict at meta['{SNAPSHOT_KEY}'] for node '{node.name}', got {type(stored)}"
        )
    for key, value in stored.items():
        if key in CORE_NODE_FIELDS:
            continue
        setattr(node, key, value)
