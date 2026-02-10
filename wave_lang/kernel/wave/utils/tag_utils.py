# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tag Propagation Utilities

This module provides helper functions for propagating operation tags
through the Wave compilation pipeline. Tags are used by custom schedulers
to identify and organize operations.

Usage:
    from wave_lang.kernel.wave.utils.tag_utils import propagate_tag, set_tag

    # Propagate tag from source node to target node
    propagate_tag(source_node, target_node)

    # Set tag on node if tag is provided
    set_tag(node, "my_tag")
"""

from typing import Optional
import torch.fx as fx


def propagate_tag(source_node: fx.Node, target_node: fx.Node) -> None:
    """
    Propagate tag from source node to target node if present.

    This is the primary mechanism for preserving tags through compilation
    passes that decompose, reshape, or broadcast operations.

    Args:
        source_node: The node to copy the tag from
        target_node: The node to copy the tag to
    """
    tag = getattr(source_node, "tag", None)
    if tag is not None:
        target_node.tag = tag


def set_tag(node: fx.Node, tag: Optional[str | set[str]]) -> None:
    """
    Set tag on node if tag is provided.

    This is useful when creating new nodes that should inherit a tag
    from a variable rather than another node.

    Args:
        node: The node to set the tag on
        tag: The tag string or set of strings, or None to skip setting.
            If a string is provided, it will be converted to a single-element set.
    """
    if tag is not None and node is not None:
        if isinstance(tag, str):
            tag = {tag}
        node.tag = tag
