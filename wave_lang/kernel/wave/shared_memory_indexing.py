# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import AtomicOp, CustomOp, GatherToLDS, Read, Write, get_custom
from .constraints import Constraint
from .utils.general_utils import is_shared_mem_access, remove_global_indexing


def apply_shared_memory_indexing_corrections(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    This function removes global indexing from ops that deal with shared memory.
    Global indexing is an indexing that arises from Workgroup constraints
    and Tiling constraints.
    """

    def get_all_sources(node: fx.Node) -> list[fx.Node]:
        source1 = propagate_loop_carried_vars(node, 0)
        source2 = propagate_loop_carried_vars(node, 1)
        if source1 != source2:
            return [source1, source2]
        return [source1]

    def is_shared_memory_ops(node: fx.Node) -> bool:
        custom = get_custom(node)
        if isinstance(custom, (AtomicOp, Read, Write)) and is_shared_mem_access(custom):
            custom.index = remove_global_indexing(custom.index, constraints)
        return False

    def is_shared_memory_op2(node: CustomOp) -> list[fx.Node]:
        if (
            isinstance(node, (Read, Write, AtomicOp))
            and node.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            return get_all_sources(node.memory)

        if isinstance(node, GatherToLDS):
            return get_all_sources(node.dst)

        return []

    trace.walk(is_shared_memory_ops)
