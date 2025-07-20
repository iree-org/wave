# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import AtomicOp, Read, Write, get_custom
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

    def is_shared_memory_ops(node: fx.Node):
        custom = get_custom(node)
        if isinstance(custom, (AtomicOp, Read, Write)) and is_shared_mem_access(custom):
            custom.index = remove_global_indexing(custom.index, constraints)
        return False

    trace.walk(is_shared_memory_ops)
