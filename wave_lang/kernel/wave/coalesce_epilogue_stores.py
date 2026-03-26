# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Graph pass that coalesces epilogue bf16 stores via permlane16_swap.

Marks eligible Write nodes so the codegen combines each thread's 4 bf16
values with its partner lane's (16 lanes apart) via v_permlane16_swap_b32,
producing 8 consecutive bf16 written as a single buffer_store_dwordx4.
No LDS staging or barriers required.

Precondition: the output memory must have M as the innermost (contiguous)
dimension (i.e. transpose_output=True producing [N, M] layout) so that 8
consecutive bf16 elements span 8 adjacent M rows.
"""

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import GLOBAL_ADDRESS_SPACE
from ..ops.wave_ops import Write, get_custom
from .region_canonicalization import RegionFormat, requires_region_format
from .utils.symbol_utils import subs_idxc


@requires_region_format(RegionFormat.SCHEDULE_SIGNATURE_PLACEHOLDERS)
def coalesce_epilogue_stores(trace: CapturedTrace):
    """Tag epilogue bf16 global writes for permlane16_swap packing.

    Walks the root graph and sets ``_permlane_pack_global = True`` on
    every Write node that targets global memory with bf16 dtype.
    The codegen in ``_write_permlane_pack_to_global`` handles the rest.
    """
    import wave_lang.kernel.lang as tkl

    root_graph = trace.get_root_graph()

    for node in root_graph.nodes:
        if node.op != "call_function":
            continue
        custom = get_custom(node)
        if not isinstance(custom, Write):
            continue
        mem_type = custom.memory_type
        if (
            subs_idxc(mem_type.address_space) == GLOBAL_ADDRESS_SPACE
            and mem_type.dtype == tkl.bf16
        ):
            node._permlane_pack_global = True
