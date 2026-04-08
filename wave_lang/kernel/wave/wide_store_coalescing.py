# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Graph pass that tags eligible epilogue bf16 stores for wide store coalescing.

When a kernel uses swapped MFMA operands (wide_stores=True), the
accumulator's 4-contiguous values align with the output's stride-1
dimension. This pass identifies Write nodes that use the source/target
dimension remapping pattern (indicating swapped operands) and tags them
so the codegen emits v_permlane16_swap_b32 + buffer_store_dwordx4
instead of scalar buffer_store_short.

Only tags writes that satisfy ALL conditions:
  1. Target memory is global address space
  2. Output dtype is bf16
  3. Write uses source/target syntax (swapped-operand layout)
"""

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import GLOBAL_ADDRESS_SPACE
from ..ops.wave_ops import Write, get_custom
from .region_canonicalization import RegionFormat, requires_region_format
from .utils.symbol_utils import subs_idxc


@requires_region_format(RegionFormat.SCHEDULE_SIGNATURE_PLACEHOLDERS)
def coalesce_wide_stores(trace: CapturedTrace):
    """Tag eligible bf16 global writes for permlane16_swap wide stores.

    Only tags Write nodes that use the source/target dimension remapping
    pattern, which indicates the kernel was built with ``wide_stores=True``
    (swapped MFMA operands). Writes without source/target are left
    untouched, making this pass safe to run unconditionally.
    """
    import wave_lang.kernel.lang as tkl

    root_graph = trace.get_root_graph()

    for node in root_graph.nodes:
        if node.op != "call_function":
            continue
        custom = get_custom(node)
        if not isinstance(custom, Write):
            continue
        if custom.source is None or custom.target is None:
            continue
        mem_type = custom.memory_type
        if (
            subs_idxc(mem_type.address_space) == GLOBAL_ADDRESS_SPACE
            and mem_type.dtype == tkl.bf16
        ):
            node._permlane_pack_global = True
