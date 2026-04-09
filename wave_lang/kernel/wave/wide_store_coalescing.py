# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Graph pass that tags eligible epilogue bf16 stores for wide store coalescing.

When a kernel uses swapped MFMA operands (e.g.
``get_tagged_mxfp4_gemm_preshuffle_b_wide_store``), the accumulator's
4-contiguous values align with the output's stride-1 dimension. This
pass identifies Write nodes that use the source/target dimension
remapping pattern (indicating swapped operands) and tags them so the
codegen emits v_permlane16_swap_b32 + buffer_store_dwordx4 instead of
scalar buffer_store_short.

Eligible writes are paired so that each lane in a (lane, lane+16)
pair writes a *different* tile group's wide store, eliminating the
duplicate stores that occur when both lanes write the same data.
The number of eligible writes must be even (asserted).

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
    pattern (swapped MFMA operands, as produced by the wide_store kernel
    variant). Writes without source/target are left untouched, making
    this pass safe to run unconditionally.

    Writes are paired for register-only deduplication: the first node
    in each pair stashes its codegen state, and the second triggers a
    paired ``permlane16_swap`` that lets each lane emit a unique store.
    """
    import wave_lang.kernel.lang as tkl

    root_graph = trace.get_root_graph()

    eligible_writes = []
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
            eligible_writes.append(node)

    # TODO: Add a fallback path for odd number of writes.
    assert len(eligible_writes) % 2 == 0, (
        f"Expected even number of eligible wide-store writes, "
        f"got {len(eligible_writes)}."
    )

    # Pair adjacent writes so the codegen can pass both tiles as
    # separate old_dst / src operands to a single permlane16_swap.
    # The "first" node stashes its codegen state; the "second" node
    # retrieves it and emits one unique store per lane (lower lane
    # writes tile A, upper lane writes tile B) — no duplicate stores.
    for first, second in zip(eligible_writes[0::2], eligible_writes[1::2]):
        first._permlane_pack_global = True
        first._permlane_pack_role = "first"
        first._permlane_partner = second
        second._permlane_pack_global = True
        second._permlane_pack_role = "second"
        second._permlane_partner = first
