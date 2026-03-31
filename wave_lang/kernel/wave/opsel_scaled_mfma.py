# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MLIR pass: replace extract+bitcast chains on scale operands of
amdgpu.scaled_mfma with vector-level bitcast and opsel.

Before (per scale operand):
    %vec4   = vector.load ... : vector<4xi8>
    %slice  = vector.extract_strided_slice %vec4
                {offsets=[N], sizes=[1], strides=[1]}
                : vector<4xi8> to vector<1xi8>
    %bc1    = vector.bitcast %slice : vector<1xi8> to vector<1xf8E8M0FNU>
    %scalar = vector.extract %bc1[0] : f8E8M0FNU
    amdgpu.scaled_mfma ... (%scalar[0] * ...) ...

After:
    %vec4   = vector.load ... : vector<4xi8>
    %bc4    = vector.bitcast %vec4 : vector<4xi8> to vector<4xf8E8M0FNU>
    amdgpu.scaled_mfma ... (%bc4[N] * ...) ...

The extract_strided_slice, per-element bitcast and vector.extract are
dead-code eliminated by a subsequent canonicalization pass.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from iree.compiler.ir import (
    ArrayAttr,
    Block,
    Float8E8M0FNUType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    Value,
    VectorType,
)
from iree.compiler.dialects import (
    amdgpu as amdgpu_d,
    scf as scf_d,
    vector as vector_d,
)

from wave_lang.support.logging import get_logger

logger = get_logger("wave.opsel_scaled_mfma")

SCALE_VECTOR_WIDTH = 4

_i8 = None
_i64 = None
_v1xi8 = None
_sizes_1 = None
_strides_1 = None
_types_ctx = None


def _init_types():
    """Initialize MLIR type constants for the current context."""
    global _i8, _i64, _v1xi8, _sizes_1, _strides_1, _types_ctx
    from iree.compiler.ir import Context as _Ctx

    cur_ctx = _Ctx.current
    if _types_ctx is cur_ctx:
        return
    _types_ctx = cur_ctx
    _i8 = IntegerType.get_signless(8)
    _i64 = IntegerType.get_signless(64)
    _v1xi8 = VectorType.get([1], _i8)
    _sizes_1 = ArrayAttr.get([IntegerAttr.get(_i64, 1)])
    _strides_1 = ArrayAttr.get([IntegerAttr.get(_i64, 1)])


def _make_extract_slice(source: Value, offset: int, size: int = 1):
    """Create ``extract_strided_slice(source, offset, size, 1)``."""
    result_type = VectorType.get([size], _i8) if size > 1 else _v1xi8
    offsets = ArrayAttr.get([IntegerAttr.get(_i64, offset)])
    sizes = _sizes_1 if size == 1 else ArrayAttr.get([IntegerAttr.get(_i64, size)])
    return vector_d.ExtractStridedSliceOp(
        result_type, source, offsets, sizes, _strides_1
    )


def _is_op_named(op, name: str) -> bool:
    """Check if *op* is an MLIR operation with the given name."""
    return hasattr(op, "name") and op.name == name


def _look_through_select(op):
    """Skip arith.select (from flatten-bounds masking) to find the true-branch producer."""
    if _is_op_named(op, "arith.select"):
        return op.operands[1].owner
    return op


def _trace_scale_chain(scale_value):
    """Trace a scaled_mfma scale operand back through the extract+bitcast chain.

    Returns (source_vector_4xi8, byte_offset) if the pattern matches,
    or None if it doesn't.

    Expected chain (walking backwards from the scale operand):
        scale_value            : f8E8M0FNU   (scalar)
        <- vector.extract [0] : vector<1xf8E8M0FNU> -> f8E8M0FNU
        <- vector.bitcast     : vector<1xi8> -> vector<1xf8E8M0FNU>
        <- vector.extract_strided_slice {offsets=[N], sizes=[1], strides=[1]}
                               : vector<4xi8> -> vector<1xi8>
        <- source             : vector<4xi8>  (typically a vector.load)
    """
    # Step 1: scale_value must come from vector.extract [0] : vector<1xf8E8M0FNU>
    extract_op = scale_value.owner
    if not _is_op_named(extract_op, "vector.extract"):
        return None
    extract_source = extract_op.operands[0]
    extract_source_type = extract_source.type
    if not isinstance(extract_source_type, VectorType):
        return None
    if extract_source_type.rank != 1 or extract_source_type.shape[0] != 1:
        return None

    # Step 2: that vector<1xf8E8M0FNU> must come from vector.bitcast(vector<1xi8>)
    bitcast_op = extract_source.owner
    if not _is_op_named(bitcast_op, "vector.bitcast"):
        return None
    bitcast_source = bitcast_op.operands[0]
    bitcast_source_type = bitcast_source.type
    if not isinstance(bitcast_source_type, VectorType):
        return None
    if bitcast_source_type.rank != 1 or bitcast_source_type.shape[0] != 1:
        return None

    # Step 3: that vector<1xi8> must come from extract_strided_slice of a
    # vector<4xi8> (skip through arith.select from flatten-bounds masking).
    slice_op = _look_through_select(bitcast_source.owner)
    if not _is_op_named(slice_op, "vector.extract_strided_slice"):
        return None
    offset = IntegerAttr(slice_op.opview.offsets[0]).value

    # Step 4: the slice source must be exactly vector<4xi8> — the width
    # that amdgpu.scaled_mfma requires for vector scale operands.
    slice_source = slice_op.operands[0]
    slice_source_type = slice_source.type
    if not isinstance(slice_source_type, VectorType):
        return None
    if slice_source_type.rank != 1 or slice_source_type.shape[0] != SCALE_VECTOR_WIDTH:
        return None

    return (slice_source, offset)


def _walk_operations(op: Operation):
    """Recursively yield all operations nested inside *op* (post-order).

    Post-order is required by ``_coalesce_vector_iter_args``: inner
    ``scf.for`` ops must be processed before their enclosing parents so
    that erased ops are never revisited.
    """
    for region in op.regions:
        for block in region:
            for child_op in block:
                yield from _walk_operations(child_op)
    yield op


# ----------------------------------------------------------------------
# Pre-pass: coalesce vector<1xi8> scf.for iter_args into vector<4xi8>
# ----------------------------------------------------------------------


def _trace_extract_strided_slice(
    value: Value,
) -> Optional[tuple[Value, int]]:
    """Check if *value* is produced by extract_strided_slice of a vector<4xi8>.

    Looks through ``arith.select`` (inserted by flatten-bounds masking)
    to find the underlying extract_strided_slice.

    Returns ``(source_vec4xi8, byte_offset)`` or ``None``.
    """
    op = _look_through_select(value.owner)
    if not _is_op_named(op, "vector.extract_strided_slice"):
        return None
    source = op.operands[0]
    source_type = source.type
    if not isinstance(source_type, VectorType):
        return None
    if source_type.rank != 1 or source_type.shape[0] != SCALE_VECTOR_WIDTH:
        return None
    offset = IntegerAttr(op.opview.offsets[0]).value
    return (source, offset)


def _find_yield_op(for_view) -> Optional[Operation]:
    """Return the scf.yield terminator of *for_view*'s body block."""
    for op in for_view.body.operations:
        if _is_op_named(op, "scf.yield"):
            return op
    return None


def _find_mergeable_groups(
    for_view, yield_op: Operation
) -> list[tuple[Value, Value, dict[int, int]]]:
    """Find groups of ``vector<1xi8>`` iter_args that can be coalesced.

    A group is valid when:
    * At least 2 init values are ``extract_strided_slice`` at distinct
      offsets from the same ``vector<4xi8>`` source.
    * The corresponding yield values follow the same pattern from a
      (possibly different) ``vector<4xi8>`` source.
    * For each member, init_offset == yield_offset (byte identity is
      preserved across iterations).

    Partial groups (e.g. only offsets {0, 2}) are accepted — the
    coalesced ``vector<4xi8>`` iter_arg simply carries unused bytes.

    Returns a list of ``(init_source, yield_source, {offset: iter_index})``.
    """
    _init_types()

    init_args = list(for_view.initArgs)
    yield_operands = list(yield_op.operands)

    # Collect per-arg info: (iter_index, offset, init_source, yield_source).
    # yield_source may be None when the yield value doesn't trace back to
    # extract_strided_slice (e.g. pipelined sub-word loads from a swapped
    # double-buffer).  Such groups are still mergeable — we construct the
    # yield vector from individual bytes later.
    # Phase 1: Classify each vector<1xi8> iter_arg.
    # For each one, check whether its init and yield values trace back to
    # extract_strided_slice of a vector<4xi8>.  Three outcomes:
    #   - Both trace, same offset  -> normal group member (yield_src set)
    #   - Both trace, diff offset  -> address-shifted (yield_src = None)
    #   - Yield doesn't trace      -> untraceable (yield_src = None)
    eligible = []
    for i, iter_arg in enumerate(for_view.inner_iter_args):
        if iter_arg.type != _v1xi8:
            continue
        init_info = _trace_extract_strided_slice(init_args[i])
        if init_info is None:
            continue
        init_src, init_off = init_info
        yield_info = _trace_extract_strided_slice(yield_operands[i])
        if yield_info is not None:
            yield_src, yield_off = yield_info
            if init_off != yield_off:
                # Init and yield extract different bytes from their respective
                # dword loads.  This happens when the pipeliner shifts the LDS
                # address by one scale element between init and yield (e.g.
                # init loads at addr X and extracts byte 1, yield loads at
                # addr X+1 and extracts byte 0 — same physical byte).
                # Mark yield as untraceable so the coalescer constructs the
                # yield vector from the yield source's parent vector<4xi8>.
                yield_src = None
                logger.debug(
                    f"iter_arg {i}: init offset {init_off} != yield offset "
                    f"{yield_off} — treating yield as untraceable for "
                    f"dword coalescing"
                )
        else:
            yield_src = None
            logger.debug(
                f"iter_arg {i}: init traces to offset {init_off} but yield "
                f"value does not trace to extract_strided_slice — will "
                f"construct yield vector from individual bytes"
            )
        eligible.append((i, init_off, init_src, yield_src))

    # Phase 2: Group eligible iter_args by their init source vector<4xi8>.
    # Uses hash() for MLIR Value identity (id() can alias across bindings).
    by_init_src = defaultdict(list)
    for entry in eligible:
        _, _, init_src, _ = entry
        by_init_src[hash(init_src.owner)].append(entry)

    # Phase 3: Within each source group, greedily form mergeable groups
    # from distinct byte offsets.  A group needs >= 2 offsets to be worth
    # coalescing (full groups of 4 are common; partial like {0, 2} arise
    # from preshuffle scales).
    result = []
    for entries in by_init_src.values():
        by_offset = defaultdict(list)
        for entry in entries:
            _, off, _, _ = entry
            by_offset[off].append(entry)

        present_offsets = [o for o in range(SCALE_VECTOR_WIDTH) if by_offset[o]]
        while len(present_offsets) >= 2:
            members = {}
            init_source = None
            yield_owners = set()
            yield_source = None
            has_untraceable_yield = False
            for o in present_offsets:
                idx, _, isrc, ysrc = by_offset[o].pop(0)
                members[o] = idx
                init_source = isrc
                if ysrc is not None:
                    yield_source = ysrc
                    yield_owners.add(hash(ysrc.owner))
                else:
                    has_untraceable_yield = True
            if has_untraceable_yield:
                # Some yield values don't trace — yield_source=None signals
                # _coalesce_vector_iter_args to construct the yield vector.
                result.append((init_source, None, members))
            elif len(yield_owners) == 1:
                # All yield values come from the same vector<4xi8> source.
                result.append((init_source, yield_source, members))
            # else: yields come from multiple sources — can't safely merge.
            present_offsets = [o for o in range(SCALE_VECTOR_WIDTH) if by_offset[o]]

    return result


@dataclass
class _CoalescePlan:
    """Pure-data plan for coalescing iter_args in a single scf.for."""

    groups: list[tuple[Value, Value, dict[int, int]]]
    merged_indices: set[int] = field(default_factory=set)
    index_to_group: dict[int, tuple[int, int]] = field(default_factory=dict)
    group_new_iter_idx: dict[int, int] = field(default_factory=dict)
    old_to_new_iter_idx: dict[int, int] = field(default_factory=dict)
    new_init_args: list = field(default_factory=list)
    new_yield_vals: list = field(default_factory=list)
    num_iter_args: int = 0


def _build_coalesce_plan(
    groups: list[tuple[Value, Value, dict[int, int]]],
    for_view,
    yield_op: Operation,
) -> _CoalescePlan:
    """Compute index mappings and new init/yield arg lists (no IR mutation)."""
    plan = _CoalescePlan(groups=groups)

    # Map each merged iter_arg index to its group and byte offset.
    for g_idx, (_, _, members) in enumerate(groups):
        for offset, iter_idx in members.items():
            plan.merged_indices.add(iter_idx)
            plan.index_to_group[iter_idx] = (g_idx, offset)

    old_init_args = list(for_view.initArgs)
    old_yield_operands = list(yield_op.operands)
    plan.num_iter_args = len(list(for_view.inner_iter_args))

    # Build new_init_args: replace each group's N individual vector<1xi8>
    # init values with one vector<4xi8> (the group's init_source).
    # Non-merged iter_args pass through unchanged.
    cur = 0
    seen_groups: set[int] = set()
    for i in range(plan.num_iter_args):
        if i in plan.merged_indices:
            g_idx, _ = plan.index_to_group[i]
            if g_idx not in seen_groups:
                seen_groups.add(g_idx)
                plan.new_init_args.append(groups[g_idx][0])
                plan.group_new_iter_idx[g_idx] = cur
                cur += 1
            plan.old_to_new_iter_idx[i] = plan.group_new_iter_idx[g_idx]
        else:
            plan.new_init_args.append(old_init_args[i])
            plan.old_to_new_iter_idx[i] = cur
            cur += 1

    # Build new_yield_vals: similarly replace grouped yields with the
    # group's yield_source vector<4xi8>.
    seen_groups = set()
    for i in range(plan.num_iter_args):
        if i in plan.merged_indices:
            g_idx, _ = plan.index_to_group[i]
            if g_idx not in seen_groups:
                seen_groups.add(g_idx)
                plan.new_yield_vals.append(groups[g_idx][1])
        else:
            plan.new_yield_vals.append(old_yield_operands[i])

    return plan


def _rewire_for_results(
    plan: _CoalescePlan,
    old_results: list,
    new_results: list,
    for_op,
) -> None:
    """Replace uses of old scf.for results with extracts from the new for.

    Non-merged results map 1:1.  For merged groups, insert
    extract_strided_slice ops after the new for to extract individual
    bytes from the coalesced vector<4xi8> result.
    """
    seen_groups: set[int] = set()
    for i in range(plan.num_iter_args):
        # Non-merged results pass through directly.
        if i not in plan.merged_indices:
            new_idx = plan.old_to_new_iter_idx[i]
            old_results[i].replace_all_uses_with(new_results[new_idx])
            continue

        # Process each merged group once (skip duplicate member visits).
        g_idx, _ = plan.index_to_group[i]
        if g_idx in seen_groups:
            continue
        seen_groups.add(g_idx)

        members = plan.groups[g_idx][2]
        new_idx = plan.group_new_iter_idx[g_idx]
        # Skip extract creation if no downstream code uses these results.
        has_users = any(
            any(True for _ in old_results[members[o]].uses) for o in members
        )
        if not has_users:
            continue

        # Insert byte extracts from the coalesced result after the for op.
        with InsertionPoint(for_op):
            for o, old_i in members.items():
                if not any(True for _ in old_results[old_i].uses):
                    continue
                extract_slice = _make_extract_slice(new_results[new_idx], o)
                old_results[old_i].replace_all_uses_with(extract_slice.result)


def _coalesce_vector_iter_args(module: Module) -> None:
    """Merge groups of ``vector<1xi8>`` scf.for iter_args into ``vector<4xi8>``.

    Pipeline double-buffering splits a ``vector<4xi8>`` scale load into
    individual bytes for loop-carry.  This pass merges them back so that
    ``_trace_scale_chain`` sees the full ``extract_strided_slice`` pattern
    inside the loop body and the opsel optimisation fires.

    Handles both full groups (all 4 offsets present) and partial groups
    (e.g. only offsets {0, 2} from preshuffle scales).
    """
    _init_types()

    for_ops = [
        op for op in _walk_operations(module.operation) if _is_op_named(op, "scf.for")
    ]

    for for_op in for_ops:
        for_view = for_op.opview
        yield_op = _find_yield_op(for_view)
        if yield_op is None:
            continue

        groups = _find_mergeable_groups(for_view, yield_op)
        if not groups:
            continue

        logger.debug(f"Coalescing {len(groups)} group(s) of vector<1xi8> iter_args")

        # For groups whose yield values don't trace to extract_strided_slice
        # (e.g. pipelined sub-word loads from a swapped double-buffer),
        # create a single wide vector<4xi8> load at the same base address
        # as the byte-0 yield load.  This replaces the sub-word loads with
        # one dword load that waveasm can map to a ds_read_b32.
        old_yield_operands = list(yield_op.operands)
        for g_idx, (init_source, yield_source, members) in enumerate(groups):
            if yield_source is not None:
                continue

            # Try to find a yield value whose source vector<4xi8> already
            # exists (from an extract_strided_slice of a wider load).
            # This handles the address-shifted pattern where init and yield
            # extract different offsets from their respective dword loads.
            found_yield_src = False
            for member_idx in members.values():
                yield_val = old_yield_operands[member_idx]
                yield_info = _trace_extract_strided_slice(yield_val)
                if yield_info is not None:
                    yield_vec_src, _ = yield_info
                    groups[g_idx] = (init_source, yield_vec_src, members)
                    logger.debug(
                        f"Group {g_idx}: reusing existing vector<4xi8> yield "
                        f"source (offset-shifted pattern)"
                    )
                    found_yield_src = True
                    break
            if found_yield_src:
                continue

            if 0 not in members:
                logger.debug(
                    f"Group {g_idx}: no byte-0 member, cannot determine base "
                    f"address for wide load — skipping"
                )
                continue
            byte0_yield = old_yield_operands[members[0]]
            byte0_op = byte0_yield.owner
            if not _is_op_named(byte0_op, "vector.load"):
                logger.debug(
                    f"Group {g_idx}: byte-0 yield is not a vector.load "
                    f"({byte0_op.name}) — skipping"
                )
                continue
            # Assumes the 4 scale bytes are contiguous starting at the
            # byte-0 address.  This holds for LDS scale loads produced by
            # the pipeliner but is not validated here.
            load_view = byte0_op.opview
            memref = load_view.base
            indices = list(load_view.indices)
            v4xi8 = VectorType.get([SCALE_VECTOR_WIDTH], _i8)
            with InsertionPoint(byte0_op):
                wide_load = vector_d.load(v4xi8, memref, indices)
            groups[g_idx] = (init_source, wide_load, members)
            logger.debug(
                f"Created wide vector<{SCALE_VECTOR_WIDTH}xi8> load for "
                f"group {g_idx} yield value (replaces {len(members)} "
                f"sub-word loads)"
            )

        # Drop unresolved groups: if neither strategy above found a valid
        # vector<4xi8> yield source, the group's yield_source is still None.
        # Passing None to scf.YieldOp would crash, so we exclude these
        # groups from coalescing.  This can happen when:
        #   - No member's yield traces to extract_strided_slice (strategy 1)
        #   - No byte-0 member exists, or byte-0 yield isn't a vector.load
        #     (strategy 2 prerequisites)
        groups = [g for g in groups if g[1] is not None]
        if not groups:
            continue

        plan = _build_coalesce_plan(groups, for_view, yield_op)
        old_iter_args = list(for_view.inner_iter_args)
        old_iv = for_view.induction_variable
        old_results = list(for_view.results_)

        # --- create new scf.for ---
        new_for = scf_d.ForOp(
            for_view.lowerBound,
            for_view.upperBound,
            for_view.step,
            plan.new_init_args,
            ip=InsertionPoint(for_op),
        )

        with InsertionPoint.at_block_begin(new_for.body):
            new_yield = scf_d.YieldOp(plan.new_yield_vals)

        # --- move old body ops (except old yield) into new for ---
        old_body_ops = [
            op for op in for_view.body.operations if not _is_op_named(op, "scf.yield")
        ]
        for op in old_body_ops:
            op.move_before(new_yield.operation)

        # --- insert extract_strided_slice ops at block begin ---
        first_op = next(iter(new_for.body.operations))
        extract_results: dict[int, Value] = {}
        with InsertionPoint(first_op):
            for g_idx, (_, _, members) in enumerate(groups):
                merged_arg = new_for.inner_iter_args[plan.group_new_iter_idx[g_idx]]
                for offset, iter_idx in members.items():
                    extract_slice = _make_extract_slice(merged_arg, offset)
                    extract_results[iter_idx] = extract_slice.result

        # --- rewire block arg uses ---
        old_iv.replace_all_uses_with(new_for.induction_variable)
        for i in range(plan.num_iter_args):
            if i in plan.merged_indices:
                old_iter_args[i].replace_all_uses_with(extract_results[i])
            else:
                new_idx = plan.old_to_new_iter_idx[i]
                old_iter_args[i].replace_all_uses_with(new_for.inner_iter_args[new_idx])

        # --- rewire for results and clean up ---
        _rewire_for_results(plan, old_results, list(new_for.results_), for_op)
        for_op.erase()


def apply_opsel_scaled_mfma(module: Module):
    """Walk the MLIR module and apply the opsel optimization to scaled_mfma ops.

    For each scaled_mfma, if a scale operand traces back through:
        vector.extract[0] <- vector.bitcast(1xi8->1xf8E8M0FNU)
            <- vector.extract_strided_slice(Nxi8->1xi8, offset=K)
    then replace the scale with a vector.bitcast(Nxi8->Nxf8E8M0FNU)
    of the source and set scales_idx to K.
    """
    mlir_ctx = module.operation.context

    with mlir_ctx, Location.unknown():
        # Pre-pass: merge vector<1xi8> scf.for iter_args back into
        # vector<4xi8> so that scale chains are visible to the main pass.
        # Non-loop-carried loads that are already vector<4xi8> (produced by
        # merge_contiguous_reads) don't need additional merging.
        _coalesce_vector_iter_args(module)

        f8e8m0 = Float8E8M0FNUType.get()

        # Collect all scaled_mfma ops to process.
        scaled_mfma_ops = []
        for op in _walk_operations(module.operation):
            if _is_op_named(op, "amdgpu.scaled_mfma"):
                scaled_mfma_ops.append(op.opview)

        if not scaled_mfma_ops:
            return

        logger.debug(f"Found {len(scaled_mfma_ops)} scaled_mfma ops")

        # For each scaled_mfma, try to trace each scale operand back
        # through the extract+bitcast chain to find the source vector<4xi8>.
        replacements = []
        for mfma_op in scaled_mfma_ops:
            idx_a = int(mfma_op.scalesIdxA)
            idx_b = int(mfma_op.scalesIdxB)

            new_scale_a = None
            new_idx_a = idx_a
            new_scale_b = None
            new_idx_b = idx_b

            chain_a = _trace_scale_chain(mfma_op.scalesA)
            if chain_a is not None:
                new_scale_a, new_idx_a = chain_a

            chain_b = _trace_scale_chain(mfma_op.scalesB)
            if chain_b is not None:
                new_scale_b, new_idx_b = chain_b

            if new_scale_a is not None or new_scale_b is not None:
                replacements.append(
                    (mfma_op, new_scale_a, new_idx_a, new_scale_b, new_idx_b)
                )

        if not replacements:
            logger.debug("No opsel optimization opportunities found")
            return

        logger.debug(f"Applying opsel optimization to {len(replacements)} ops")

        i32 = IntegerType.get_signless(32)

        # Bitcast caches: ensure one bitcast per source vector<4xi8>,
        # shared across all scaled_mfma ops that reference the same load.
        source_op_to_bitcast = {}
        block_arg_bitcasts: dict[tuple, Value] = {}

        def get_wide_bitcast(source_vec: Value) -> Value:
            """Get or create vector.bitcast(vector<Nxi8> -> vector<Nxf8E8M0FNU>).

            For block arguments (loop-carried iter_args), inserts at block
            begin.  For op results, inserts immediately after the defining op.
            """
            defining_op = source_vec.owner

            # Block argument: source is a loop iter_arg, not an op result.
            if isinstance(defining_op, Block):
                cache_key = (defining_op, source_vec.arg_number)
                if cache_key in block_arg_bitcasts:
                    return block_arg_bitcasts[cache_key]
                source_type = source_vec.type
                vec_width = source_type.shape[0]
                result_type = VectorType.get([vec_width], f8e8m0)
                with InsertionPoint.at_block_begin(defining_op):
                    bc = vector_d.bitcast(result_type, source_vec)
                block_arg_bitcasts[cache_key] = bc
                return bc

            # Op result: use (op, result_number) as cache key for
            # multi-result ops, or just the op for single-result ops.
            if len(defining_op.results) > 1:
                cache_key = (defining_op, source_vec.result_number)
            else:
                cache_key = defining_op
            if cache_key in source_op_to_bitcast:
                return source_op_to_bitcast[cache_key]

            source_type = source_vec.type
            vec_width = source_type.shape[0]
            result_type = VectorType.get([vec_width], f8e8m0)
            with InsertionPoint(defining_op):
                bc = vector_d.bitcast(result_type, source_vec)
            bc.owner.move_after(defining_op)
            source_op_to_bitcast[cache_key] = bc
            return bc

        # Apply: replace each scalar-scale scaled_mfma with a new one
        # that uses vector<4xf8E8M0FNU> scales + opsel byte index.
        for mfma_op, new_scale_a, new_idx_a, new_scale_b, new_idx_b in replacements:
            actual_scale_a = mfma_op.scalesA
            actual_scale_b = mfma_op.scalesB

            if new_scale_a is not None:
                actual_scale_a = get_wide_bitcast(new_scale_a)
            if new_scale_b is not None:
                actual_scale_b = get_wide_bitcast(new_scale_b)

            with InsertionPoint(mfma_op):
                new_mfma = amdgpu_d.scaled_mfma(
                    m=mfma_op.attributes["m"],
                    n=mfma_op.attributes["n"],
                    k=mfma_op.attributes["k"],
                    source_a=mfma_op.sourceA,
                    source_b=mfma_op.sourceB,
                    dest_c=mfma_op.destC,
                    scales_a=actual_scale_a,
                    scales_b=actual_scale_b,
                    scales_idx_a=IntegerAttr.get(i32, new_idx_a),
                    scales_idx_b=IntegerAttr.get(i32, new_idx_b),
                )
            mfma_op.result.replace_all_uses_with(new_mfma)
            mfma_op.operation.erase()

    logger.debug("opsel optimization applied successfully")
