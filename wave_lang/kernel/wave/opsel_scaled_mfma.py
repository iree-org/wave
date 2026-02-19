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

from iree.compiler.ir import (
    Float8E8M0FNUType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    VectorType,
)
from iree.compiler.dialects import (
    amdgpu as amdgpu_d,
    vector as vector_d,
)

from wave_lang.support.logging import get_logger

logger = get_logger("wave.opsel_scaled_mfma")


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
    extract_op = scale_value.owner
    if not hasattr(extract_op, "name") or extract_op.name != "vector.extract":
        return None

    extract_source = extract_op.operands[0]
    extract_source_type = extract_source.type
    if not isinstance(extract_source_type, VectorType):
        return None
    if extract_source_type.rank != 1 or extract_source_type.shape[0] != 1:
        return None

    bitcast_op = extract_source.owner
    if not hasattr(bitcast_op, "name") or bitcast_op.name != "vector.bitcast":
        return None

    bitcast_source = bitcast_op.operands[0]
    bitcast_source_type = bitcast_source.type
    if not isinstance(bitcast_source_type, VectorType):
        return None
    if bitcast_source_type.rank != 1 or bitcast_source_type.shape[0] != 1:
        return None

    slice_op = bitcast_source.owner
    if not hasattr(slice_op, "name") or slice_op.name != "vector.extract_strided_slice":
        return None

    offsets = slice_op.attributes["offsets"]
    offset = IntegerAttr(offsets[0]).value

    slice_source = slice_op.operands[0]
    slice_source_type = slice_source.type
    if not isinstance(slice_source_type, VectorType):
        return None
    # Only apply opsel optimization to vector<4xi8> sources.
    # The amdgpu.scaled_mfma operation requires vector<4xf8E8M0FNU> scale operands.
    if slice_source_type.rank != 1 or slice_source_type.shape[0] != 4:
        return None

    return (slice_source, offset)


def _walk_operations(op):
    """Recursively yield all operations nested inside op (post-order)."""
    for region in op.regions:
        for block in region:
            for child_op in block:
                yield from _walk_operations(child_op)
    yield op


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
        f8e8m0 = Float8E8M0FNUType.get()

        scaled_mfma_ops = []
        for op in _walk_operations(module.operation):
            if hasattr(op, "name") and op.name == "amdgpu.scaled_mfma":
                scaled_mfma_ops.append(op.opview)

        if not scaled_mfma_ops:
            return

        logger.debug(f"Found {len(scaled_mfma_ops)} scaled_mfma ops")

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

        # Cache: defining Operation -> bitcast result Value.
        # Using the Operation object identity ensures one bitcast per source load.
        source_op_to_bitcast = {}

        def get_wide_bitcast(source_vec):
            """Get or create a wide bitcast vector<Nxi8> -> vector<Nxf8E8M0FNU>."""
            defining_op = source_vec.owner
            if defining_op in source_op_to_bitcast:
                return source_op_to_bitcast[defining_op]

            source_type = source_vec.type
            n = source_type.shape[0]
            result_type = VectorType.get([n], f8e8m0)

            with InsertionPoint(defining_op):
                bc = vector_d.bitcast(result_type, source_vec)
            bc.owner.move_after(defining_op)

            source_op_to_bitcast[defining_op] = bc
            return bc

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
