# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import Any, Optional

import math
import sympy
import torch.fx as fx

from wave_lang.support.ir_imports import (
    Attribute,
    DenseElementsAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    llvm_d,
    memref_d,
    vector_d,
    func_d,
    Operation,
)
from .ir_utils import (
    is_float_type,
)

from ..._support.indexing import IndexExpr, IndexingContext, IndexSequence, IndexSymbol
from ...compiler.base import ValidationError
from ...compiler.builder import IRProxyValue
from ...compiler.utils import strides_from_symbolic_shape
from ...lang.global_symbols import *
from ...lang.wave_types import IndexMapping
from ...ops.wave_ops import (
    CustomOp,
    gather_to_lds,
    get_custom,
    read,
    write,
    scatter_add,
    read_meets_hw_transpose_requirements,
)
from ..utils.general_utils import get_fastest_index, linearize_index
from ..utils.mapping_utils import transform_index_on_mapping
from ..utils.symbol_utils import safe_subs
from .emitter import (
    WaveEmitter,
    add_emitter_subs,
    cast_kernel_buffer,
    cast_py_literal,
    cast_py_value,
    cast_vector,
    gen_sympy_index,
    get_constant_attr,
    get_type_or_element_type,
    handle_op,
)


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


@functools.lru_cache
def _simplify(expr):
    """
    Simple wrapper around simplify in order to utilize LRU Cache.
    This is important to minimize compile time caused by re-simplifying
    expressions.
    """
    return sympy.simplify(expr)


def _split_index(src: IndexExpr | int) -> tuple[IndexExpr, IndexExpr]:
    """
    Split index expr into thread-dependent and thread-independent parts
    """
    subs_wg = {WORKGROUP_0: 0, WORKGROUP_1: 0, WORKGROUP_2: 0}
    # Replace all wg symbols with 0s to get thread-dependent index.
    # All dynamic values will also be part of thread-index.
    thread_dependent_index = safe_subs(src, subs_wg)

    # Compute thread-independent index as `orig_index - thread_dependent_index`
    # All thread symbols and dynamic should cancel-out in the result.
    thread_independent_index = _simplify(src - thread_dependent_index)
    if thread_independent_index.free_symbols - set(subs_wg.keys()):
        # If we have any symbols besides wg symbols, means some thread or
        # dynamic symbols were not canceled out, use the entire index as
        # thread dependent index.
        thread_independent_index = sympy.sympify(0)
        thread_dependent_index = src

    return thread_independent_index, thread_dependent_index


def _extract0(src):
    static_pos = [0] * src.type.rank
    return vector_d.extract(src, static_position=static_pos, dynamic_position=[])


def _build_dyn_vals_map(
    mapping: Optional[IndexMapping], dynamic_vals: tuple[Value, ...]
) -> dict[IndexExpr, Value]:
    if mapping is None:
        return {}

    assert len(mapping.dynamic_val_indices) == len(
        dynamic_vals
    ), f"Expected {len(mapping.dynamic_val_indices)} dynamic values but got {len(dynamic_vals)}"
    return {
        sym: _extract0(val)
        for sym, val in zip(mapping.dynamic_val_indices.keys(), dynamic_vals)
    }


def _build_start_indices(
    emitter: WaveEmitter,
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
    dynamic_values: dict[IndexExpr, Any] = {},
) -> tuple[list[OpResult], list[OpResult], list[OpResult]]:
    start_indices = _get_start_indices(src_indices)
    split_indices = [_split_index(i) for i in start_indices]
    subs = add_emitter_subs(emitter, dynamic_values)
    indices = [gen_sympy_index(subs, i) for i in start_indices]
    indices_wg = [gen_sympy_index(subs, i[0]) for i in split_indices]
    indices_th = [gen_sympy_index(subs, i[1]) for i in split_indices]

    return indices, indices_wg, indices_th


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def _get_symbolic_shape(node: fx.Node) -> tuple[IndexExpr]:
    return get_custom(node).type.symbolic_shape


def _build_mask(
    emitter: WaveEmitter,
    index: dict[IndexExpr, IndexExpr],
    elements_per_thread: int,
    bounds: Optional[dict[IndexSymbol, IndexExpr]],
    dynamic_values: dict[IndexExpr, Any] = {},
) -> Optional[OpResult]:
    if not bounds:
        return None

    idxc = IndexingContext.current()
    fastest_dim = get_fastest_index(index)
    last_dim = list(index)[fastest_dim]
    new_index = {k: _get_start_index(v) for k, v in index.items()}

    new_index[last_dim] = new_index[last_dim] + idxc.iota(elements_per_thread)

    mask_expr = functools.reduce(
        lambda a, b: sympy.And(a, b),
        (new_index[dim] < bound for dim, bound in bounds.items()),
    )
    mask = gen_sympy_index(add_emitter_subs(emitter, dynamic_values), mask_expr)

    mask_vec_type = VectorType.get([elements_per_thread], IntegerType.get_signless(1))
    if mask.type != mask_vec_type:
        mask = vector_d.broadcast(mask_vec_type, mask)

    return mask


def _get_splat_const(vec_type: IrType, value: Any) -> Value:
    splat = DenseElementsAttr.get_splat(
        vec_type, get_constant_attr(value, vec_type.element_type)
    )
    return arith_d.constant(vec_type, splat)


def _constant_mask(vec_type: IrType) -> Value:
    return _get_splat_const(vec_type, 1)


def _get_max_buffer_size(elem_type: IrType) -> int:
    """
    Return max memref size suitable for buffer ops.

    Buffer ops offsets are i32, return maximum memref size in elements.
    """
    return ((1 << 31) - 1) // (elem_type.width // 8)


def _linearize_memref(
    mem: Value,
    offsets_wg: tuple[Value | int],
    offsets_th: tuple[Value | int],
    strides: tuple[Value],
) -> tuple[Value, Value]:
    """
    Convert n-D memref into 1-D memref, suitable for buffer ops.

    Apply offsets to the memref and convert result to 1-D. Resulting memref size
    is set to `max_buffer_size - 1` so buffer access to the last element will be
    no-op.
    """
    memref_type = mem.type
    offset = None
    offset_th = None
    overflow_flags = arith_d.IntegerOverflowFlags.nsw
    for ind_wg, ind_th, stride in zip(offsets_wg, offsets_th, strides):
        if isinstance(ind_wg, int):
            ind_wg = arith_d.constant(IndexType.get(), ind_wg)

        if isinstance(ind_th, int):
            ind_th = arith_d.constant(IndexType.get(), ind_th)

        off_wg = arith_d.muli(ind_wg, stride, overflow_flags=overflow_flags)
        if offset is None:
            offset = off_wg
        else:
            offset = arith_d.addi(offset, off_wg, overflow_flags=overflow_flags)

        off_th = arith_d.muli(ind_th, stride, overflow_flags=overflow_flags)
        if offset_th is None:
            offset_th = off_th
        else:
            offset_th = arith_d.addi(offset_th, off_th, overflow_flags=overflow_flags)

    size_full = arith_d.constant(
        IndexType.get(), _get_max_buffer_size(memref_type.element_type) - 1
    )

    dyn_val = ShapedType.get_dynamic_size()
    res_shape = [dyn_val]
    element_type = memref_type.element_type
    memory_space = memref_type.memory_space
    resut_type = MemRefType.get(
        res_shape,
        element_type,
        layout=Attribute.parse("strided<[1], offset: ?>"),
        memory_space=memory_space,
    )
    memref_metadata = memref_d.extract_strided_metadata(mem)
    memref_base_offset = memref_metadata[1]
    offset = arith_d.addi(offset, memref_base_offset, overflow_flags=overflow_flags)
    return (
        memref_d.reinterpret_cast(
            resut_type,
            mem,
            offsets=[offset],
            sizes=[size_full],
            strides=[],
            static_offsets=[dyn_val],
            static_sizes=[dyn_val],
            static_strides=[1],
        ),
        offset_th,
    )


def _linearize_shared_mem(memory: CustomOp) -> Value:
    """
    Convert shared memory with statically shaped N-d memref into 1-D memref.
    """
    flat_numel = math.prod(memory.type.shape)
    assert (
        memory.type.has_static_shape
    ), "Expecting static shape to linearize for shared memory."
    memory_space = memory.type.memory_space
    flat_memref_type = MemRefType.get(
        [flat_numel], memory.type.element_type, memory_space=memory_space
    )
    flattened_mem = memref_d.reinterpret_cast(
        flat_memref_type,
        memory,
        offsets=[],
        sizes=[],
        strides=[],
        static_offsets=[0],
        static_sizes=[flat_numel],
        static_strides=[1],
    )
    return flattened_mem


def _get_splat_input(src: Optional[Value]) -> Optional[Value]:
    """
    If `src` is vector.splat result, return splat input, otherwise return None.
    """
    if src is None:
        return None

    owner = getattr(src, "owner", None)
    if owner is None:
        return None

    op = src.owner.opview
    if isinstance(op, vector_d.BroadcastOp) and not VectorType.isinstance(
        op.source.type
    ):
        return op.source

    return None


def _valid_bytes_buffer(elem_type: IrType) -> int:
    """
    Make valid bytes to be the address of the last byte of the second to last element that can fit in a 32 bit offset to memory address
    """
    ans = (1 << 31) - 1 - (elem_type.width // 8)

    assert isinstance(ans, int)
    return ans


def _get_out_of_bounds_index(element_type: IrType) -> int:
    """
    returns the first index that's out of bounds of a buffer based on the element type and maximum bytes
    """
    element_width_in_bytes = element_type.width // 8
    oob_index_value = (
        _valid_bytes_buffer(element_type) + element_width_in_bytes
    ) // element_width_in_bytes
    assert (oob_index_value * element_width_in_bytes) > _valid_bytes_buffer(
        element_type
    )
    assert (oob_index_value * element_width_in_bytes) < (1 << 31)
    return oob_index_value


def _get_constant_value(candidate: Value):
    """
    returns constantOp's value if candidate is arith.constantOp. Else, returns None.
    """
    if not isinstance(candidate.owner, Operation):
        return None
    if not hasattr(candidate.owner, "name"):
        return None
    if candidate.owner.name != "arith.constant":
        return None
    return candidate.owner.attributes["value"].value


def _cast_buffer_and_encode_stride(
    ptr: Value, strides: tuple[Value], elem_type: IrType, emitter: WaveEmitter
) -> Value:
    uint64 = IntegerType.get_signless(64)
    uint14 = IntegerType.get_signless(14)

    valid_bytes = _valid_bytes_buffer(
        elem_type
    )  # max bytes that are in range to be addressed from a buffer
    valid_bytes_constant = get_constant_attr(valid_bytes, uint64)
    valid_bytes_constant = arith_d.constant(uint64, valid_bytes_constant)
    stride_rank = len(strides)
    swizzle_stride = None

    if stride_rank >= 2:
        # fastest_dim_bound == second to last stride.
        stride_candidate = strides[-2]
        stride_int = _get_constant_value(stride_candidate)
        # Only swizzle if stride is static and <= 8192(the useful case).
        if stride_int and stride_int <= 8192:
            swizzle_stride = arith_d.index_cast(uint14, stride_candidate)

    if swizzle_stride:
        ptr = amdgpu_d.fat_raw_buffer_cast(
            ptr,
            cache_swizzle_stride=swizzle_stride,
            bounds_check=True,
            reset_offset=True,
            valid_bytes=valid_bytes_constant,
        )
    else:
        ptr = amdgpu_d.fat_raw_buffer_cast(
            ptr,
            bounds_check=True,
            reset_offset=True,
            valid_bytes=valid_bytes_constant,
        )

    return ptr


def _create_vec_read_write(
    emitter: WaveEmitter,
    symbolic_shape: tuple[IndexExpr, ...],
    mem: Value,
    value: Optional[Value],
    vector_type: Optional[IrType],
    start_indices: tuple[Value],
    start_indices_wg: tuple[Value],
    start_indices_th: tuple[Value],
    elements_per_thread: int,
    memory: CustomOp,
    mask: Optional[Value],
    node_index: Optional[IndexSequence] = None,
) -> Optional[Value]:
    is_read = value is None
    uint32 = IntegerType.get_signless(32)

    def extract(vec, ind):
        return vector_d.extract(vec, static_position=[ind], dynamic_position=[])

    if memory.type.address_space == SHARED_ADDRESS_SPACE and hasattr(
        memory, "distributed_shape"
    ):
        symbolic_shape = memory.distributed_shape

    # only use buffer ops on global memory
    is_global_mem = mem.type.memory_space is None
    buffer_ops_enabled = emitter.options.use_buffer_ops and is_global_mem
    is_shared_mem = memory.type.address_space == SHARED_ADDRESS_SPACE and node_index
    linearize_shared_mem = is_shared_mem and emitter.options.linearize_shared_access

    stride_values = strides_from_symbolic_shape(
        IndexingContext.current(), symbolic_shape, allow_mixed_shapes=True
    )
    has_int_strides = all(isinstance(s, int) for s in stride_values)
    strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in stride_values]

    no_masked_load_store_ops = buffer_ops_enabled

    mask_splat = _get_splat_input(mask)
    splatted_mask = mask_splat is not None

    if vector_type is None:
        vector_type = value.type

    element_type = vector_type.element_type
    # Case 1: Generate load/stores with no mask and no offset
    if mask is None:
        offset_th = None
        if buffer_ops_enabled:
            # TODO: If strides cannot be converted into integers, means they are dynamic
            # and linearize breaks, need to investigate later.
            mem, offset_th = _linearize_memref(
                mem, start_indices_wg, start_indices_th, strides
            )
            mem = _cast_buffer_and_encode_stride(mem, strides, element_type, emitter)
        if linearize_shared_mem:
            mem = _linearize_shared_mem(mem)
            linearized_index = {
                "linearized_idx": linearize_index(node_index, stride_values)
            }
            start_indices, _, _ = _build_start_indices(emitter, linearized_index)

        indices = [offset_th] if buffer_ops_enabled else start_indices
        if is_read:
            return vector_d.load(vector_type, mem, indices)
        else:
            vector_d.store(value, mem, indices)
            return

    zero = get_constant_attr(0, element_type)
    zero = arith_d.constant(element_type, zero)

    if mask is None:
        mask_vec_type = VectorType.get(
            [elements_per_thread], IntegerType.get_signless(1)
        )
        mask = _constant_mask(mask_vec_type)

    # make offsets 0, 1, 2 ...
    offsets_vec_type = VectorType.get(vector_type.shape, IndexType.get())
    vals = [IntegerAttr.get(IndexType.get(), v) for v in range(elements_per_thread)]

    offsets_vec = arith_d.constant(
        offsets_vec_type, DenseElementsAttr.get(vals, offsets_vec_type)
    )

    if buffer_ops_enabled:
        mem, offset_th = _linearize_memref(
            mem, start_indices_wg, start_indices_th, strides
        )
        mem = _cast_buffer_and_encode_stride(mem, strides, element_type, emitter)

    indices = [offset_th] if buffer_ops_enabled else start_indices

    if no_masked_load_store_ops:
        # find the index at which memory out of bounds of buffer
        oob_index_value = _get_out_of_bounds_index(element_type)
        oob_index = arith_d.constant(IndexType.get(), oob_index_value)

        oob_index = vector_d.broadcast(
            VectorType.get(vector_type.shape, IndexType.get()), oob_index
        )

        offset_th = vector_d.broadcast(
            VectorType.get(vector_type.shape, IndexType.get()), offset_th
        )

        uint32_vec_type = VectorType.get([elements_per_thread], uint32)
        indexvec_type = VectorType.get([elements_per_thread], IndexType.get())

        offsets_vec = arith_d.index_cast(uint32_vec_type, offsets_vec)
        offset_th = arith_d.index_cast(uint32_vec_type, offset_th)

        # add the thread offset and the vec offsets
        offsets_vec = arith_d.addi(offsets_vec, offset_th)
        offsets_vec = arith_d.index_cast(indexvec_type, offsets_vec)

        # based on mask, select between the offsets_vec and out of bounds. In this case all 3 operands can be vectors
        selected_index = arith_d.select(mask, offsets_vec, oob_index)
        elems = list()

        if splatted_mask:
            # mask is same for all of them, can just pick the first index
            selected_index = extract(selected_index, 0)

            if is_read:
                return vector_d.load(vector_type, mem, indices=[selected_index])

            else:
                vector_d.store(value, mem, indices=[selected_index])
                return

        for i in range(elements_per_thread):
            # mask is not same for all elements, need to unroll
            this_index = extract(selected_index, i)  # this element

            # Unmasked load, using selected_index
            singlenumvec_type = VectorType.get([1], vector_type.element_type)
            if is_read:
                elem = vector_d.load(singlenumvec_type, mem, indices=[this_index])
                elem = extract(elem, 0)
                elems.append(elem)
            else:
                elem = extract(value, i)
                single_num_vector = vector_d.broadcast(singlenumvec_type, elem)
                vector_d.store(single_num_vector, mem, indices=[this_index])

        if is_read:
            # now make a vector from all the elements loaded
            return vector_d.from_elements(vector_type, elems)

        else:  # it was a store, return
            return

    else:
        # normal masked load/store

        if is_read:
            passthru = vector_d.broadcast(vector_type, zero)
            return vector_d.maskedload(vector_type, mem, indices, mask, passthru)
        else:
            vector_d.maskedstore(mem, indices, mask, value)
            return


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping, dyn_vals, bounds, *rest = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    input_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    dyn_vals = tuple(
        cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
    )
    dynamic_vals_map_start = _build_dyn_vals_map(mapping, dyn_vals)

    mask = _build_mask(emitter, index, elements_per_thread, bounds)
    if mapping:
        index = transform_index_on_mapping(mapping, input_shape, index, is_read=True)

    start_indices, start_indices_wg, start_indices_th = _build_start_indices(
        emitter, index, dynamic_vals_map_start
    )
    if read_meets_hw_transpose_requirements(get_custom(node), emitter.constraints):
        result = amdgpu_d.transpose_load(vector_type, kb_src, start_indices)
    else:
        result = _create_vec_read_write(
            emitter,
            input_shape,
            kb_src,
            None,
            vector_type,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            get_custom(memory),
            mask,
            node_index=index,
        )

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        (
            register,
            memory,
            elements_per_thread,
            mapping,
            dyn_vals,
            bounds,
            *rest,
        ) = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # memory has no IR node yet.
    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    vector_shape = cast_py_literal(emitter, (elements_per_thread,))

    # TODO: Support elements_per_thread size mismatch and broadcasting

    assert (
        tuple(insert_type.shape) == vector_shape
    ), f"Shape doesn't match: {tuple(insert_type.shape)} and {(vector_shape)}"

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected write to have index attr.")

    index = node.index

    input_shape = _get_symbolic_shape(register)
    output_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    dyn_vals = tuple(
        cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
    )
    dynamic_vals_map_start = _build_dyn_vals_map(mapping, dyn_vals)

    mask = _build_mask(emitter, index, elements_per_thread, bounds)
    if mapping:
        index = transform_index_on_mapping(mapping, output_shape, index, is_read=False)

    start_indices, start_indices_wg, start_indices_th = _build_start_indices(
        emitter, index, dynamic_vals_map_start
    )
    _create_vec_read_write(
        emitter,
        output_shape,
        kb_dest,
        insert_vector,
        None,
        start_indices,
        start_indices_wg,
        start_indices_th,
        elements_per_thread,
        get_custom(memory),
        mask,
        node_index=index,
    )


def assume_index_subgroup_uniform(value: Value, element_type: IrType) -> Value:
    original_type = value.type
    idx = arith_d.index_cast(element_type, value)
    # TODO: use a proper ROCDL intrinsic for this after IREE is updated.
    res = llvm_d.call_intrinsic(
        element_type, "llvm.amdgcn.readfirstlane", [idx], [], []
    )
    res = arith_d.index_cast(original_type, res)
    return res


@handle_op(gather_to_lds)
def handle_gather_to_lds(emitter: WaveEmitter, node: fx.Node):
    try:
        (
            src,
            dst,
            src_idx,
            dst_idx,
            element_type,
            elements_per_thread,
            src_mapping,
            dst_mapping,
            src_bounds,
            src_mapping_dyn_vals,
            dst_mapping_dyn_vals,
        ) = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    element_type = IrType.parse(element_type.dtype.ir_type_asm())

    src_symbolic_shape = _get_symbolic_shape(src)
    dst_symbolic_shape = _get_symbolic_shape(dst)

    src = cast_py_value(emitter, src)
    dst = cast_py_value(emitter, dst)
    src_data_type = get_type_or_element_type(src.ir_value.type)
    dst_data_type = get_type_or_element_type(dst.ir_value.type)

    if not (
        MemRefType.isinstance(src.ir_value.type)
        and MemRefType.isinstance(dst.ir_value.type)
    ):
        op = get_custom(node)
        raise ValidationError(
            f"Expected src and dst to be of Memref type for\n"
            f"{op}\nGot\n"
            f"src: {src.ir_value.type}\n"
            f"dst: {dst.ir_value.type}\n"
        )

    if src_data_type != dst_data_type:
        op = get_custom(node)
        raise ValidationError(
            f"Expected src and dst to have same data type for\n"
            f"{op}\nGot\n"
            f"src: {src_data_type} vs dst: {dst_data_type}\n"
        )

    src = src.ir_value
    dst = dst.ir_value
    dynamic_vals_map_start = {}

    if src_mapping:
        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get())
            for reg in src_mapping_dyn_vals
        )
        src_idx = transform_index_on_mapping(src_mapping, src_symbolic_shape, src_idx)
        dynamic_vals_map_start = _build_dyn_vals_map(src_mapping, dyn_vals)
    if dst_mapping:
        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get())
            for reg in dst_mapping_dyn_vals
        )
        dst_idx = transform_index_on_mapping(dst_mapping, dst_symbolic_shape, dst_idx)
        dynamic_vals_map_start = _build_dyn_vals_map(dst_mapping, dyn_vals)

    store_type = VectorType.get((elements_per_thread,), element_type)

    src_index, src_index_wg, src_index_th = _build_start_indices(
        emitter, src_idx, dynamic_vals_map_start
    )

    ip = InsertionPoint.current

    induction_vars = set(emitter.get_induction_vars_and_syms()[1])

    # Hoist to the function level, if not using induction variables.
    if not any(
        induction_vars.intersection(set(index.start.free_symbols))
        for index in dst_idx.values()
    ):
        while not isinstance(ip.block.owner, func_d.FuncOp):
            ip = InsertionPoint(ip.block.owner)

    with ip:
        dst_index, _, _ = _build_start_indices(emitter, dst_idx, dynamic_vals_map_start)
        # We are indexing shared mem so i32 is enough.
        i32 = IntegerType.get_signless(32)
        dst_index = [assume_index_subgroup_uniform(idx, i32) for idx in dst_index]

    strides = strides_from_symbolic_shape(
        IndexingContext.current(), src_symbolic_shape, allow_mixed_shapes=True
    )
    strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in strides]

    src, offset_th = _linearize_memref(src, src_index_wg, src_index_th, strides)
    src = _cast_buffer_and_encode_stride(src, strides, element_type, emitter)

    # We previously checked mask is same for all elements, so we can use
    # elements_per_thread=1 to build the mask.
    mask = _build_mask(
        emitter,
        src_idx,
        elements_per_thread=1,
        bounds=src_bounds,
        dynamic_values=dynamic_vals_map_start,
    )
    if mask:
        mask = vector_d.extract(mask, static_position=[0], dynamic_position=[])
        oob_index_value = _get_out_of_bounds_index(element_type)
        oob_index = arith_d.constant(IndexType.get(), oob_index_value)
        offset_th = arith_d.select(mask, offset_th, oob_index)

    src_index = [offset_th]

    amdgpu_d.gather_to_lds(
        src=src,
        src_indices=src_index,
        dst=dst,
        dst_indices=dst_index,
        transfer_type=store_type,
    )


def _handle_scatter_op(
    emitter: WaveEmitter,
    node: fx.Node,
    rmw_kind: arith_d.AtomicRMWKind,
):
    try:
        (
            register_src,
            register_idx,
            dim,
            memory,
            mapping,
            elements_per_thread,
            bounds,
        ) = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    output_shape = _get_symbolic_shape(memory)
    elements_per_thread = int(cast_py_literal(emitter, elements_per_thread))
    cast_vector(emitter, register_idx, element_type=IndexType.get())

    index_mapping = mapping.map_output_indices(output_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(i.subs(idxc.subs) for i in index_mapping)
    iters = mapping.iters
    index = node.index
    subs = [
        (sym, expr.start) for sym, expr in zip(iters.keys(), index.values())
    ] + list(idxc.subs.items())

    result_index = {key: m.subs(subs) for key, m in zip(output_shape, index_mapping)}

    mask = _build_mask(emitter, index, elements_per_thread, bounds)
    if mask is None:
        mask_vec_type = VectorType.get(
            [elements_per_thread], IntegerType.get_signless(1)
        )
        mask = _constant_mask(mask_vec_type)

    start_indices, start_indices_wg, start_indices_th = _build_start_indices(
        emitter, result_index
    )

    register_idx = cast_py_value(emitter, register_idx).ir_value
    register_src = cast_py_value(emitter, register_src).ir_value
    memory = cast_py_value(emitter, memory).ir_value

    results = []
    for i in range(elements_per_thread):
        index_elem = vector_d.extract(
            register_idx, static_position=[i], dynamic_position=[]
        )
        index_elem = arith_d.index_cast(IndexType.get(), index_elem)
        reg_elem = vector_d.extract(
            register_src, static_position=[i], dynamic_position=[]
        )
        indices = list(start_indices)
        if dim >= len(indices):
            raise ValueError(
                f"Invalid scatter dim {dim} for rank-{len(indices)} memory"
            )

        indices[dim] = index_elem

        # In case 4 elements per thread are used, makes sure values are stored at the right non-scatter dimension
        if elements_per_thread > 1:
            other_dims = [d for d in range(len(indices)) if d != dim]
            if other_dims:
                # Heuristic: offset the innermost (fastest varying) dimension
                # TODO: Ideally emit a vectorized atomic op instead of 4 scalar atomics that store to consecutive locations
                fast_dim = other_dims[-1]
                indices[fast_dim] = arith_d.addi(
                    indices[fast_dim], arith_d.constant(IndexType.get(), i)
                )
        result = memref_d.atomic_rmw(rmw_kind, reg_elem, memory, indices)
        results.append(result)

    result_type = VectorType.get([elements_per_thread], register_src.type.element_type)
    result_vector = vector_d.from_elements(result_type, results)


@handle_op(scatter_add)
def handle_scatter_add(emitter: WaveEmitter, node: fx.Node):
    register_src = cast_py_value(emitter, node.args[0])
    src_data_type = get_type_or_element_type(register_src.ir_value.type)
    if is_float_type(src_data_type):
        rmw_kind = arith_d.AtomicRMWKind.addf
    else:
        rmw_kind = arith_d.AtomicRMWKind.addi
    _handle_scatter_op(emitter, node, rmw_kind)
