# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import partial


from wave_lang.support.ir_imports import (
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    Float4E2M1FNType,
    Float6E2M3FNType,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    Float8E8M0FNUType,
    FloatAttr,
    IndexType,
    IntegerType,
    RankedTensorType,
    VectorType,
    arith_d,
    vector_d,
)

###############################################################################
# Helpers
###############################################################################


# API name  inspired by mlir/python/mlir/dialects/_arith_ops_ext.py
def is_float_type(type):
    return isinstance(
        type,
        (
            BF16Type,
            F16Type,
            F32Type,
            F64Type,
            Float8E4M3FNType,
            Float8E4M3FNUZType,
            Float8E5M2Type,
            Float8E5M2FNUZType,
            Float8E8M0FNUType,
            Float6E2M3FNType,
            Float4E2M1FNType,
        ),
    )


def is_index_type(type):
    return isinstance(type, (IndexType))


def is_integer_like_type(type):
    return isinstance(type, (IntegerType, IndexType))


def is_signed_or_signless_type(type):
    return getattr(type, "is_signed", False) or getattr(type, "is_signless", False)


def get_conversion_op(src_elem_type, dst_elem_type, fastmath=None):
    is_src_float = is_float_type(src_elem_type)
    is_dst_float = is_float_type(dst_elem_type)
    is_src_int = is_integer_like_type(src_elem_type)
    is_dst_int = is_integer_like_type(dst_elem_type)
    if (
        is_src_int
        and is_dst_int
        and (is_index_type(src_elem_type) or is_index_type(dst_elem_type))
    ):
        conversion_op = arith_d.index_cast
        return conversion_op
    # Special case of casting bool (IntergerType(i1) to float) so that when value is true is casted to 1 and when value is false to cast to 0
    if (
        isinstance(src_elem_type, IntegerType)
        and src_elem_type.width == 1
        and is_dst_float
    ):

        def bool_to_float_select(dst_type, vector_src):

            # scalar constants
            one_const = arith_d.constant(
                dst_elem_type, FloatAttr.get(dst_elem_type, 1.0)
            )
            zero_const = arith_d.constant(
                dst_elem_type, FloatAttr.get(dst_elem_type, 0.0)
            )

            # Broadcast to vector if the destination is a vector
            if VectorType.isinstance(dst_type):
                one = vector_d.broadcast(dst_type, one_const)
                zero = vector_d.broadcast(dst_type, zero_const)
            elif RankedTensorType.isinstance(dst_type):
                raise NotImplementedError(
                    "RankedTensorType broadcasting is not implemented for casting bool to float."
                )
            else:
                one = one_const
                zero = zero_const

            return arith_d.select(vector_src, one, zero)

        # return caller function for get_conversion op
        return bool_to_float_select

    conversion_ops = {
        (True, False): arith_d.fptosi,
        (False, True): arith_d.sitofp,
    }

    float_cast_ops = {
        True: arith_d.extf,
        False: arith_d.truncf,
    }

    int_cast_ops = {
        True: arith_d.extsi,
        False: arith_d.trunci,
    }

    if is_src_float and is_dst_float:
        conversion_op = float_cast_ops[src_elem_type.width < dst_elem_type.width]
        conversion_op = partial(conversion_op, fastmath=fastmath)
    elif is_src_int and is_dst_int:
        # Currently extsi/trunci do not support fast_math option.
        conversion_op = int_cast_ops[src_elem_type.width < dst_elem_type.width]
    else:
        conversion_op = conversion_ops[(is_src_float, is_dst_float)]
    return conversion_op
