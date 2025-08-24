# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *

from ..wave.constraints import Constraint, DeviceConstraint


def _substitute_recursive(obj, subs_dict):
    if isinstance(obj, (list, tuple)):
        new_obj = [_substitute_recursive(item, subs_dict) for item in obj]
        return tuple(new_obj) if isinstance(obj, tuple) else new_obj
    elif isinstance(obj, dict):
        return {
            key: _substitute_recursive(value, subs_dict) for key, value in obj.items()
        }
    elif hasattr(obj, "symbolic_shape"):
        obj.symbolic_shape = _substitute_recursive(obj.symbolic_shape, subs_dict)
        return obj
    else:
        # Check if obj is in subs_dict only if it's hashable
        try:
            if obj in subs_dict:
                return subs_dict[obj]
        except TypeError:
            # obj is not hashable, skip this check
            pass

    return obj


def reshape_per_device(trace: CapturedTrace, constraints: list[Constraint]) -> None:
    """
    Reshape the input and output tensors in the kernel to match the device constraints

    While the kernel authors write the kernel using high-level symbols (M, N, K), the kernel
    is compiler for one device. The device constraints specify the tile sizes the kernel should be optimized for.
    This function substitutes the high-level symbols with the device-specific tile sizes.
    """
    device_constraints = [x for x in constraints if isinstance(x, DeviceConstraint)]

    if len(device_constraints) == 0:
        # No device constraints, nothing to do
        return

    # Create a mapping from dimension symbols to tile sizes per device
    # for example, {M: DEVICE_M, N: DEVICE_N}
    symbol_map = {}
    for constraint in device_constraints:
        symbol_map[constraint.dim] = constraint.tile_size

    # Walk through all the nodes in the graph and substitute the dim symbols if they have a device specific tile size.
    # Need to be thorough as the kernel authors may use the symbols in all kinds of manner in the kernel code.
    # For example, through trace.walk() we may encounter a Register or Memory type that uses the dim symbols which are
    # then updated to device constrainted tile sizes through _substitute_recursive.
    # For example:
    # Before we may have some nodes like:
    # placeholder(_name=a, _type=Memory[M, K].of(f16)) type(Memory[M, K].of(f16))
    # placeholder(_name=b, _type=Memory[N, K].of(f16)) type(Memory[N, K].of(f16))
    # register(shape=(M, N), dtype=f32, value=0.0) type(None)
    # After substituting the symbols we will get:
    # placeholder(_name=a, _type=Memory[DEVICE_M, K].of(f16)) type(Memory[DEVICE_M, K].of(f16))
    # placeholder(_name=b, _type=Memory[N, K].of(f16)) type(Memory[N, K].of(f16))
    # register(shape=(DEVICE_M, N), dtype=f32, value=0.0) type(None)
    for node in trace.walk():
        node.args = _substitute_recursive(node.args, symbol_map)
        node.kwargs = _substitute_recursive(node.kwargs, symbol_map)

        if node.type:
            _substitute_recursive(node.type, symbol_map)

        if node.meta:
            node.meta = _substitute_recursive(node.meta, symbol_map)

    return
