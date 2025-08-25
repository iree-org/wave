# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.kernel.wave.utils.symbol_utils import get_induction_symbol
from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    get_custom,
    Write,
    Placeholder,
    DebugLog,
)
from .._support.dtype import DataType
from .._support.indexing import IndexSymbol
from ..lang.wave_types import IndexMapping, Memory
from .compile_options import WaveCompileOptions
from .constraints import Constraint, TilingConstraint
from typing import TypedDict, Any
import sympy


class DebugArgInfo(TypedDict):
    symbol_name: str
    debug_output_arg_id: int
    dtype: DataType
    symbolic_shape: tuple[IndexSymbol, ...]
    printer: Any


def is_debug_log_transformer(node):
    return isinstance(get_custom(node), DebugLog)


def debug_log_hoist(trace: CapturedTrace, debug_handlers: list[Any]):
    """
    Finds debug log ops and hoists kernel inputs for them.
    Must be run near the very beginning of the compilation pipeline.

    The idea of this feature is that there are debug_log_transformer ops that are logically writes to global memory.
    However, these debug ops are intended to conveniently transform the kernel and the launcher code to automatically insert the extra global memory arguments and possibly process them (eg. printing logged data in the launcher).
    So the user can add a single debug print op to a Wave kernel on one line without changing any other lines of code to get rich printing functionality handled by a mix of kernel and python code.
    """
    root = trace.get_root_graph()
    with root.inserting_before(None):
        debug_log_ops = trace.walk(is_debug_log_transformer)
        for index, debug_op in enumerate(debug_log_ops):
            custom = get_custom(debug_op)
            if custom.handler:
                debug_handlers.append(custom.handler)
            placeholder_name = custom.label or f"debug_log_output_{index}"
            type_expr = None
            placeholder = Placeholder(placeholder_name, type_expr).add_to_graph(root)
            custom.fx_node.memory = placeholder
            placeholder.meta["debug_output_arg_id"] = index
            placeholder.meta["symbol_name"] = placeholder_name
            placeholder.meta["printer"] = custom.printer
            placeholder.meta["extra_iteration_dimensions"] = (
                custom.extra_iteration_dimensions or []
            )


def debug_log_write_replace(
    trace: CapturedTrace,
    constraints: list[Constraint],
    options: WaveCompileOptions,
    debug_arg_info: list[DebugArgInfo],
):
    """
    This pass has 3 jobs:

    * Replace all debug_log_transformers with write operations.
    * Set metadata on the debug output Placeholder objects (created in debug_log_hoist pass).
    * Collect debug_arg_info so that the kernel invoke method can be massaged to automatically handle debug log outputs.  It mutates the input argument so that the data can be passed to the WaveKernel object.

    Must be run after type checking, but before indexing.
    """
    root = trace.get_root_graph()

    # Part 1: replace debug log transformers with write operations.
    debug_log_ops = trace.walk(is_debug_log_transformer)
    for index, debug_op in enumerate(debug_log_ops):
        graph = debug_op.graph
        doc = get_custom(debug_op)

        mapping = doc.mapping
        # If there are extra iteration dimensions, we need to create the
        # dimensions (including TilingConstraint for it) and add a mapping to
        # write to different slots along this dimension for each iteration.
        if doc.extra_iteration_dimensions:
            # Create an identity mapping if no mapping currently exists,
            # otherwise adding new dimensions would be invalid.
            if mapping is None:
                reg_type = get_custom(doc.register_).type
                input_shape = reg_type.symbolic_shape
                num_dims = len(input_shape)

                input_mapping = {
                    dim: IndexMapping.iterator(i) for i, dim in enumerate(input_shape)
                }
                output_mapping = {
                    dim: IndexMapping.iterator(i) for i, dim in enumerate(input_shape)
                }

                mapping = IndexMapping(num_dims, input_mapping, output_mapping)

            new_output_mapping = dict(mapping.output_mapping)
            new_dyn_vals = list(mapping.dynamic_val_mappings) or []
            for dim_symbol, iter_axis, size in doc.extra_iteration_dimensions:
                tiling_constraint = TilingConstraint(dim_symbol)
                constraints.append(tiling_constraint)
                options.dynamic_symbols.append(dim_symbol)
                new_dyn_vals.append(dim_symbol)

                induction_var = get_induction_symbol(iter_axis)
                new_output_mapping[dim_symbol] = sympy.Min(induction_var, size - 1)

            mapping = IndexMapping(
                mapping.num_iterators,
                mapping.input_mapping,
                new_output_mapping,
                new_dyn_vals,
            )

        with graph.inserting_before(debug_op):
            new_write = Write(
                doc.register_,
                doc.memory,
                mapping=mapping,
                mapping_dynamic_vals=doc.mapping_dynamic_vals,
            ).add_to_graph(graph)
            get_custom(new_write).infer_type()
        doc.erase()

    # Part 2: add metadata to debug output Placeholders
    def is_debug_log_placeholder(node):
        return isinstance(get_custom(node), Placeholder) and (
            node.meta.get("debug_output_arg_id", None) != None
        )

    def is_arg_placeholder(node):
        return isinstance(get_custom(node), Placeholder) and ("arg_id" in node.meta)

    root = trace.get_root_graph()
    arg_placeholders = trace.walk(is_arg_placeholder)
    debug_placeholder_ops = trace.walk(is_debug_log_placeholder)
    for debug_placeholder_op in debug_placeholder_ops:
        debug_placeholder_op.meta["arg_id"] = (
            len(arg_placeholders) + debug_placeholder_op.meta["debug_output_arg_id"]
        )
        custom = get_custom(debug_placeholder_op)
        debug_placeholder_op.meta["dtype"] = custom.type.dtype

        symbolic_shape = custom.type.symbolic_shape
        extra_iter_dims = debug_placeholder_op.meta.get(
            "extra_iteration_dimensions", None
        )

        if extra_iter_dims:
            extra_symbols = tuple(dim_symbol for dim_symbol, _, _ in extra_iter_dims)
            symbolic_shape = extra_symbols + symbolic_shape
            new_type = Memory[symbolic_shape, GLOBAL_ADDRESS_SPACE, custom.type.dtype]
            custom.type = new_type
            debug_placeholder_op.type = new_type

        debug_placeholder_op.meta["symbolic_shape"] = symbolic_shape
        # Insert in front since the placeholders are in reverse order compared to the log ops in source code.
        debug_arg_info.insert(0, debug_placeholder_op.meta)
