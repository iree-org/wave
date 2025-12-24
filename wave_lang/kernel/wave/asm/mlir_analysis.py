# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Shared MLIR analysis helpers for the ASM backend.

Goal: keep MLIR scanning logic (function selection, translation_info parsing,
and workgroup-id detection) in ONE place to avoid drift across:
- `AsmEmitter.from_mlir_string`
- `driver.py`
- `KernelModuleCompiler.compile_mlir_string`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from wave_lang.support.ir_imports import Operation, func_d, gpu_d, OpAttributeMap

from .utils import parse_wg_and_subgroup


def walk_ops_recursively(operation: Operation) -> Iterable[Operation]:
    """Recursively walk all operations in an MLIR operation tree."""
    for region in operation.regions:
        for block in region.blocks:
            for inner_operation in block.operations:
                yield inner_operation
                yield from walk_ops_recursively(inner_operation)


def should_skip_function(fn: func_d.FuncOp) -> bool:
    """Return True if this function should not be treated as a kernel."""
    name = fn.sym_name.value
    # Skip async wrapper variants and benchmark scaffolding.
    return name.startswith("isolated_benchmark") or name.endswith("$async")


def detect_needed_workgroup_ids(fn: func_d.FuncOp) -> Tuple[bool, bool, bool]:
    """
    Scan MLIR function to detect which workgroup IDs are needed.

    Returns:
        (needs_wgid_x, needs_wgid_y, needs_wgid_z)
    """
    needs_x, needs_y, needs_z = False, False, False

    def walk_ops(op):
        nonlocal needs_x, needs_y, needs_z

        if isinstance(op, gpu_d.BlockIdOp):
            dim_str = str(op.dimension)
            if "dim x" in dim_str:
                needs_x = True
            elif "dim y" in dim_str:
                needs_y = True
            elif "dim z" in dim_str:
                needs_z = True

        if hasattr(op, "regions"):
            for region in op.regions:
                for block in region.blocks:
                    for inner_op in block.operations:
                        walk_ops(inner_op)

    walk_ops(fn)
    return (needs_x, needs_y, needs_z)


@dataclass(frozen=True)
class TranslationInfo:
    wg_size: Tuple[int, int, int]
    subgroup_size: int


def extract_translation_info(fn: func_d.FuncOp) -> TranslationInfo:
    """Extract (wg_size, subgroup_size) from translation_info attributes."""
    # Defaults align with existing code paths.
    wg_size: Tuple[int, int, int] = (64, 1, 1)
    subgroup_size: int = 64

    function_attributes = (
        dict(fn.attributes) if isinstance(fn.attributes, OpAttributeMap) else {}
    )
    translation_info = function_attributes.get("translation_info")
    if translation_info is not None:
        workgroup_size_tuple, sg_size = parse_wg_and_subgroup(translation_info)
        if workgroup_size_tuple:
            # parse_wg_and_subgroup returns a tuple already normalized by upstream
            # conventions (1-3 dims). Ensure 3-tuple.
            if len(workgroup_size_tuple) == 3:
                wg_size = workgroup_size_tuple
            elif len(workgroup_size_tuple) == 2:
                wg_size = (workgroup_size_tuple[0], workgroup_size_tuple[1], 1)
            elif len(workgroup_size_tuple) == 1:
                wg_size = (workgroup_size_tuple[0], 1, 1)
        if sg_size:
            subgroup_size = sg_size

    return TranslationInfo(wg_size=wg_size, subgroup_size=subgroup_size)


