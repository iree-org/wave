# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for creating LLVM debug information attributes necessary for DWARF output.
LLVM requires more than just file/line/col locations to produce DWARF location output.
This file is for attaching these extra attributes like DICompileUnitAttr as FusedLocs to the overall kernel.

It seems that the python bindings don't yet support the LLVM dialect DIAttr
bindings, so we use string parsing for now.
"""

import os
from typing import Callable, Optional

from wave_lang.support.ir_imports import (
    Attribute,
    Location,
    llvm_d,
)
from wave_lang.kernel._support.location import FileLineColInfo, StackTraceInfo


def create_di_compile_unit(
    get_next_distinct_id: Callable[[], int], di_file: Attribute
) -> Attribute:
    distinct_id = get_next_distinct_id()
    # The sourceLanguage field requires a non-blank value from a fixed enum.
    # We choose DW_LANG_C here because C is commonly used as a default for DSLs
    # and is widely supported by debuggers and tooling.
    di_cu_str = (
        f"#llvm.di_compile_unit<"
        f"id = distinct[{distinct_id}]<>, "
        f"sourceLanguage = DW_LANG_C, "
        f"file = {di_file}, "
        f'producer = "Wave Compiler", '
        f"isOptimized = false, "
        f"emissionKind = LineTablesOnly>"
    )

    return Attribute.parse(di_cu_str)


def create_di_subprogram(
    get_next_distinct_id: Callable[[], int],
    di_compile_unit: Attribute,
    di_file: Attribute,
    di_subroutine_type: Attribute,
    function_name: str,
    line_number: int,
    scope_line: Optional[int] = None,
    subprogram_flags: llvm_d.DISubprogramFlags = llvm_d.DISubprogramFlags.Definition,
) -> Attribute:
    if scope_line is None:
        scope_line = line_number

    distinct_id = get_next_distinct_id()
    di_sp_str = (
        f"#llvm.di_subprogram<"
        f"id = distinct[{distinct_id}]<>, "
        f"compileUnit = {di_compile_unit}, "
        f"scope = {di_file}, "
        f'name = "{function_name}", '
        f'linkageName = "{function_name}", '
        f"file = {di_file}, "
        f"line = {line_number}, "
        f"scopeLine = {scope_line}, "
        f"subprogramFlags = {subprogram_flags.name}, "
        f"type = {di_subroutine_type}>"
    )
    return Attribute.parse(di_sp_str)


def create_debug_info_for_kernel(
    get_next_distinct_id: Callable[[], int],
    function_name: str,
    location_info: Optional[FileLineColInfo | StackTraceInfo] = None,
) -> Location:
    """
    To output locations in the final DWARF, LLVM needs the kernel to have a
    DISubprogram attr, which also includes some other attrs (DIFile,
    DICompileUnit, DISubroutineType, DISubprogram).  This function creates a
    FusedLoc that has the necessary attributes for the kernel function.

    Args:
        get_next_distinct_id: Callable that returns the next distinct ID
        function_name: Name of the function
        location_info: Location information captured from the source

    Returns:
        location_with_di: FusedLoc with DISubprogram attached
    """
    if location_info is None:
        return Location.unknown()

    # we need to dig some info out of the location, so we need to unwrap StackTraceInfo
    if isinstance(location_info, StackTraceInfo):
        primary_loc = location_info.frames[0] if location_info.frames else None
    else:
        primary_loc = location_info

    full_path = primary_loc.filename
    filename = os.path.basename(full_path)
    directory = os.path.dirname(os.path.abspath(full_path))

    # Extract line number (handle tuples for ranges)
    if isinstance(primary_loc.line, tuple):
        line_number = primary_loc.line[0]
    else:
        line_number = primary_loc.line

    # Convert to MLIR location
    base_location = location_info.to_mlir()

    # Create debug attributes
    di_file_str = f'#llvm.di_file<"{filename}" in "{directory}">'
    di_file = Attribute.parse(di_file_str)

    di_compile_unit = create_di_compile_unit(get_next_distinct_id, di_file)
    # This is a stand-in subroutine type with no args for now, but when we have
    # variable tracking we should update this to have real types.
    di_subroutine_type = Attribute.parse(
        "#llvm.di_subroutine_type<callingConvention = DW_CC_normal>"
    )

    di_subprogram = create_di_subprogram(
        get_next_distinct_id,
        di_compile_unit,
        di_file,
        di_subroutine_type,
        function_name,
        line_number,
    )

    location_with_di = Location.fused([base_location], metadata=di_subprogram)

    return location_with_di
