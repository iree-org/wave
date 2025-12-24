# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
MLIR to AMDGCN assembly emitter package.

For instruction emission, use the UnifiedEmitter class or AsmEmitter.unified property.
Instruction definitions are in instruction_defs/*.yaml files.
"""

from .driver import main
from .asm_emitter import AsmEmitter
from .mlir_walker import IRWalker
from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator
from .instructions import Instruction
from .unified_emitter import UnifiedEmitter, EmissionMode
from .instruction_registry import get_registry, InstructionDef, InstructionCategory
from .utils import (
    parse_vector_type,
    parse_memref_type,
    parse_vector_type_from_obj,
    parse_memref_type_from_obj,
    attrs_to_dict,
    parse_wg_and_subgroup,
    tid_upper_bound_from_thread_id,
    simplify_expression,
)

__all__ = [
    "main",
    "AsmEmitter",
    "IRWalker",
    "KernelInfo",
    "MemRefInfo",
    "BindingUse",
    "VecAccess",
    "RegFile",
    "SGPRAllocator",
    "VGPRAllocator",
    # Base instruction class (for backwards compatibility)
    "Instruction",
    # Unified emitter infrastructure
    "UnifiedEmitter",
    "EmissionMode",
    "get_registry",
    "InstructionDef",
    "InstructionCategory",
    # Utilities
    "parse_vector_type",
    "parse_memref_type",
    "parse_vector_type_from_obj",
    "parse_memref_type_from_obj",
    "attrs_to_dict",
    "parse_wg_and_subgroup",
    "tid_upper_bound_from_thread_id",
    "simplify_expression",
]
