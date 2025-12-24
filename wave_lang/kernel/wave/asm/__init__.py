# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
MLIR to AMDGCN assembly emitter package.

Architecture:
    InstructionRegistry (YAML) -> InstructionFormatter -> Assembly Text
                                        ^
                                        |
    UnifiedEmitter (virtual emission) --+
    KernelGenerator (physical rendering)
    MetadataEmitter (assembler directives)

All physical instruction formatting goes through InstructionFormatter.
"""

from .driver import main
from .asm_emitter import AsmEmitter  # Legacy, will be removed
from .mlir_walker import IRWalker
from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator
from .instructions import Instruction
from .unified_emitter import UnifiedEmitter, EmissionMode
from .instruction_registry import get_registry, InstructionDef, InstructionCategory
from .instruction_formatter import InstructionFormatter, get_formatter
from .metadata_emitter import MetadataEmitter, KernelMetadata, create_metadata
from .kernel_pipeline import KernelCompilationContext
from .kernel_generator import KernelGenerator, PhysicalMapping
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
    # Core compilation
    "KernelCompilationContext",
    "KernelGenerator",
    "PhysicalMapping",
    # Metadata emission
    "MetadataEmitter",
    "KernelMetadata",
    "create_metadata",
    # Instruction formatting
    "InstructionFormatter",
    "get_formatter",
    # Unified emitter
    "UnifiedEmitter",
    "EmissionMode",
    # Instruction registry
    "get_registry",
    "InstructionDef",
    "InstructionCategory",
    # MLIR walker
    "IRWalker",
    # Kernel model
    "KernelInfo",
    "MemRefInfo",
    "BindingUse",
    "VecAccess",
    # Register allocation
    "RegFile",
    "SGPRAllocator",
    "VGPRAllocator",
    # Legacy (deprecated)
    "AsmEmitter",
    "Instruction",
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
