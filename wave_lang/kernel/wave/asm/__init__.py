# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
MLIR to AMDGCN assembly emitter package.
"""

from .driver import main
from .asm_emitter import AsmEmitter
from .mlir_walker import IRWalker
from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess
from .register_allocator import RegFile, SGPRAllocator, VGPRAllocator
from .instructions import (
    Instruction,
    MemoryInstruction,
    ArithmeticInstruction,
    ControlFlowInstruction,
    WaitInstruction,
    LoadInstruction,
    StoreInstruction,
    SLoadDwordx2,
    BufferLoadDwordx4,
    BufferStoreDwordx4,
    SMovB32,
    VMbcntLoU32B32,
    VMbcntHiU32B32,
    VLshlRevB32,
    VMovB32,
    VMulLoU32,
    VAddU32,
    SWaitcnt,
    SEndpgm,
    InstructionBuilder,
    emit_kernargs,
    emit_srd_setup,
    emit_vector_load_store,
)
from .utils import (
    parse_vector_type,
    parse_memref_type,
    parse_vector_type_from_obj,
    parse_memref_type_from_obj,
    attrs_to_dict,
    parse_wg_and_subgroup,
    tid_upper_bound_from_thread_id,
    simplify_expression,
    emit_expression_asm,
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
    "Instruction",
    "MemoryInstruction",
    "ArithmeticInstruction",
    "ControlFlowInstruction",
    "WaitInstruction",
    "LoadInstruction",
    "StoreInstruction",
    "SLoadDwordx2",
    "BufferLoadDwordx4",
    "BufferStoreDwordx4",
    "SMovB32",
    "VMbcntLoU32B32",
    "VMbcntHiU32B32",
    "VLshlRevB32",
    "VMovB32",
    "VMulLoU32",
    "VAddU32",
    "SWaitcnt",
    "SEndpgm",
    "InstructionBuilder",
    "emit_kernargs",
    "emit_srd_setup",
    "emit_vector_load_store",
    "parse_vector_type",
    "parse_memref_type",
    "parse_vector_type_from_obj",
    "parse_memref_type_from_obj",
    "attrs_to_dict",
    "parse_wg_and_subgroup",
    "tid_upper_bound_from_thread_id",
    "simplify_expression",
    "emit_expression_asm",
]
