# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Operation handlers for MLIR operations in the ASM backend.

This module contains handlers for various MLIR operations that are encountered
during the IR traversal for assembly code generation.
"""

import operator

import sympy

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    gpu_d,
    memref_d,
    rocdl_d,
    scf_d,
    stream_d,
    vector_d,
)

from .utils import (
    parse_vector_type_from_obj,
    parse_memref_type_from_obj,
    tid_upper_bound_from_thread_id,
    simplify_expression,
    split_const_dynamic,
)
from .gather_to_shared import G2SHandler

from .kernel_model import KernelInfo, MemRefInfo, BindingUse, VecAccess



__all__ = [n for n in globals().keys() if not n.startswith("__")]
