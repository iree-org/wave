# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compatibility wrapper for the kernel IR pipeline.

This file used to contain the full implementation (>2k LOC). It now re-exports
`KernelCompilationContext` and `KernelModuleCompiler` from split modules to keep
all ASM-backend files under 1000 LOC.
"""

from .kernel_compilation_context import KernelCompilationContext
from .kernel_module_compiler import KernelModuleCompiler

__all__ = ["KernelCompilationContext", "KernelModuleCompiler"]
