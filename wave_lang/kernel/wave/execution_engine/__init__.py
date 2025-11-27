# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
ExecutionEngine module for Wave JIT compilation.

This module provides both low-level C++ bindings and a high-level Python wrapper
for the LLVM-based execution engine used by Wave. The wrapper caches a single
ExecutionEngine instance using weak references and configures it via environment
variables.
"""

# Import Python wrapper with caching
from .execution_engine import (
    ExecutionEngine,
    ExecutionEngineOptions,
    clear_engine_cache,
    get_execution_engine,
    is_engine_cached,
    is_execution_engine_available,
)

__all__ = [
    # C++ bindings
    "ExecutionEngine",
    "ExecutionEngineOptions",
    # Python wrapper
    "is_execution_engine_available",
    "get_execution_engine",
    "clear_engine_cache",
    "is_engine_cached",
]
