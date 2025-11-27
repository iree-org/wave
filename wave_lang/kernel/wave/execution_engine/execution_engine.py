# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python wrapper for ExecutionEngine with weak reference caching.

This module provides a simple singleton wrapper around the native ExecutionEngine
that caches a single instance using weak references. Options are configured via
environment variables.
"""

import os
import weakref
from typing import Optional

import ctypes
import platform
from pathlib import Path

try:
    from .wave_execution_engine import ExecutionEngine, ExecutionEngineOptions
except ImportError:
    # Allow import to succeed even if C++ module not built yet
    ExecutionEngine = None
    ExecutionEngineOptions = None


def is_execution_engine_available() -> bool:
    return ExecutionEngine is not None and ExecutionEngineOptions is not None


# Global weak reference to the cached ExecutionEngine instance
_cached_engine: Optional[weakref.ref] = None


def _load_library(lib_basename: str):
    """
    Load a shared library from the execution_engine directory.

    Args:
        lib_basename: Base name of library (e.g., "wave_runtime_helpers")

    Returns:
        ctypes.CDLL library handle

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """

    # Find the library file
    lib_name = {
        "Linux": f"lib{lib_basename}.so",
        "Darwin": f"lib{lib_basename}.dylib",
        "Windows": f"{lib_basename}.dll",
    }.get(platform.system())

    if not lib_name:
        raise RuntimeError(f"Unsupported platform: {platform.system()}.")

    # Look for the library in the same directory as this module
    module_dir = Path(__file__).parent
    lib_path = module_dir / lib_name

    if not lib_path.exists():
        raise RuntimeError(f"{lib_basename} library not found at {lib_path}.")

    return ctypes.CDLL(str(lib_path))


def _get_symbol(lib: ctypes.CDLL, name: str) -> ctypes.c_void_p:
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value


def _add_symbol(symbol_map: dict, lib: ctypes.CDLL, name: str) -> None:
    symbol_map[name] = _get_symbol(lib, name)


def _load_runtime_helpers():
    """
    Load the wave_runtime_helpers shared library and return symbol addresses.

    This library contains runtime helper functions like wave_get_buffer that are
    needed by JIT-compiled code.

    Returns:
        Dictionary mapping symbol names to their addresses

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """
    lib = _load_library("wave_runtime_helpers")

    symbol_map = {}

    _add_symbol(symbol_map, lib, "_mlir_ciface_wave_get_buffer")
    _add_symbol(symbol_map, lib, "_mlir_ciface_wave_get_int64")
    _add_symbol(symbol_map, lib, "_mlir_ciface_wave_get_float64")
    _add_symbol(symbol_map, lib, "_mlir_ciface_wave_get_dim")

    return symbol_map


def _load_hip_runtime():
    """
    Load the wave_hip_runtime shared library and return symbol addresses.

    This library contains HIP runtime functions for kernel loading and launching.

    Returns:
        Dictionary mapping symbol names to their addresses

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """
    lib = _load_library("wave_hip_runtime")

    lib.load_functions()

    symbol_map = {}

    _add_symbol(symbol_map, lib, "wave_load_kernel")
    _add_symbol(symbol_map, lib, "wave_launch_kernel")

    return symbol_map


def _create_options_from_env() -> "ExecutionEngineOptions":
    """
    Create ExecutionEngineOptions from environment variables.

    Environment Variables:
        WAVE_ENABLE_OBJECT_CACHE: Enable object cache (default: 0)
        WAVE_ENABLE_GDB_LISTENER: Enable GDB notification listener (default: 0)
        WAVE_ENABLE_PERF_LISTENER: Enable Perf notification listener (default: 0)

    Returns:
        ExecutionEngineOptions configured from environment
    """
    if ExecutionEngineOptions is None:
        raise RuntimeError(
            "wave_execution_engine module not available. "
            "Ensure the C++ extension is built and installed."
        )

    options = ExecutionEngineOptions()

    # Read options from environment variables
    def _env_enabled(var: str, default: str = "0") -> bool:
        return bool(int(os.environ.get(var, default)))

    options.enable_object_cache = _env_enabled("WAVE_ENABLE_OBJECT_CACHE")
    options.enable_gdb_notification_listener = _env_enabled("WAVE_ENABLE_GDB_LISTENER")
    options.enable_perf_notification_listener = _env_enabled(
        "WAVE_ENABLE_PERF_LISTENER"
    )

    return options


def get_execution_engine() -> "ExecutionEngine":
    """
    Get or create the global ExecutionEngine instance.

    This function maintains a single cached ExecutionEngine instance using
    weak references. If the cached instance has been garbage collected, a
    new one is created. Options are configured via environment variables.

    Returns:
        ExecutionEngine instance

    Example:
        >>> engine = get_execution_engine()
        >>> handle = engine.load_module(my_mlir_module)
        >>> func_ptr = engine.lookup(handle, "my_function")
        >>> engine.release_module(handle)
    """
    global _cached_engine

    # Try to get cached instance
    if _cached_engine is not None:
        engine = _cached_engine()
        if engine is not None:
            return engine

    symbol_map = {}

    # Load runtime helpers (wave_get_buffer, etc.)
    symbol_map.update(_load_runtime_helpers())

    # Load HIP runtime (wave_load_kernel, wave_launch_kernel, etc.)
    symbol_map.update(_load_hip_runtime())

    options = _create_options_from_env()

    if symbol_map:
        options.set_symbol_map(symbol_map)

    engine = ExecutionEngine(options)

    _cached_engine = weakref.ref(engine)

    return engine


def clear_engine_cache():
    """
    Clear the cached execution engine instance.

    Note: The engine will only be destroyed if there are no other
    references to it. If you're holding a reference, the engine
    will remain alive until that reference is released.
    """
    global _cached_engine
    _cached_engine = None


def is_engine_cached() -> bool:
    """
    Check if an execution engine is currently cached.

    Returns:
        True if an engine is cached and still alive, False otherwise
    """
    global _cached_engine
    if _cached_engine is None:
        return False
    return _cached_engine() is not None


__all__ = [
    "get_execution_engine",
    "clear_engine_cache",
    "is_engine_cached",
]
