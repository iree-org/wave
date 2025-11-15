# Copyright 2025 The IREE Authors
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

try:
    from wave_execution_engine import ExecutionEngine, ExecutionEngineOptions
except ImportError:
    # Allow import to succeed even if C++ module not built yet
    ExecutionEngine = None
    ExecutionEngineOptions = None


# Global weak reference to the cached ExecutionEngine instance
_cached_engine: Optional[weakref.ref] = None


def _get_wave_get_buffer_address():
    """
    Get the address of the wave_get_buffer function.

    This function is defined in buffer_utils.cpp and needs to be accessible
    to JIT-compiled code.

    Returns:
        Integer address of wave_get_buffer, or None if not available
    """
    import ctypes
    import sys

    # Try to find wave_get_buffer in the current process
    # It should be loaded as part of the wave_execution_engine extension
    try:
        # Get handle to current process
        if sys.platform == "linux":
            RTLD_DEFAULT = ctypes.cast(0, ctypes.c_void_p)
            libc = ctypes.CDLL(None)
            dlsym = libc.dlsym
            dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            dlsym.restype = ctypes.c_void_p

            addr = dlsym(RTLD_DEFAULT, b"wave_get_buffer")
            if addr:
                return addr
        elif sys.platform == "darwin":
            # macOS
            RTLD_DEFAULT = ctypes.cast(-2, ctypes.c_void_p)
            libc = ctypes.CDLL(None)
            dlsym = libc.dlsym
            dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            dlsym.restype = ctypes.c_void_p

            addr = dlsym(RTLD_DEFAULT, b"wave_get_buffer")
            if addr:
                return addr
    except Exception:
        pass

    return None


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

    # Create new instance with options from environment
    options = _create_options_from_env()
    engine = ExecutionEngine(options)

    # Cache using weak reference
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
