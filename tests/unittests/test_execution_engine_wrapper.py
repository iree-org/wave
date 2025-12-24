# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the ExecutionEngine wrapper with weak reference caching.
"""

import gc
import os
import pytest
import weakref
import torch

from wave_lang.kernel.wave.execution_engine import (
    clear_engine_cache,
    get_execution_engine,
    is_engine_cached,
    is_execution_engine_available,
)

pytestmark = pytest.mark.skipif(
    not is_execution_engine_available() or (torch.cuda.device_count() < 1),
    reason="ExecutionEngine not available (C++ extension not built) or no GPU found",
)


ENV_VARS = [
    "WAVE_ENABLE_OBJECT_CACHE",
    "WAVE_ENABLE_GDB_LISTENER",
    "WAVE_ENABLE_PERF_LISTENER",
]


@pytest.fixture
def clean_env():
    """Fixture to clean cache and environment variables before/after each test."""
    clear_engine_cache()

    # Save original env vars
    orig_env = {}
    for key in ENV_VARS:
        orig_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    yield

    # Restore original env vars
    for key, value in orig_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

    clear_engine_cache()
    gc.collect()


def test_basic_creation(clean_env):
    """Test basic creation of execution engine."""
    engine = get_execution_engine()
    assert engine is not None
    assert is_engine_cached()


def test_cache_returns_same_instance(clean_env):
    """Test that cache returns the same instance."""
    engine1 = get_execution_engine()
    engine2 = get_execution_engine()

    # Should be the exact same object
    assert engine1 is engine2


def test_weak_reference_cleanup(clean_env):
    """Test that engine is cleaned up when references are released."""
    # Create engine and get weak reference
    engine = get_execution_engine()
    weak_ref = weakref.ref(engine)

    # Verify it's cached
    assert is_engine_cached()

    # Delete strong reference
    del engine
    gc.collect()

    # Weak reference should be dead
    assert weak_ref() is None

    # Cache should report no engine
    assert not is_engine_cached()


def test_clear_cache(clean_env):
    """Test that clear_cache removes cached engine."""
    engine = get_execution_engine()
    assert is_engine_cached()

    # Clear cache
    clear_engine_cache()

    # Cache should be empty
    assert not is_engine_cached()

    # Engine should still be alive (we have strong reference)
    assert engine is not None


def test_env_var_object_cache(clean_env):
    """Test WAVE_ENABLE_OBJECT_CACHE environment variable."""
    os.environ["WAVE_ENABLE_OBJECT_CACHE"] = "1"

    engine = get_execution_engine()
    assert engine is not None
    # We can't directly test if object cache is enabled without
    # accessing internal state, but we verify the engine was created


def test_env_var_gdb_listener(clean_env):
    """Test WAVE_ENABLE_GDB_LISTENER environment variable."""
    os.environ["WAVE_ENABLE_GDB_LISTENER"] = "1"

    engine = get_execution_engine()
    assert engine is not None


def test_env_var_perf_listener(clean_env):
    """Test WAVE_ENABLE_PERF_LISTENER environment variable."""
    os.environ["WAVE_ENABLE_PERF_LISTENER"] = "1"

    engine = get_execution_engine()
    assert engine is not None


def test_all_env_vars(clean_env):
    """Test all environment variables together."""
    os.environ["WAVE_ENABLE_OBJECT_CACHE"] = "1"
    os.environ["WAVE_ENABLE_GDB_LISTENER"] = "1"
    os.environ["WAVE_ENABLE_PERF_LISTENER"] = "1"

    engine = get_execution_engine()
    assert engine is not None
    assert is_engine_cached()


def test_multiple_references(clean_env):
    """Test that multiple references to the same engine work correctly."""
    engine1 = get_execution_engine()
    engine2 = get_execution_engine()
    engine3 = get_execution_engine()

    # All should be the same object
    assert engine1 is engine2
    assert engine2 is engine3


def test_engine_interface(clean_env):
    """Test that the engine provides the correct interface."""
    engine = get_execution_engine()

    # Check that required methods exist
    assert hasattr(engine, "load_module")
    assert hasattr(engine, "release_module")
    assert hasattr(engine, "lookup")
    assert hasattr(engine, "dump_to_object_file")

    # Check that methods are callable
    assert callable(engine.load_module)
    assert callable(engine.release_module)
    assert callable(engine.lookup)
    assert callable(engine.dump_to_object_file)
