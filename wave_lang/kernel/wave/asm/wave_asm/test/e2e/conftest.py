# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pytest configuration for WaveASM E2E tests."""

import os
import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end GPU tests",
    )
    parser.addoption(
        "--keep-temp",
        action="store_true",
        default=False,
        help="Keep temporary files for debugging",
    )
    parser.addoption(
        "--target",
        default=None,
        help="Target GPU architecture (e.g., gfx942, gfx950)",
    )
    parser.addoption(
        "--backend",
        action="store",
        default="cpp",
        choices=["cpp", "python", "both"],
        help="Backend to use: cpp (default), python, or both (compare)",
    )
    parser.addoption(
        "--dump-asm",
        action="store_true",
        default=False,
        help="Dump assembly files to /tmp for debugging",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires GPU)"
    )
    config.addinivalue_line(
        "markers", "translation: mark test as translation-only (no GPU)"
    )


@pytest.fixture
def run_e2e(request):
    """Fixture to check if e2e tests are enabled."""
    return request.config.getoption("--run-e2e")


@pytest.fixture
def keep_temp(request):
    """Fixture to check if temp files should be kept."""
    return request.config.getoption("--keep-temp")


@pytest.fixture
def target_arch(request):
    """Fixture to get target architecture."""
    target = request.config.getoption("--target")
    if target:
        return target

    # Check environment
    if "WAVE_DEFAULT_ARCH" in os.environ:
        return os.environ["WAVE_DEFAULT_ARCH"]

    # Try to detect from GPU
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if hasattr(props, 'gcnArchName'):
                return props.gcnArchName
    except Exception:
        pass

    return "gfx942"  # Default


@pytest.fixture
def backend(request):
    """Get the selected backend from command line."""
    return request.config.getoption("--backend")


@pytest.fixture
def dump_asm(request):
    """Get the dump-asm flag from command line."""
    return request.config.getoption("--dump-asm")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if not config.getoption("--run-e2e"):
        # Skip e2e tests unless explicitly enabled
        skip_e2e = pytest.mark.skip(reason="Need --run-e2e to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
