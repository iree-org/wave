# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
pytest configuration for ASM backend tests.

This conftest enables stricter validation during testing to catch issues early
and provides CLI options / fixtures for WaveASM e2e tests.
"""

import os
import pytest


def pytest_addoption(parser):
    """Add WaveASM-specific command line options."""
    parser.addoption(
        "--backend",
        action="store",
        default="cpp",
        choices=["cpp", "python", "both"],
        help="WaveASM backend to use: cpp (default), python, or both (compare)",
    )
    parser.addoption(
        "--dump-asm",
        action="store_true",
        default=False,
        help="Dump assembly files to /tmp for debugging",
    )


def pytest_configure(config):
    """
    Enable strict formatter validation for all ASM backend tests.

    The strict formatter validates operand counts and types, catching
    formatting errors before they reach the assembler. This is the last
    line of defense before assembly compilation, so validation here
    catches bugs that would otherwise produce cryptic assembler errors.

    Set WAVE_STRICT_FORMATTER=0 to disable if debugging a specific issue.
    """
    if os.environ.get("WAVE_STRICT_FORMATTER") is None:
        os.environ["WAVE_STRICT_FORMATTER"] = "1"


@pytest.fixture
def backend(request):
    """Get the selected backend from command line."""
    return request.config.getoption("--backend")


@pytest.fixture
def dump_asm(request):
    """Get the dump-asm flag from command line."""
    return request.config.getoption("--dump-asm")
