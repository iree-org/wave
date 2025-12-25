# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
pytest configuration for ASM backend tests.

This conftest enables stricter validation during testing to catch issues early.
"""

import os
import pytest


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
