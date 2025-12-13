# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc

import pytest
import torch


"""
This is a special file that is used by pytest to inject a fixture into all tests
in this package.
"""


@pytest.fixture(scope="module", autouse=True)
def cleanup_memory():
    """Auto-apply fixture that forces garbage collection and releases torch
    cache memory after each module."""
    yield

    # Code after yield runs after the test.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
