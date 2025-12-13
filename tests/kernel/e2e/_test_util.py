# Copyright 2024-2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os

import pytest


default_test_shapes = [
    (1, 27),
    (111, 813),
    (1, 128),
    (256, 64),
    (256, 128),
    (256, 256),
    (256, 1024),
]

user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)


def mark_shapes_skip(src_shapes, skip_shapes, reason=None):
    """Mark certain shapes to be skipped in parametrized tests."""
    mark = lambda *a: pytest.param(*a, marks=pytest.mark.skip(reason=reason))
    return [(mark(s) if s in skip_shapes else s) for s in src_shapes]


if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    """Get test shapes for a given test name from config or defaults."""
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes
