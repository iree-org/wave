# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os


def find_binary(name: str) -> str:
    this_path = os.path.dirname(__file__)
    tool_path = os.path.join(this_path, "bin", name)
    assert os.path.isfile(tool_path) and os.access(
        tool_path, os.X_OK
    ), f"Could not find the {name} executable in package."
    return tool_path
