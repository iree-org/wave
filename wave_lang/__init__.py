"""
The wave package provides a python dsl for authoring
high performance machine learning kernels on GPUs.
"""

# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util

msg = """No module named 'torch'. Follow https://pytorch.org/get-started/locally/#start-locally to install 'torch'.
For example, on Linux to install with CPU support run:
  pip3 install torch --index-url https://download.pytorch.org/whl/cpu
"""

if importlib.util.find_spec("torch") is None:
    raise ModuleNotFoundError(msg)
