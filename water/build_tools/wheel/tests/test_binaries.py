# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
from water_mlir.binaries import find_binary
from water_mlir import ir
from water_mlir.dialects import wave


def test_find_binary():
    subprocess.check_call([find_binary("water-opt"), "--version"])


def test_wave_python_binding():
    with ir.Context() as ctx:
        wave.register_dialect(ctx)
