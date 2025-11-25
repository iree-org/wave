# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import subprocess
from wave_lang.kernel.wave.water import find_binary, is_water_available


@pytest.mark.skipif(
    not is_water_available(), reason="Water MLIR package not installed."
)
def test_find_binary():
    subprocess.check_call([find_binary("water-opt"), "--version"])
