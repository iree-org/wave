# Copyright 2026 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from wave_lang.kernel.wave.utils.general_utils import check_leaks
import torch
import pytest


def test_check_leaks():
    live = []
    try:
        os.environ["WAVE_CHECK_LEAKS"] = "1"

        @check_leaks
        def test_inner():
            nonlocal live
            live.append(torch.tensor([1, 2, 3]))

        with pytest.raises(RuntimeError):
            test_inner()
    finally:
        os.environ.pop("WAVE_CHECK_LEAKS", None)
