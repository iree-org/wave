# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from wave_lang.kernel.wave.utils.run_utils import get_default_arch

require_e2e = pytest.mark.require_e2e
expensive_test = pytest.mark.expensive_test
require_cdna2 = pytest.mark.skipif(
    "gfx90" not in get_default_arch(),
    reason="Default architecture is not CDNA2, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(),
    reason="Default architecture is not CDNA3, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna4 = pytest.mark.skipif(
    "gfx95" not in get_default_arch(),
    reason="Default architecture is not CDNA4, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna_3_or_4 = pytest.mark.skipif(
    "gfx94" not in get_default_arch() and "gfx95" not in get_default_arch(),
    reason="Default architecture is not CDNA3 or CDNA4, default architecture is '{}'".format(
        get_default_arch()
    ),
)
# ISAs = ["gfx90", "gfx94", "gfx95"]
require_cdna_2_or_3_or_4 = pytest.mark.skipif(
    "gfx90" not in get_default_arch()
    and "gfx94" not in get_default_arch()
    and "gfx95" not in get_default_arch(),
    reason="Default architecture is not CDNA2 or CDNA3 or CDNA4, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_rdna4 = pytest.mark.skipif(
    "gfx120" not in get_default_arch(),
    reason="Default architecture is not RDNA4, default architecture is '{}'".format(
        get_default_arch()
    ),
)

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
expensive_test_param = lambda *a: pytest.param(*a, marks=pytest.mark.expensive_test)


def param_bool(name, shortname=None, values=None):
    shortname = shortname or name
    values = values or [False, True]
    ids = [f"{shortname}" if v else f"no_{shortname}" for v in values]
    return pytest.mark.parametrize(name, [pytest.param(v) for v in values], ids=ids)
