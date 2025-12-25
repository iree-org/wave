# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.kernel.wave.asm.metadata_emitter import get_register_granularity
from wave_lang.kernel.wave.asm.utils import normalize_wg_size
from wave_lang.kernel.wave.asm.abi import system_vgpr_workitem_id_from_wg_size


def test_normalize_wg_size() -> None:
    assert normalize_wg_size((64,)) == (64, 1, 1)
    assert normalize_wg_size((64, 4)) == (64, 4, 1)
    assert normalize_wg_size((64, 4, 2)) == (64, 4, 2)


def test_register_granularity_targets() -> None:
    # CDNA3 family used by tests
    assert get_register_granularity("gfx942") == (8, 8)
    assert get_register_granularity("gfx950") == (8, 8)
    # Legacy-ish fallback: ensure the default rule remains (vgpr=4, sgpr=8)
    assert get_register_granularity("gfx90a") == (4, 8)


def test_system_vgpr_workitem_id_policy() -> None:
    assert system_vgpr_workitem_id_from_wg_size((64, 1, 1)) == 0
    assert system_vgpr_workitem_id_from_wg_size((64, 2, 1)) == 1
    assert system_vgpr_workitem_id_from_wg_size((64, 1, 2)) == 1
