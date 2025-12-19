# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from wave_lang.kernel.wave.utils.run_utils import get_default_arch
from pathlib import Path
from dataclasses import dataclass, field

require_e2e = pytest.mark.require_e2e
expensive_test = pytest.mark.expensive_test
require_gpus = pytest.mark.require_gpus
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
require_gfx1250 = pytest.mark.skipif(
    "gfx125" not in get_default_arch(),
    reason="Default architecture is not GFX1250, default architecture is '{}'".format(
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
    return pytest.mark.parametrize(name, values, ids=ids)


def _is_water_and_ee_available() -> bool:
    from wave_lang.support.detect_water import is_water_available
    from wave_lang.kernel.wave.execution_engine import is_execution_engine_available

    return is_water_available() and is_execution_engine_available()


require_water_and_ee = pytest.mark.skipif(
    not _is_water_and_ee_available(),
    reason="Water or execution engine are not available.",
)


_water_backend_enable = [False, pytest.param(True, marks=require_water_and_ee)]


def use_water_backend_bool(name: str):
    return param_bool(name, "water", values=_water_backend_enable)


def glob_asm_files(path: Path) -> list[Path]:
    return list(filter(lambda x: x.suffix in [".s", ".rocmasm"], path.glob("*")))


@dataclass
class KernelMetadata:
    """Metadata extracted from kernel assembly."""

    vgpr_count: int | None = None
    vgpr_spill_count: int | None = None
    sgpr_count: int | None = None
    sgpr_spill_count: int | None = None
    waitcnt_ops: list[str] = field(default_factory=list)


def extract_kernel_metadata(asm_text: str) -> KernelMetadata:
    """
    Extract kernel metadata from ROCm assembly text.

    Args:
        asm_text: Assembly text content (e.g., from .rocmasm file)

    Returns:
        KernelMetadata containing:
        - vgpr_count: Number of VGPRs allocated
        - vgpr_spill_count: Number of VGPRs spilled
        - sgpr_count: Number of SGPRs allocated
        - sgpr_spill_count: Number of SGPRs spilled
        - waitcnt_ops: List of all waitcnt operations found in the assembly
    """
    import re

    metadata = KernelMetadata()

    # Extract from YAML metadata section (more reliable)
    # Look for patterns like:
    #   .vgpr_count:     3
    #   .vgpr_spill_count: 0
    #   .sgpr_count:     8
    #   .sgpr_spill_count: 0

    vgpr_count_match = re.search(r"\.vgpr_count:\s+(\d+)", asm_text)
    if vgpr_count_match:
        metadata.vgpr_count = int(vgpr_count_match.group(1))

    vgpr_spill_match = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm_text)
    if vgpr_spill_match:
        metadata.vgpr_spill_count = int(vgpr_spill_match.group(1))

    sgpr_count_match = re.search(r"\.sgpr_count:\s+(\d+)", asm_text)
    if sgpr_count_match:
        metadata.sgpr_count = int(sgpr_count_match.group(1))

    sgpr_spill_match = re.search(r"\.sgpr_spill_count:\s+(\d+)", asm_text)
    if sgpr_spill_match:
        metadata.sgpr_spill_count = int(sgpr_spill_match.group(1))

    # Extract all waitcnt operations
    # Pattern: s_waitcnt followed by any arguments
    # Examples: s_waitcnt lgkmcnt(0), s_waitcnt vmcnt(0), etc.
    waitcnt_pattern = re.compile(r"s_waitcnt\s+[^\n]+")
    metadata.waitcnt_ops = waitcnt_pattern.findall(asm_text)

    return metadata
