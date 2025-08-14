# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum


@dataclass
class KernelLaunchInfo:
    grid: tuple[int] = None
    blocks: tuple[int] = None
    shared_memory_bytes: int = 0
    func_name: str = ""
    grid_str: str = ""


############################################
# Wave Ops related Utils
############################################


# GPU shuffle modes
class ShuffleMode(Enum):
    XOR = 0
    DOWN = 1
    UP = 2
    IDX = 3


class DPPMode(Enum):
    QUAD_PERM = 0
    WAVE_SHL = 1
    WAVE_SHR = 2
    WAVE_ROR = 3
    WAVE_ROL = 4
    ROW_SHL = 5
    ROW_SHR = 6
    ROW_ROR = 7
    ROW_MIRROR = 8
    ROW_HALF_MIRROR = 9
    ROW_BCAST_15 = 10
    ROW_BCAST_31 = 11


class SubgroupReduceMode(Enum):
    ADD = 0
    MUL = 1
    MINUI = 2
    MINSI = 3
    MAXUI = 4
    MAXSI = 5
    AND = 6
    OR = 7
    XOR = 8
    MINNUMF = 9
    MAXNUMF = 10
    MINIMUMF = 11
    MAXIMUMF = 12
