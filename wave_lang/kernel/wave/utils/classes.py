# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import List
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


class AttentionOperationType(Enum):
    """Enumeration of attention operation types for prefetch stages."""

    MMA_0 = "mma_0"
    MMA_1 = "mma_1"
    SOFTMAX_0 = "softmax_0"
    SOFTMAX_1 = "softmax_1"
    LOCAL_LOAD_0 = "local_load_0"
    GLOBAL_LOAD_0 = "global_load_0"
    LOCAL_STORE_0 = "local_store_0"
    LOCAL_LOAD_1 = "local_load_1"
    GLOBAL_LOAD_1 = "global_load_1"
    LOCAL_STORE_1 = "local_store_1"

    @classmethod
    def get_all_types(cls) -> List["AttentionOperationType"]:
        """Get all operation types as a list."""
        return list(cls)

    @classmethod
    def from_string(cls, value: str) -> "AttentionOperationType":
        """Create an enum value from a string, with error handling."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown attention operation type: {value}")

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value
