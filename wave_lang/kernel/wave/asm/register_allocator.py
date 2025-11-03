# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Register file and allocators for AMDGCN assembly generation.
"""

from typing import Tuple


class RegFile:
    def __init__(self):
        self.s_max = 1  # s[0:1] reserved for kernarg ptr
        self.v_used = set([0])  # v0 used by lane id; v2 will be used on demand


class SGPRAllocator:
    def __init__(self, register_file: RegFile):
        self.register_file = register_file
        self.next_sgpr = 2

    def pair(self) -> Tuple[int, int]:
        first_register, second_register = self.next_sgpr, self.next_sgpr + 1
        self.next_sgpr += 2
        self.register_file.s_max = max(self.register_file.s_max, second_register)
        return first_register, second_register

    def quad(self) -> Tuple[int, int, int, int]:
        # Ensure alignment to 4 SGPRs for SRD requirements
        self.next_sgpr = (self.next_sgpr + 3) & (-4)
        register_0, register_1, register_2, register_3 = (
            self.next_sgpr,
            self.next_sgpr + 1,
            self.next_sgpr + 2,
            self.next_sgpr + 3,
        )
        self.next_sgpr += 4
        self.register_file.s_max = max(self.register_file.s_max, register_3)
        return register_0, register_1, register_2, register_3


class VGPRAllocator:
    def __init__(self, register_file: RegFile, base=2):
        self.register_file = register_file
        self.base = max(base, 2)  # we will use v2 as the standard voffset register

    def quad(self) -> Tuple[int, int, int, int]:
        # find next free aligned to 4 and not conflicting with reserved v2
        base_register = max(self.base, 4)
        while any(((base_register + i) in self.register_file.v_used) for i in range(4)):
            base_register += 4
        registers = (
            base_register,
            base_register + 1,
            base_register + 2,
            base_register + 3,
        )
        self.register_file.v_used.update(registers)
        self.base = base_register + 4
        return registers
