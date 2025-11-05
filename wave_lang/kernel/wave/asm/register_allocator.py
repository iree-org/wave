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
        self.v_used = set()  # No implicit reservations
        self.a_used = set()  # Track AGPR usage
        self.a_max = -1  # Track maximum AGPR index used


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
    def __init__(self, register_file: RegFile, base=0):
        self.register_file = register_file
        self.base = base
        self.reserved = set()  # Reserved registers that should never be allocated

    def reserve(self, reg: int):
        """Mark a register as reserved (e.g., for special purposes)."""
        self.reserved.add(reg)
        self.register_file.v_used.add(reg)

    def alloc_v(self) -> int:
        """Allocate a single VGPR."""
        reg = self.base
        while reg in self.register_file.v_used or reg in self.reserved:
            reg += 1
        self.register_file.v_used.add(reg)
        return reg

    def alloc_v_pair(self) -> Tuple[int, int]:
        """Allocate a pair of VGPRs (aligned to 2)."""
        base_reg = (self.base + 1) & ~1  # Align to 2
        while any(
            (base_reg + i) in self.register_file.v_used
            or (base_reg + i) in self.reserved
            for i in range(2)
        ):
            base_reg += 2
        regs = (base_reg, base_reg + 1)
        self.register_file.v_used.update(regs)
        return regs

    def alloc_v_quad(self) -> Tuple[int, int, int, int]:
        """Allocate a quad of VGPRs (aligned to 4)."""
        base_reg = (self.base + 3) & ~3  # Align to 4
        while any(
            (base_reg + i) in self.register_file.v_used
            or (base_reg + i) in self.reserved
            for i in range(4)
        ):
            base_reg += 4
        regs = (base_reg, base_reg + 1, base_reg + 2, base_reg + 3)
        self.register_file.v_used.update(regs)
        return regs

    def free_v(self, reg: int):
        """Free a single VGPR."""
        if reg in self.reserved:
            return  # Don't free reserved registers
        self.register_file.v_used.discard(reg)

    def free_v_pair(self, regs: Tuple[int, int]):
        """Free a pair of VGPRs."""
        for reg in regs:
            if reg not in self.reserved:
                self.register_file.v_used.discard(reg)

    def free_v_quad(self, regs: Tuple[int, int, int, int]):
        """Free a quad of VGPRs."""
        for reg in regs:
            if reg not in self.reserved:
                self.register_file.v_used.discard(reg)

    # Legacy method for compatibility
    def quad(self) -> Tuple[int, int, int, int]:
        """Legacy quad allocation (kept for compatibility)."""
        return self.alloc_v_quad()


class AGPRAllocator:
    """
    Allocator for Accumulation GPRs (AGPRs) used by MFMA instructions.
    AGPRs are typically allocated in quads for matrix operations.
    """

    def __init__(self, register_file: RegFile, base=0):
        self.register_file = register_file
        self.base = base

    def alloc_a_quad(self) -> Tuple[int, int, int, int]:
        """Allocate a quad of AGPRs (aligned to 4) for MFMA results."""
        base_reg = (self.base + 3) & ~3  # Align to 4
        while any((base_reg + i) in self.register_file.a_used for i in range(4)):
            base_reg += 4
        regs = (base_reg, base_reg + 1, base_reg + 2, base_reg + 3)
        self.register_file.a_used.update(regs)
        self.register_file.a_max = max(self.register_file.a_max, base_reg + 3)
        return regs

    def free_a_quad(self, regs: Tuple[int, int, int, int]):
        """Free a quad of AGPRs."""
        for reg in regs:
            self.register_file.a_used.discard(reg)
