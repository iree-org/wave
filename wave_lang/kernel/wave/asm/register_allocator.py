# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Register file and allocators for AMDGCN assembly generation.
"""

from typing import Tuple, List, Union, Optional
from enum import Enum
import os


# Global debug flag for allocator logging (set DEBUG_ASM_ALLOCATOR=1 to enable)
DEBUG_ALLOCATOR = os.environ.get("DEBUG_ASM_ALLOCATOR", "0") == "1"


class RegisterOp(Enum):
    """Register operation types for logging."""

    ALLOC_V = "alloc_v"
    ALLOC_V_PAIR = "alloc_v_pair"
    ALLOC_V_QUAD = "alloc_v_quad"
    FREE_V = "free_v"
    FREE_V_PAIR = "free_v_pair"
    FREE_V_QUAD = "free_v_quad"
    RESERVE = "reserve"
    ALLOC_S = "alloc_s"
    FREE_S = "free_s"


def _log_alloc(msg: str):
    """Log allocator operations if debug flag is enabled."""
    if DEBUG_ALLOCATOR:
        import sys

        print(f"[ALLOCATOR] {msg}", file=sys.stderr)


class RegFile:
    def __init__(self):
        self.s_max = 1  # s[0:1] reserved for kernarg ptr
        self.v_used = set()  # No implicit reservations
        self.a_used = set()  # Track AGPR usage
        self.a_max = -1  # Track maximum AGPR index used

        # Track allocation history for debugging (enabled via DEBUG_ALLOCATOR)
        self.allocation_history: List[str] = (
            []
        )  # Human-readable log of alloc/free events


class SGPRAllocator:
    def __init__(self, register_file: RegFile):
        self.register_file = register_file
        self.next_sgpr = 2
        self.allocated_sgprs: set = set()  # Track allocated SGPRs for validation

    def alloc_s(self) -> int:
        """Allocate a single SGPR."""
        register = self.next_sgpr
        self.next_sgpr += 1
        self.register_file.s_max = max(self.register_file.s_max, register)
        self.allocated_sgprs.add(register)
        _log_alloc(f"{RegisterOp.ALLOC_S.value}: s{register}")
        self.register_file.allocation_history.append(
            f"{RegisterOp.ALLOC_S.value}: s{register}"
        )
        return register

    def free_s(self, register: int):
        """Free a single SGPR (for future optimization - currently a no-op)."""
        # Validation: Check for double-free
        if register not in self.allocated_sgprs:
            _log_alloc(
                f"WARNING: Attempted to free unallocated/already-freed SGPR s{register}"
            )
        else:
            self.allocated_sgprs.discard(register)
            _log_alloc(f"{RegisterOp.FREE_S.value}: s{register}")
            self.register_file.allocation_history.append(
                f"{RegisterOp.FREE_S.value}: s{register}"
            )
        # Simple allocator doesn't reuse SGPRs yet
        pass

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

        # Free lists for LIFO reuse per size class
        self.free_list_singles = []  # List of freed single VGPRs
        self.free_list_pairs = []  # List of freed VGPR pairs (tuples)
        self.free_list_quads = []  # List of freed VGPR quads (tuples)

        # Track peak VGPR usage for metadata
        self.v_peak = -1

    def reserve(self, reg: int):
        """Mark a register as reserved (e.g., for special purposes)."""
        if reg in self.reserved:
            _log_alloc(f"WARNING: v{reg} already reserved, skipping duplicate reserve")
            return
        self.reserved.add(reg)
        self.register_file.v_used.add(reg)
        self.v_peak = max(self.v_peak, reg)
        _log_alloc(f"{RegisterOp.RESERVE.value}: v{reg}")
        self.register_file.allocation_history.append(
            f"{RegisterOp.RESERVE.value}: v{reg}"
        )

    def _try_reuse(
        self, free_list: List, op: RegisterOp
    ) -> Optional[Union[int, Tuple]]:
        """Try to reuse registers from a free list."""
        while free_list:
            regs = free_list.pop()
            # Convert to tuple for uniform handling
            regs_tuple = (regs,) if isinstance(regs, int) else regs

            # Check if any register is reserved or in use
            if not any(
                r in self.reserved or r in self.register_file.v_used for r in regs_tuple
            ):
                # Mark as used
                self.register_file.v_used.update(regs_tuple)

                # Log the reuse
                if isinstance(regs, int):
                    _log_alloc(f"{op.value}: v{regs} (reused)")
                    self.register_file.allocation_history.append(
                        f"{op.value}: v{regs} (reused)"
                    )
                else:
                    _log_alloc(f"{op.value}: v[{regs[0]}:{regs[-1]}] (reused)")
                    self.register_file.allocation_history.append(
                        f"{op.value}: v[{regs[0]}:{regs[-1]}] (reused)"
                    )

                return regs
        return None

    def _allocate_new(
        self, count: int, alignment: int, op: RegisterOp
    ) -> Union[int, Tuple]:
        """Allocate new register(s) with given count and alignment."""
        # Align base register
        base_reg = (self.base + alignment - 1) & ~(alignment - 1)

        # Find free aligned block
        while any(
            (base_reg + i) in self.register_file.v_used
            or (base_reg + i) in self.reserved
            for i in range(count)
        ):
            base_reg += alignment

        # Create register tuple
        if count == 1:
            regs = base_reg
            regs_tuple = (base_reg,)
        elif count == 2:
            regs = (base_reg, base_reg + 1)
            regs_tuple = regs
        else:  # count == 4
            regs = (base_reg, base_reg + 1, base_reg + 2, base_reg + 3)
            regs_tuple = regs

        # Mark as used and update peak
        self.register_file.v_used.update(regs_tuple)
        self.v_peak = max(self.v_peak, base_reg + count - 1)

        # Log allocation
        if count == 1:
            _log_alloc(f"{op.value}: v{regs} (new, v_peak={self.v_peak})")
            self.register_file.allocation_history.append(f"{op.value}: v{regs} (new)")
        else:
            _log_alloc(
                f"{op.value}: v[{regs[0]}:{regs[-1]}] (new, v_peak={self.v_peak})"
            )
            self.register_file.allocation_history.append(
                f"{op.value}: v[{regs[0]}:{regs[-1]}] (new)"
            )

        return regs

    def alloc_v(self) -> int:
        """Allocate a single VGPR with reuse from free list."""
        result = self._try_reuse(self.free_list_singles, RegisterOp.ALLOC_V)
        if result is not None:
            return result
        return self._allocate_new(count=1, alignment=1, op=RegisterOp.ALLOC_V)

    def alloc_v_pair(self) -> Tuple[int, int]:
        """Allocate a pair of VGPRs (aligned to 2) with reuse."""
        result = self._try_reuse(self.free_list_pairs, RegisterOp.ALLOC_V_PAIR)
        if result is not None:
            return result
        return self._allocate_new(count=2, alignment=2, op=RegisterOp.ALLOC_V_PAIR)

    def alloc_v_quad(self) -> Tuple[int, int, int, int]:
        """Allocate a quad of VGPRs (aligned to 4) with reuse."""
        result = self._try_reuse(self.free_list_quads, RegisterOp.ALLOC_V_QUAD)
        if result is not None:
            return result
        return self._allocate_new(count=4, alignment=4, op=RegisterOp.ALLOC_V_QUAD)

    def _free_registers(
        self, regs: Union[int, Tuple], free_list: List, op: RegisterOp
    ) -> None:
        """Generic register freeing with validation."""
        # Normalize to tuple for uniform handling
        regs_tuple = (regs,) if isinstance(regs, int) else regs

        # Validate: check if any register is reserved
        if any(r in self.reserved for r in regs_tuple):
            if isinstance(regs, int):
                _log_alloc(f"WARNING: Attempted to free reserved VGPR v{regs}")
            else:
                _log_alloc(
                    f"WARNING: Attempted to free reserved VGPRs v[{regs[0]}:{regs[-1]}]"
                )
            return

        # Validate: check if all registers are actually in use
        if not all(r in self.register_file.v_used for r in regs_tuple):
            if isinstance(regs, int):
                _log_alloc(
                    f"WARNING: Attempted to free unallocated/already-freed VGPR v{regs}"
                )
            else:
                _log_alloc(
                    f"WARNING: Attempted to free unallocated/already-freed VGPRs v[{regs[0]}:{regs[-1]}]"
                )
            return

        # Free all registers atomically
        for reg in regs_tuple:
            self.register_file.v_used.discard(reg)

        # Add to free list for reuse
        free_list.append(regs)

        # Log the operation
        if isinstance(regs, int):
            _log_alloc(f"{op.value}: v{regs}")
            self.register_file.allocation_history.append(f"{op.value}: v{regs}")
        else:
            _log_alloc(f"{op.value}: v[{regs[0]}:{regs[-1]}]")
            self.register_file.allocation_history.append(
                f"{op.value}: v[{regs[0]}:{regs[-1]}]"
            )

    def free_v(self, reg: int):
        """Free a single VGPR."""
        self._free_registers(reg, self.free_list_singles, RegisterOp.FREE_V)

    def free_v_pair(self, regs: Tuple[int, int]):
        """Free a pair of VGPRs."""
        self._free_registers(regs, self.free_list_pairs, RegisterOp.FREE_V_PAIR)

    def free_v_quad(self, regs: Tuple[int, int, int, int]):
        """Free a quad of VGPRs."""
        self._free_registers(regs, self.free_list_quads, RegisterOp.FREE_V_QUAD)

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
