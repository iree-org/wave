# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Gather-to-Shared (G2S) handling for buffer_load...lds instructions.

This module contains all functionality related to gather_to_lds operations:
- Scheduling and M0 pre-computation
- SRD tracing through memref cast chains
- VGPR offset computation
- The main gather_to_lds handler
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, TYPE_CHECKING

import sympy

from wave_lang.support.ir_imports import amdgpu_d, MemRefType, VectorType

from .instructions import (
    BufferLoadDwordLds,
    BufferLoadDwordx4Lds,
    SMovB32,
    SMovB32M0,
    SAndB32,
    SOrB32,
    VMovB32,
    VReadfirstlaneB32,
)
from .utils import parse_vector_type_from_obj, parse_memref_type_from_obj

if TYPE_CHECKING:
    from .kernel_model import KernelInfo
    from .handlers import OperationHandlers

# =============================================================================
# Constants
# =============================================================================

# Default LDS column count for 2D memrefs
_DEFAULT_LDS_COLS = 32

# Default cache swizzle stride (LLVM uses 0x40400000 = stride 64)
_DEFAULT_CACHE_SWIZZLE_STRIDE = 64

# Cache swizzle enable bit (bit 30)
_CACHE_SWIZZLE_ENABLE = 0x40000000

# SRD word2: maximum buffer size
_SRD_MAX_BUFFER_SIZE = 0x7FFFFFFD

# SRD word3: data_format=4 (32-bit), num_format=7, add_tid_enable=1
_SRD_WORD3_LDS = 0x27000

# Element size scale for f16 (bytes per element)
_F16_ELEM_BYTES = 2


# =============================================================================
# Scheduling functions
# =============================================================================


@dataclass
class G2SSchedule:
    """Scheduling info for g2s region."""

    first_g2s_idx: int
    g2s_ops: List[Any]


def analyze_g2s_region(ops: List[Any]) -> Optional[G2SSchedule]:
    """Find gather_to_lds operations and their scheduling info."""
    g2s_ops = []
    first_g2s_idx = None

    for i, op in enumerate(ops):
        if isinstance(op, amdgpu_d.GatherToLDSOp):
            if first_g2s_idx is None:
                first_g2s_idx = i
            g2s_ops.append(op)

    if not g2s_ops:
        return None

    return G2SSchedule(first_g2s_idx=first_g2s_idx, g2s_ops=g2s_ops)


def precompute_m0_values(
    schedule: G2SSchedule,
    kernel_info: "KernelInfo",
    handlers: "OperationHandlers",
) -> None:
    """Pre-compute M0 values for all g2s ops before any buffer_load...lds."""
    for g2s_op in schedule.g2s_ops:
        handlers.g2s.precompute_g2s_m0(g2s_op, kernel_info)


# =============================================================================
# G2SHandler
# =============================================================================


class G2SHandler:
    """Handles gather_to_lds operations for the ASM backend."""

    def __init__(self, handlers: "OperationHandlers"):
        self.handlers = handlers
        self.walker = handlers.walker

    # -------------------------------------------------------------------------
    # SRD tracing
    # -------------------------------------------------------------------------

    def _trace_memref_to_srd(
        self, memref_ssa: str, kernel_info: "KernelInfo"
    ) -> Tuple[Optional[tuple], Optional[str]]:
        """Trace through cast chain to find the original memref with an SRD."""
        srds = self.walker.emitter.srds
        fat_sources = getattr(self.walker, "_fat_buffer_sources", {})
        cast_sources = getattr(self.walker, "_memref_cast_sources", {})

        current = memref_ssa
        visited = set()

        while current and current not in visited:
            visited.add(current)

            # Found SRD
            if current in srds:
                return srds[current], current

            # Follow cast chain
            if current in fat_sources:
                info = fat_sources[current]
                current = (
                    info.get("source_ssa", current) if isinstance(info, dict) else info
                )
            elif current in cast_sources:
                current = cast_sources[current]
            else:
                break

        return None, None

    # -------------------------------------------------------------------------
    # SRD allocation and emission
    # -------------------------------------------------------------------------

    def _allocate_aligned_srd(self) -> Tuple[int, int, int, int]:
        """Allocate 4-aligned SGPRs for an SRD."""
        emitter = self.walker.emitter
        temp_sgprs = []
        while True:
            s = emitter.sgpr_allocator.alloc_s()
            if s % 4 == 0:
                srd = (
                    s,
                    emitter.sgpr_allocator.alloc_s(),
                    emitter.sgpr_allocator.alloc_s(),
                    emitter.sgpr_allocator.alloc_s(),
                )
                break
            temp_sgprs.append(s)
        for s in temp_sgprs:
            emitter.sgpr_allocator.free_s(s)
        return srd

    def _compute_cache_swizzle_bits(self, src_memref_ssa: str) -> int:
        """Compute cache swizzle bits from fat_raw_buffer_cast info."""
        fat_buffer_info = getattr(self.walker, "_fat_buffer_sources", {}).get(
            src_memref_ssa
        )
        stride = _DEFAULT_CACHE_SWIZZLE_STRIDE
        if fat_buffer_info and isinstance(fat_buffer_info, dict):
            stride = fat_buffer_info.get(
                "cache_swizzle_stride", _DEFAULT_CACHE_SWIZZLE_STRIDE
            )

        if stride > 0:
            return _CACHE_SWIZZLE_ENABLE | (1 << (int(math.log2(stride)) + 16))
        return 0

    def _emit_srd_copy(
        self,
        srd_regs: Tuple[int, int, int, int],
        orig_srd_regs: tuple,
        cache_swizzle_bits: int,
    ) -> None:
        """Emit SRD copy with cache swizzle modifications."""
        emitter = self.walker.emitter
        s0, s1, s2, s3 = srd_regs

        emitter.emit_instruction(SMovB32(s0, f"s{orig_srd_regs[0]}", "SRD word0"))

        if cache_swizzle_bits == 0:
            emitter.emit_instruction(SMovB32(s1, f"s{orig_srd_regs[1]}", "SRD word1"))
        else:
            emitter.emit_instruction(SAndB32(s1, f"s{orig_srd_regs[1]}", 0xFFFF))
            emitter.emit_instruction(
                SOrB32(s1, f"s{s1}", cache_swizzle_bits, "cache swizzle")
            )

        emitter.emit_instruction(SMovB32(s2, _SRD_MAX_BUFFER_SIZE, "SRD word2"))
        emitter.emit_instruction(SMovB32(s3, _SRD_WORD3_LDS, "SRD word3"))

    def _get_or_create_srd(
        self, orig_srd_regs: tuple, src_memref_ssa: str
    ) -> Tuple[int, int, int, int]:
        """Get cached SRD or create a new one for gather_to_lds."""
        cache_key = f"lds_srd_copy_{orig_srd_regs[0]}"
        srd_copies = getattr(self.walker, "_lds_srd_copies", None)
        if srd_copies is None:
            self.walker._lds_srd_copies = {}
            srd_copies = self.walker._lds_srd_copies

        if cache_key in srd_copies:
            return srd_copies[cache_key]

        srd_regs = self._allocate_aligned_srd()
        cache_swizzle_bits = self._compute_cache_swizzle_bits(src_memref_ssa)
        self._emit_srd_copy(srd_regs, orig_srd_regs, cache_swizzle_bits)
        srd_copies[cache_key] = srd_regs
        return srd_regs

    # -------------------------------------------------------------------------
    # Offset computation
    # -------------------------------------------------------------------------

    def _compute_vgpr_offset(
        self, index_ssa: str, kernel_info: "KernelInfo", scale_bytes: int
    ) -> int:
        """Compute a VGPR containing the offset value, scaled to bytes."""
        value = kernel_info.index_env.get(index_ssa)

        if isinstance(value, int):
            voffset = self.walker.emitter.vgpr_allocator.alloc_v()
            self.walker.emitter.emit_instruction(
                VMovB32(voffset, value * scale_bytes, "offset")
            )
            return voffset

        if isinstance(value, sympy.Expr):
            scaled_expr = value * scale_bytes if scale_bytes != 1 else value
            return int(
                self.handlers._get_expr_emitter(kernel_info).get_or_emit(scaled_expr)[
                    1:
                ]
            )

        raise ValueError(f"Cannot compute offset for {index_ssa}: value={value}")

    def _compute_vaddr(self, src_index_ssa: str, kernel_info: "KernelInfo") -> int:
        """Compute vaddr VGPR for global memory offset."""
        return self._compute_vgpr_offset(
            src_index_ssa, kernel_info, scale_bytes=_F16_ELEM_BYTES
        )

    # -------------------------------------------------------------------------
    # M0 computation
    # -------------------------------------------------------------------------

    def _get_lds_layout(self, dst_memref_type) -> Tuple[int, int]:
        """Extract LDS columns and element bytes from destination memref type."""
        if not isinstance(dst_memref_type, MemRefType):
            raise ValueError(f"Unsupported destination memref type: {dst_memref_type}")
        shape, _, elem_bytes = parse_memref_type_from_obj(dst_memref_type)
        lds_cols = shape[1] if len(shape) >= 2 else _DEFAULT_LDS_COLS
        return lds_cols, elem_bytes

    def _emit_m0(
        self,
        operation,
        kernel_info: "KernelInfo",
        lds_base_bytes: int,
        lds_cols: int,
        elem_bytes: int,
    ) -> None:
        """Emit M0 register setup for LDS destination."""
        emitter = self.walker.emitter

        # Check for pre-computed M0
        m0_sgprs = getattr(self.walker, "_m0_sgprs", None)
        if m0_sgprs:
            precomputed = m0_sgprs.get(id(operation))
            if precomputed is not None:
                emitter.emit_instruction(
                    SMovB32M0(f"s{precomputed}", "Pre-computed M0")
                )
                return

        # Compute M0 inline from row/col indices
        dst_row = kernel_info.index_env.get(str(operation.operands[3]))
        dst_col = kernel_info.index_env.get(str(operation.operands[4]))

        # Build M0 expression: lds_base + (row * cols + col) * elem_bytes
        row = (
            dst_row if isinstance(dst_row, sympy.Expr) else sympy.Integer(dst_row or 0)
        )
        col = (
            dst_col if isinstance(dst_col, sympy.Expr) else sympy.Integer(dst_col or 0)
        )
        m0_expr = sympy.simplify(lds_base_bytes + (row * lds_cols + col) * elem_bytes)

        # Emit: either constant or expression -> readfirstlane -> M0
        if m0_expr.is_number:
            emitter.emit_instruction(SMovB32M0(int(m0_expr), "LDS offset"))
        else:
            m0_vreg = int(
                self.handlers._get_expr_emitter(kernel_info).get_or_emit(m0_expr)[1:]
            )
            s_tmp = emitter.sgpr_allocator.alloc_s()
            emitter.emit_instruction(VReadfirstlaneB32(s_tmp, m0_vreg))
            emitter.emit_instruction(SMovB32M0(f"s{s_tmp}"))
            emitter.sgpr_allocator.free_s(s_tmp)

    def _emit_buffer_load(
        self, transfer_bytes: int, voffset: int, srd_regs: tuple
    ) -> None:
        """Emit buffer_load...lds instruction."""
        emitter = self.walker.emitter

        if transfer_bytes == 4:
            emitter.emit_instruction(
                BufferLoadDwordLds(voffset, srd_regs, 0, 0, f"gather {transfer_bytes}B")
            )
        elif transfer_bytes == 16:
            emitter.emit_instruction(
                BufferLoadDwordx4Lds(
                    voffset, srd_regs, 0, 0, f"gather {transfer_bytes}B"
                )
            )
        else:
            raise NotImplementedError(
                f"gather_to_lds: transfer_bytes={transfer_bytes} not supported. "
                f"Only 4B (buffer_load_dword) and 16B (buffer_load_dwordx4) are valid."
            )

    # -------------------------------------------------------------------------
    # M0 pre-computation
    # -------------------------------------------------------------------------

    def precompute_g2s_m0(
        self, operation: amdgpu_d.GatherToLDSOp, kernel_info: "KernelInfo"
    ) -> Optional[int]:
        """Pre-compute M0 value before barrier for LLVM-style optimization."""
        lds_base = self.walker._lds_view_base_bytes.get(str(operation.operands[2]), 0)
        lds_cols, elem_bytes = self._get_lds_layout(operation.operands[2].type)

        dst_row = kernel_info.index_env.get(str(operation.operands[3]))
        dst_col = kernel_info.index_env.get(str(operation.operands[4]))

        if dst_row is None or dst_col is None:
            return None  # Can't pre-compute without index values

        # Build M0 expression
        row = dst_row if isinstance(dst_row, sympy.Expr) else sympy.Integer(dst_row)
        col = dst_col if isinstance(dst_col, sympy.Expr) else sympy.Integer(dst_col)
        m0_expr = sympy.simplify(lds_base + (row * lds_cols + col) * elem_bytes)

        emitter = self.walker.emitter
        s_m0 = emitter.sgpr_allocator.alloc_s()

        if m0_expr.is_number:
            emitter.emit_instruction(SMovB32(s_m0, int(m0_expr), "Pre-compute M0"))
        else:
            m0_vreg = int(
                self.handlers._get_expr_emitter(kernel_info).get_or_emit(m0_expr)[1:]
            )
            emitter.emit_instruction(VReadfirstlaneB32(s_m0, m0_vreg, "Pre-compute M0"))

        return s_m0

    # -------------------------------------------------------------------------
    # Main handler
    # -------------------------------------------------------------------------

    def handle_gather_to_lds_op(
        self, operation: amdgpu_d.GatherToLDSOp, kernel_info: "KernelInfo"
    ):
        """Handle amdgpu.gather_to_lds - gather from global memory directly to LDS."""
        emitter = self.walker.emitter

        # Parse transfer type
        transfer_type = operation.attributes["transferType"].value
        if not isinstance(transfer_type, VectorType):
            raise ValueError(f"Unsupported transfer type: {transfer_type}")
        elem_count, elem_bytes, _ = parse_vector_type_from_obj(transfer_type)
        transfer_bytes = elem_count * elem_bytes

        # Get operands
        src_memref_ssa = str(operation.operands[0])
        src_index_ssa = str(operation.operands[1])
        dst_memref_ssa = str(operation.operands[2])

        # Find/create SRD
        orig_srd_regs, original_ssa = self._trace_memref_to_srd(
            src_memref_ssa, kernel_info
        )
        if orig_srd_regs is None:
            emitter.emit(f"    # ERROR: no SRD found for {src_memref_ssa}")
            return

        srd_regs = self._get_or_create_srd(orig_srd_regs, src_memref_ssa)
        emitter.emit(
            f"    # gather_to_lds: {transfer_bytes}B from {original_ssa} to LDS"
        )

        # Compute vaddr
        voffset = self._compute_vaddr(src_index_ssa, kernel_info)

        # Get LDS layout and emit M0
        lds_base_bytes = self.walker._lds_view_base_bytes.get(dst_memref_ssa, 0)
        lds_cols, elem_bytes = self._get_lds_layout(operation.operands[2].type)
        self._emit_m0(operation, kernel_info, lds_base_bytes, lds_cols, elem_bytes)

        # Emit buffer load
        self._emit_buffer_load(transfer_bytes, voffset, srd_regs)

        # Track g2s LDS views
        g2s_views = getattr(self.walker, "_g2s_lds_views", None)
        if g2s_views is None:
            self.walker._g2s_lds_views = set()
        self.walker._g2s_lds_views.add(dst_memref_ssa)
