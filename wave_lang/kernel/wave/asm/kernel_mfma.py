# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Optional, Tuple

from .kernel_pipeline_shared import KReg, KVReg, KRegRange, KInstr, KImm
from .instruction_registry import Instruction


class _MFMASupport:
    # =========================================================================
    # MFMA Support
    # =========================================================================

    def emit_mfma_f32_16x16x16_f16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA instruction with virtual register tracking.

        Args:
            a_regs: Tuple of 2 VGPRs for A operand (f16x2)
            b_regs: Tuple of 2 VGPRs for B operand (f16x2)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers

        Returns:
            Tuple of 4 VGPRs containing the result
        """
        # NOTE: Wait for LDS reads is now handled by the ticketing pass
        # (_apply_ticketing_waitcnt_placement) based on register dependencies.
        # This avoids redundant consecutive lgkmcnt(0) instructions.

        # Build operand ranges
        a_range = KRegRange(a_regs[0], 2, alignment=2) if len(a_regs) >= 2 else None
        b_range = KRegRange(b_regs[0], 2, alignment=2) if len(b_regs) >= 2 else None

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            # Ensure accumulator regs are marked as such for SSA validation.
            if isinstance(acc_range.base_reg, KVReg):
                self.program.register_accumulator_vreg_range(acc_range)

            # MFMA with accumulator: v_mfma dst, a, b, acc
            # Note: MFMA updates the accumulator in-place (read-modify-write).
            # We model this by defining the accumulator range as a def.
            self.program.emit(
                KInstr(
                    "_mfma_acc",  # Pseudo: uses accumulator, doesn't define new regs
                    (acc_range,),  # def: accumulator range (RMW)
                    (acc_range, a_range, b_range),
                    comment="MFMA with accumulator (in-place)",
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(
                KInstr(
                    Instruction.V_MFMA_F32_16X16X16_F16,
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA with zero accumulator",
                )
            )

        return result_regs

    def emit_mfma_f32_16x16x32_f16(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        acc_regs: Optional[Tuple[KReg, ...]] = None,
    ) -> Tuple[KReg, ...]:
        """
        Emit MFMA 16x16x32 instruction with virtual register tracking.

        For 16x16x32: A needs 8 x f16 (4 VGPRs), B needs 8 x f16 (4 VGPRs),
        result is 4 x f32 (4 VGPRs).

        Args:
            a_regs: Tuple of 4 VGPRs for A operand (f16x8)
            b_regs: Tuple of 4 VGPRs for B operand (f16x8)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers

        Returns:
            Tuple of 4 VGPRs containing the result
        """
        # Build operand ranges - 16x16x32 needs 4 VGPRs for A and B
        a_range = KRegRange(a_regs[0], 4, alignment=4) if len(a_regs) >= 4 else None
        b_range = KRegRange(b_regs[0], 4, alignment=4) if len(b_regs) >= 4 else None

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            if isinstance(acc_range.base_reg, KVReg):
                self.program.register_accumulator_vreg_range(acc_range)

            # MFMA with accumulator: v_mfma dst, a, b, acc
            self.program.emit(
                KInstr(
                    "_mfma_acc_16x16x32",  # Pseudo: in-place accumulator update
                    (acc_range,),  # def: accumulator range (RMW)
                    (acc_range, a_range, b_range),
                    comment="MFMA 16x16x32 with accumulator (in-place)",
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(
                KInstr(
                    Instruction.V_MFMA_F32_16X16X32_F16,
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA 16x16x32 with zero accumulator",
                )
            )

        return result_regs

    # Scaled MFMA format codes (cbsz/blgp values):
    #   0 = FP8 (E4M3FN), 1 = BF8 (E5M2), 2 = FP6 (E2M3FN),
    #   3 = FP6 (E3M2FN), 4 = FP4 (E2M1FN)
    SCALED_MFMA_FORMAT_CODES = {
        "f4e2m1fn": 4,
        "f4E2M1FN": 4,
        "Float4E2M1FN": 4,
        "f6e2m3fn": 2,
        "f6E2M3FN": 2,
        "Float6E2M3FN": 2,
        "f6e3m2fn": 3,
        "f6E3M2FN": 3,
        "Float6E3M2FN": 3,
        "f8e4m3fn": 0,
        "f8E4M3FN": 0,
        "Float8E4M3FN": 0,
        "f8e5m2": 1,
        "f8E5M2": 1,
        "Float8E5M2": 1,
    }

    @staticmethod
    def _get_scaled_mfma_format_code(type_str: str) -> int:
        """Get the scaled MFMA format code (cbsz/blgp) from a type string.

        Matches the C++ backend's getScaledMFMAFormatCode() logic.
        Returns 4 (FP4) as default for unrecognized types.
        """
        for key, code in _MFMASupport.SCALED_MFMA_FORMAT_CODES.items():
            if key in type_str:
                return code
        return 4  # Default to FP4

    def emit_mfma_f32_16x16x128_f8f6f4(
        self,
        a_regs: Tuple[KReg, ...],
        b_regs: Tuple[KReg, ...],
        a_scale_reg: KReg,
        b_scale_reg: KReg,
        acc_regs: Optional[Tuple[KReg, ...]] = None,
        cbsz: int = 4,
        blgp: int = 4,
        scales_idx_a: int = 0,
        scales_idx_b: int = 0,
    ) -> Tuple[KReg, ...]:
        """
        Emit scaled MFMA instruction for MXFP4 (16x16x128 F8F6F4).

        This instruction supports FP4, FP6, or FP8 independently for A and B matrices
        with per-group E8M0 scaling factors.

        For FP4: Each operand is 32 x 4-bit elements = 16 bytes = 4 VGPRs

        Args:
            a_regs: Tuple of 4 VGPRs for A operand (32 x f4E2M1FN packed as i8)
            b_regs: Tuple of 4 VGPRs for B operand (32 x f4E2M1FN packed as i8)
            a_scale_reg: Single VGPR for A scale factor (f8E8M0FNU or
                         4 packed bytes with opsel byte selection)
            b_scale_reg: Single VGPR for B scale factor (same as above)
            acc_regs: Optional tuple of 4 VGPRs for accumulator (f32x4)
                      If None, allocates new result registers
            cbsz: Format code for A source data (0=FP8, 1=BF8, 2=FP6_E2M3,
                   3=FP6_E3M2, 4=FP4). Default 4 (FP4).
            blgp: Format code for B source data. Same encoding as cbsz.
                   Default 4 (FP4).
            scales_idx_a: Byte index (0-3) within the A scale VGPR. Default 0.
            scales_idx_b: Byte index (0-3) within the B scale VGPR. Default 0.

        Returns:
            Tuple of 4 VGPRs containing the result
        """
        modifiers = f"cbsz:{cbsz} blgp:{blgp}"
        if scales_idx_a != 0 or scales_idx_b != 0:
            modifiers += f" op_sel_hi:[0,0,0,{scales_idx_a},{scales_idx_b}]"

        # Build operand ranges - For FP4: 32 elements = 16 bytes = 4 VGPRs
        a_range = KRegRange(a_regs[0], len(a_regs), alignment=4)
        b_range = KRegRange(b_regs[0], len(b_regs), alignment=4)

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)
            if isinstance(acc_range.base_reg, KVReg):
                self.program.register_accumulator_vreg_range(acc_range)

            # Scaled MFMA with accumulator: v_mfma_scale dst, a, b, acc, a_scale, b_scale
            # Hardware operand order: src_a, src_b, src_c(acc), scale_a, scale_b
            self.program.emit(
                KInstr(
                    "v_mfma_scale_f32_16x16x128_f8f6f4",
                    (acc_range,),  # def: accumulator range (RMW)
                    (a_range, b_range, acc_range, a_scale_reg, b_scale_reg),
                    comment="Scaled MFMA 16x16x128 F8F6F4 with accumulator (in-place)",
                    modifiers=modifiers,
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            self.program.register_accumulator_vreg_range(result_range)
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # Scaled MFMA with zero accumulator
            self.program.emit(
                KInstr(
                    "v_mfma_scale_f32_16x16x128_f8f6f4",
                    (result_range,),
                    (a_range, b_range, KImm(0), a_scale_reg, b_scale_reg),
                    comment="Scaled MFMA 16x16x128 F8F6F4 with zero accumulator",
                    modifiers=modifiers,
                )
            )

        return result_regs
