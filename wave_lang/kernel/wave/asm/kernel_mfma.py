# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Optional, Tuple

from .kernel_pipeline_shared import KReg, KVReg, KRegRange, KInstr, KImm


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
        # Wait for LDS reads to complete before MFMA
        self.wait_lgkm(0)

        # Build operand ranges
        a_range = KRegRange(a_regs[0], 2, alignment=2) if len(a_regs) >= 2 else None
        b_range = KRegRange(b_regs[0], 2, alignment=2) if len(b_regs) >= 2 else None

        # Determine result/accumulator
        if acc_regs is not None and len(acc_regs) == 4:
            # Use provided accumulator as both input and output
            result_regs = acc_regs
            acc_range = KRegRange(acc_regs[0], 4, alignment=4)

            # MFMA with accumulator: v_mfma dst, a, b, acc
            # Note: When reusing accumulator, we emit a pseudo-op that doesn't
            # define new registers (result == accumulator)
            self.program.emit(
                KInstr(
                    "_mfma_acc",  # Pseudo: uses accumulator, doesn't define new regs
                    (),  # No new defs - reusing accumulator
                    (acc_range, a_range, b_range),
                    comment="MFMA with accumulator (in-place)",
                )
            )
        else:
            # Allocate new quad for result, use 0 as accumulator
            result_range = self.vreg_quad()
            result_regs = tuple(KVReg(result_range.base_reg.id + i) for i in range(4))

            # MFMA with zero accumulator: v_mfma dst, a, b, 0
            self.program.emit(
                KInstr(
                    "v_mfma_f32_16x16x16_f16",
                    (result_range,),
                    (a_range, b_range, KImm(0)),
                    comment="MFMA with zero accumulator",
                )
            )

        return result_regs
