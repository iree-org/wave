# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import List

from .kernel_pipeline_shared import KRegRange, KPhysSReg, KInstr, KImm


class _LoopSupport:
    # =========================================================================
    # Loop Support
    # =========================================================================

    def begin_loop(self, lower_bound: int, upper_bound: int, step: int) -> dict:
        """
        Begin a loop structure with physical registers for loop control.

        Loop control registers (counter, step, upper_bound) use physical SGPRs
        to avoid SSA violations from the loop counter update in the latch.

        Returns a loop context dict for use with emit_loop_header/latch/end.
        """
        loop_id = self._loop_counter
        self._loop_counter += 1

        # Use physical SGPRs for loop control to avoid SSA violations
        # Reserve SGPRs starting at 24 to avoid conflicts with SRDs (which use s4+)
        # Legacy mode uses s24 for loop counter, which is known to be safe.
        # Each loop needs 3 SGPRs: counter, step, upper_bound
        base_sgpr = 24 + loop_id * 3
        counter_phys = KPhysSReg(base_sgpr)
        step_phys = KPhysSReg(base_sgpr + 1)
        upper_bound_phys = KPhysSReg(base_sgpr + 2)

        # Initialize loop counter and bounds using physical registers
        self.comment(f"Initialize loop {loop_id}")
        self.program.emit(
            KInstr(
                "s_mov_b32",
                (counter_phys,),
                (KImm(lower_bound),),
                comment=f"loop {loop_id} counter = {lower_bound}",
            )
        )
        self.program.emit(
            KInstr(
                "s_mov_b32",
                (step_phys,),
                (KImm(step),),
                comment=f"loop {loop_id} step = {step}",
            )
        )
        self.program.emit(
            KInstr(
                "s_mov_b32",
                (upper_bound_phys,),
                (KImm(upper_bound),),
                comment=f"loop {loop_id} upper = {upper_bound}",
            )
        )

        loop_ctx = {
            "loop_id": loop_id,
            "counter_sreg": counter_phys,
            "step_sreg": step_phys,
            "upper_bound_sreg": upper_bound_phys,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "step": step,
        }

        self._loop_stack.append(loop_ctx)
        return loop_ctx

    def emit_loop_header(self, loop_ctx: dict):
        """Emit loop header with comparison and conditional branch."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        upper = loop_ctx["upper_bound_sreg"]

        # Emit label
        self.emit_label(f"loop_{loop_id}_header")

        # Compare counter < upper (sets SCC)
        self.program.emit(
            KInstr(
                "s_cmp_lt_u32",
                (),
                (counter, upper),
                comment=f"compare loop {loop_id} counter < upper",
            )
        )

        # Branch to body if SCC=1
        self.program.emit(
            KInstr("s_cbranch_scc1", (), (), comment=f"loop_{loop_id}_body")
        )

        # Branch to exit if not taken
        self.program.emit(KInstr("s_branch", (), (), comment=f"loop_{loop_id}_exit"))

        # Body label
        self.emit_label(f"loop_{loop_id}_body")

    def emit_loop_latch(self, loop_ctx: dict):
        """Emit loop latch (increment counter and branch back)."""
        loop_id = loop_ctx["loop_id"]
        counter = loop_ctx["counter_sreg"]
        step = loop_ctx["step_sreg"]

        # Latch label
        self.emit_label(f"loop_{loop_id}_latch")

        # Increment counter - uses physical regs so no defs to track
        # s_add_u32 with physical regs: counter += step
        self.program.emit(
            KInstr(
                "_loop_inc",
                (),
                (counter, step),
                comment=f"loop {loop_id} counter += step",
            )
        )

        # Branch back to header
        self.program.emit(KInstr("s_branch", (), (), comment=f"loop_{loop_id}_header"))

    def end_loop(self):
        """End loop and emit exit label."""
        loop_ctx = self._loop_stack.pop()
        loop_id = loop_ctx["loop_id"]
        self.emit_label(f"loop_{loop_id}_exit")

    def alloc_accumulators(self, count: int) -> List[KRegRange]:
        """
        Allocate accumulator VGPR quads for loop iter_args.

        Returns list of KRegRange quads, each initialized to 0.
        """
        accumulators = []
        for i in range(count):
            quad = self.vreg_quad()
            accumulators.append(quad)

            # Initialize to 0 using a pseudo-instruction that defines the whole range
            # This ensures liveness analysis sees the quad with proper alignment
            self.program.emit(
                KInstr(
                    "_init_acc_quad",
                    (quad,),
                    (KImm(0),),
                    comment=f"Initialize accumulator {i} to 0.0",
                )
            )

        return accumulators
