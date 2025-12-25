# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Tests for CFG-based liveness analysis.

These tests verify that the backward dataflow analysis correctly extends
live ranges for loop-invariant values.
"""

import pytest
from wave_lang.kernel.wave.asm.kernel_ir import (
    KernelProgram,
    KernelABI,
    KInstr,
    KVReg,
    KSReg,
    KImm,
    KRegRange,
)
from wave_lang.kernel.wave.asm.kernel_liveness import (
    build_cfg,
    compute_cfg_liveness,
    compute_liveness,
    has_loops,
    compute_block_local_info,
    LiveRange,
)


class TestCFGConstruction:
    """Tests for CFG construction from kernel programs."""

    def test_simple_linear_cfg(self):
        """Test CFG for linear code (no branches)."""
        program = KernelProgram()
        v0 = program.alloc_vreg()
        v1 = program.alloc_vreg()

        # Simple sequence: v0 = 1; v1 = v0 + 1
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))

        cfg = build_cfg(program)

        assert len(cfg.blocks) == 1
        assert cfg.entry_block is not None
        assert cfg.entry_block.start_idx == 0
        assert cfg.entry_block.end_idx == 1
        assert len(cfg.entry_block.successors) == 0

    def test_cfg_with_label(self):
        """Test CFG construction with labels."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        # label: v0 = 1
        program.emit(KInstr("_label", (), (), comment="my_label"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))

        cfg = build_cfg(program)

        assert len(cfg.blocks) == 1
        assert cfg.blocks[0].label == "my_label"
        assert "my_label" in cfg.label_to_block

    def test_cfg_with_unconditional_branch(self):
        """Test CFG with unconditional branch."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        # v0 = 1; branch target; v0 = 2; target: v0 = 3
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("s_branch", (), (), comment="target"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(2),)))
        program.emit(KInstr("_label", (), (), comment="target"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(3),)))

        cfg = build_cfg(program)

        assert len(cfg.blocks) >= 2
        # First block should branch to "target" block
        target_block = cfg.label_to_block.get("target")
        assert target_block is not None

    def test_cfg_with_conditional_branch(self):
        """Test CFG with conditional branch (creates two successors)."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        # cbranch target; v0 = 1; target: v0 = 2
        program.emit(KInstr("s_cbranch_scc1", (), (), comment="target"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("_label", (), (), comment="target"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(2),)))

        cfg = build_cfg(program)

        # First block should have 2 successors (branch taken + fallthrough)
        assert len(cfg.entry_block.successors) == 2

    def test_loop_detection(self):
        """Test that has_loops detects back-edges."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        # loop_header: v0 = 1; branch loop_header
        program.emit(KInstr("_label", (), (), comment="loop_header"))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("s_branch", (), (), comment="loop_header"))

        cfg = build_cfg(program)

        assert has_loops(cfg) == True

    def test_no_loop(self):
        """Test that has_loops returns False for linear code."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(2),)))

        cfg = build_cfg(program)

        assert has_loops(cfg) == False


class TestBlockLocalInfo:
    """Tests for computing local use/def sets."""

    def test_simple_def(self):
        """Test that defs are tracked."""
        program = KernelProgram()
        v0 = program.alloc_vreg()

        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))

        cfg = build_cfg(program)
        compute_block_local_info(cfg.blocks[0], program.instructions)

        assert v0 in cfg.blocks[0].def_set
        assert v0 not in cfg.blocks[0].use_set

    def test_use_before_def(self):
        """Test that uses before defs are in use_set."""
        program = KernelProgram()
        v0 = program.alloc_vreg()
        v1 = program.alloc_vreg()

        # Manually set up: use v0 (not defined locally), then define v1
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))

        cfg = build_cfg(program)
        compute_block_local_info(cfg.blocks[0], program.instructions)

        assert v0 in cfg.blocks[0].use_set
        assert v1 in cfg.blocks[0].def_set
        assert v1 not in cfg.blocks[0].use_set  # Defined before used

    def test_def_kills_use(self):
        """Test that a def before use prevents use_set inclusion."""
        program = KernelProgram()
        v0 = program.alloc_vreg()
        v1 = program.alloc_vreg()

        # v0 = 1; v1 = v0 + 1
        # v0 should not be in use_set because it's defined first
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))

        cfg = build_cfg(program)
        compute_block_local_info(cfg.blocks[0], program.instructions)

        assert v0 in cfg.blocks[0].def_set
        assert v0 not in cfg.blocks[0].use_set  # Defined before used


class TestCFGLiveness:
    """Tests for CFG-based backward dataflow liveness."""

    def test_live_in_propagation(self):
        """Test that live-in propagates backward."""
        program = KernelProgram()
        v0 = program.alloc_vreg()
        v1 = program.alloc_vreg()

        # Block 0: v0 = 1; branch block1
        # Block 1: v1 = v0 + 1
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("s_branch", (), (), comment="block1"))
        program.emit(KInstr("_label", (), (), comment="block1"))
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))

        cfg = build_cfg(program)
        compute_cfg_liveness(cfg, program.instructions)

        # v0 should be live-in to block1
        block1 = cfg.label_to_block["block1"]
        assert v0 in block1.live_in

    def test_loop_liveness(self):
        """Test that loop-carried values stay live across iterations."""
        program = KernelProgram()
        v0 = program.alloc_vreg()  # Loop-invariant value defined before loop
        v1 = program.alloc_vreg()  # Loop counter
        v2 = program.alloc_vreg()  # Accumulator

        # Prologue: v0 = 42 (loop-invariant)
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(42),)))

        # Loop header
        program.emit(KInstr("_label", (), (), comment="loop_header"))

        # Loop body: use v0 (loop-invariant)
        program.emit(KInstr("v_add_u32", (v2,), (v0, v1)))

        # Loop latch: branch back
        program.emit(KInstr("s_branch", (), (), comment="loop_header"))

        cfg = build_cfg(program)
        compute_cfg_liveness(cfg, program.instructions)

        # v0 should be live-in to loop_header (used in body, defined before)
        loop_header = cfg.label_to_block["loop_header"]
        assert v0 in loop_header.live_in

        # v0 should be live-out of the block containing the branch back
        # (because it flows to the next iteration)
        for block in cfg.blocks:
            if block.end_idx >= 3:  # Block containing the back-edge
                assert v0 in block.live_out or v0 in block.live_in


class TestLiveRangeExtension:
    """Tests for live range extension via CFG analysis."""

    def test_linear_code_no_extension(self):
        """Test that linear code doesn't over-extend live ranges."""
        program = KernelProgram()
        v0 = program.alloc_vreg()
        v1 = program.alloc_vreg()

        # v0 = 1; v1 = v0 + 1; (v0 dead after this)
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(1),)))
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))
        program.emit(KInstr("v_mov_b32", (v1,), (KImm(2),)))

        info = compute_liveness(program, use_cfg=True)

        # v0 should end at instruction 1 (its last use)
        assert info.live_ranges[v0].end == 1

    def test_loop_invariant_extension(self):
        """Test that loop-invariant values get extended live ranges."""
        program = KernelProgram()
        v0 = program.alloc_vreg()  # Loop-invariant
        v1 = program.alloc_vreg()  # Used in loop

        # v0 = 42 (defined before loop)
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(42),)))  # idx 0

        # loop_header:
        program.emit(KInstr("_label", (), (), comment="loop_header"))  # idx 1

        # v1 = v0 + 1 (use v0 in loop body)
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))  # idx 2

        # branch loop_header (back-edge)
        program.emit(KInstr("s_branch", (), (), comment="loop_header"))  # idx 3

        info = compute_liveness(program, use_cfg=True)

        # v0 should be live through the entire loop
        # Its live range should extend at least to the back-edge
        v0_range = info.live_ranges[v0]
        assert v0_range.start == 0  # Defined at instruction 0
        assert v0_range.end >= 3  # Should extend through back-edge

    def test_cfg_vs_linear_difference(self):
        """Test that CFG analysis extends ranges that linear doesn't."""
        program = KernelProgram()
        v0 = program.alloc_vreg()  # Loop-invariant
        v1 = program.alloc_vreg()

        # v0 = 42
        program.emit(KInstr("v_mov_b32", (v0,), (KImm(42),)))  # idx 0

        # loop:
        program.emit(KInstr("_label", (), (), comment="loop"))  # idx 1

        # v1 = v0 + 1
        program.emit(KInstr("v_add_u32", (v1,), (v0, KImm(1))))  # idx 2

        # branch loop
        program.emit(KInstr("s_branch", (), (), comment="loop"))  # idx 3

        # Compare CFG-based vs linear liveness
        info_cfg = compute_liveness(program, use_cfg=True)
        info_linear = compute_liveness(program, use_cfg=False)

        # With linear analysis, v0 ends at idx 2 (last textual use)
        # With CFG analysis, v0 should extend through the loop
        assert info_linear.live_ranges[v0].end == 2
        assert info_cfg.live_ranges[v0].end >= 3


class TestIntegration:
    """Integration tests with realistic loop patterns."""

    def test_gemm_style_loop(self):
        """Test liveness for a GEMM-style loop pattern."""
        program = KernelProgram()

        # Pre-loop: compute thread ID (loop-invariant)
        tid = program.alloc_vreg()
        program.emit(KInstr("v_mov_b32", (tid,), (KImm(0),), comment="tid"))  # idx 0

        # Loop counter
        counter = program.alloc_sreg()
        bound = program.alloc_sreg()
        program.emit(KInstr("s_mov_b32", (counter,), (KImm(0),)))  # idx 1
        program.emit(KInstr("s_mov_b32", (bound,), (KImm(4),)))  # idx 2

        # loop_header:
        program.emit(KInstr("_label", (), (), comment="loop_header"))  # idx 3

        # Compare and conditional branch
        program.emit(KInstr("s_cmp_lt_u32", (), (counter, bound)))  # idx 4
        program.emit(KInstr("s_cbranch_scc0", (), (), comment="loop_exit"))  # idx 5

        # Loop body: use tid (loop-invariant)
        acc = program.alloc_vreg()
        program.emit(KInstr("v_add_u32", (acc,), (tid, KImm(1))))  # idx 6

        # Increment counter
        program.emit(KInstr("s_add_u32", (counter,), (counter, KImm(1))))  # idx 7

        # Branch back
        program.emit(KInstr("s_branch", (), (), comment="loop_header"))  # idx 8

        # loop_exit:
        program.emit(KInstr("_label", (), (), comment="loop_exit"))  # idx 9

        # Epilogue: use result
        result = program.alloc_vreg()
        program.emit(KInstr("v_mov_b32", (result,), (acc,)))  # idx 10

        info = compute_liveness(program, use_cfg=True)

        # tid should be live from definition (0) through loop body (at least 6)
        tid_range = info.live_ranges[tid]
        assert tid_range.start == 0
        assert tid_range.end >= 6

        # counter should be live through the loop
        counter_range = info.live_ranges[counter]
        assert counter_range.end >= 7  # Last use is at s_add


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
