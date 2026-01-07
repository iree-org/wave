# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unit tests for kernel-level IR and linear scan register allocation.

These tests verify:
1. KernelProgram construction and virtual register allocation
2. Liveness analysis for SSA programs
3. Linear scan allocation with constraints
4. Register range allocation (pairs, quads)
5. Precoloring and reserved registers
6. Rendering to assembly
"""

import pytest
from wave_lang.kernel.wave.asm.kernel_ir import (
    KernelProgram,
    KInstr,
    KVReg,
    KSReg,
    KPhysVReg,
    KPhysSReg,
    KRegRange,
    KImm,
    KMemOffset,
    KernelABI,
    RegClass,
    is_virtual,
    is_vgpr,
    is_sgpr,
    get_reg_class,
)
from wave_lang.kernel.wave.asm.instruction_registry import Instruction
from wave_lang.kernel.wave.asm.kernel_liveness import (
    compute_liveness,
    LiveRange,
    LivenessInfo,
    validate_ssa,
    get_live_in,
    get_live_out,
)
from wave_lang.kernel.wave.asm.kernel_regalloc import (
    KernelRegAlloc,
    allocate_kernel,
    AllocationError,
    RegPool,
)
from wave_lang.kernel.wave.asm.kernel_generator import (
    KernelRenderer,
    PhysicalMapping,
    render_program,
)
from wave_lang.kernel.wave.asm.kernel_compilation_context import (
    KernelCompilationContext,
)


class TestKernelIR:
    """Tests for kernel IR construction."""

    def test_vreg_allocation(self):
        """Test virtual VGPR allocation."""
        prog = KernelProgram()
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()

        assert v0.id == 0
        assert v1.id == 1
        assert v2.id == 2
        assert is_virtual(v0)
        assert is_vgpr(v0)

    def test_sreg_allocation(self):
        """Test virtual SGPR allocation."""
        prog = KernelProgram()
        s0 = prog.alloc_sreg()
        s1 = prog.alloc_sreg()

        assert s0.id == 0
        assert s1.id == 1
        assert is_virtual(s0)
        assert is_sgpr(s0)

    def test_vreg_range_allocation(self):
        """Test virtual VGPR range allocation."""
        prog = KernelProgram()
        pair = prog.alloc_vreg_range(2, alignment=2)

        assert pair.count == 2
        assert pair.alignment == 2
        assert isinstance(pair.base_reg, KVReg)
        # Verify it's a VGPR by checking the base register type
        assert is_vgpr(pair.base_reg)

    def test_instruction_emission(self):
        """Test instruction emission."""
        prog = KernelProgram()

        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v1,), (KImm(42),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v2,), (v1, KImm(1))))

        assert len(prog) == 2
        assert prog.instructions[0].name == Instruction.V_MOV_B32
        assert prog.instructions[1].name == Instruction.V_ADD_U32

    def test_multiple_instructions(self):
        """Test emitting multiple instructions with KernelProgram."""
        prog = KernelProgram()

        # Test register allocation and instruction emission
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        v3 = prog.alloc_vreg()
        v4 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v1,), (KImm(10),)))
        prog.emit(KInstr(Instruction.V_MUL_LO_U32, (v2,), (v1, KImm(4))))
        prog.emit(KInstr(Instruction.V_LSHLREV_B32, (v3,), (KImm(2), v1)))
        prog.emit(KInstr(Instruction.V_AND_B32, (v4,), (v2, KImm(0xFF))))

        assert len(prog) == 4
        assert all(isinstance(v, KVReg) for v in [v1, v2, v3, v4])

    def test_physical_registers(self):
        """Test physical register types."""
        pv = KPhysVReg(0)
        ps = KPhysSReg(1)

        assert not is_virtual(pv)
        assert not is_virtual(ps)
        assert is_vgpr(pv)
        assert is_sgpr(ps)
        assert repr(pv) == "v0"
        assert repr(ps) == "s1"

    def test_kernel_abi(self):
        """Test kernel ABI configuration."""
        abi = KernelABI()

        # Default: v0 for flat tid, s[0:1] for kernarg
        assert abi.flat_tid_vreg.index == 0
        assert abi.kernarg_ptr_sreg_lo.index == 0
        assert abi.kernarg_ptr_sreg_hi.index == 1

        reserved_vgprs = abi.get_reserved_vgprs()
        reserved_sgprs = abi.get_reserved_sgprs()

        assert 0 in reserved_vgprs
        assert 0 in reserved_sgprs
        assert 1 in reserved_sgprs


class TestLiveness:
    """Tests for liveness analysis."""

    def test_simple_liveness(self):
        """Test liveness for a simple program."""
        prog = KernelProgram()

        # v0 = 42 (instr 0, used at 1)
        # v1 = v0 + 1 (instr 1)
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(42),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v1,), (v0, KImm(1))))

        info = compute_liveness(prog)

        assert v0 in info.live_ranges
        assert v1 in info.live_ranges

        # v0: defined at 0, used at 1
        assert info.live_ranges[v0].start == 0
        assert info.live_ranges[v0].end == 1

        # v1: defined at 1, never used
        assert info.live_ranges[v1].start == 1
        assert info.live_ranges[v1].end == 1

    def test_longer_liveness(self):
        """Test liveness with longer live ranges."""
        prog = KernelProgram()

        # v0 = 10 (instr 0)
        # v1 = v0 * 2 (instr 1)
        # v2 = v0 + v1 (instr 2) <- v0 used again
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(10),)))
        prog.emit(KInstr(Instruction.V_MUL_LO_U32, (v1,), (v0, KImm(2))))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v2,), (v0, v1)))

        info = compute_liveness(prog)

        # v0: defined at 0, last used at 2
        assert info.live_ranges[v0].start == 0
        assert info.live_ranges[v0].end == 2

        # v1: defined at 1, used at 2
        assert info.live_ranges[v1].start == 1
        assert info.live_ranges[v1].end == 2

    def test_ssa_validation(self):
        """Test SSA validation catches violations."""
        prog = KernelProgram()

        # Manually create an SSA violation (same reg defined twice)
        v0 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(1),)))
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(2),)))  # Violation!

        errors = validate_ssa(prog)
        assert len(errors) > 0
        assert "defined" in errors[0].lower() or "ssa" in errors[0].lower()

    def test_register_pressure(self):
        """Test register pressure computation."""
        prog = KernelProgram()

        # Create overlapping live ranges to test pressure
        # v0 = 1 (def 0)
        # v1 = 2 (def 1)
        # v2 = v0 + v1 (def 2, uses v0, v1) <- pressure 3 at this point
        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(1),)))
        prog.emit(KInstr(Instruction.V_MOV_B32, (v1,), (KImm(2),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v2,), (v0, v1)))

        info = compute_liveness(prog)

        # At instruction 2, v0, v1, v2 are all potentially live
        # (v0, v1 as uses, v2 as def)
        assert info.max_vreg_pressure >= 2


class TestRegAlloc:
    """Tests for register allocation."""

    def test_simple_allocation(self):
        """Test simple allocation without constraints."""
        prog = KernelProgram()

        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(42),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v1,), (v0, KImm(1))))

        mapping, stats = allocate_kernel(prog)

        # Both virtual registers should be allocated
        assert v0.id in mapping.vreg_map
        assert v1.id in mapping.vreg_map

        # Should use different physical registers (both live at some point)
        # Actually v0 is dead after v1, so could reuse... let's just check allocated
        assert mapping.vreg_map[v0.id] >= 0
        assert mapping.vreg_map[v1.id] >= 0

    def test_reserved_registers(self):
        """Test that reserved registers are not allocated."""
        prog = KernelProgram()

        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        v2 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(42),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v1,), (v0, KImm(1))))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v2,), (v1, KImm(2))))

        # Reserve v0 (ABI flat tid)
        mapping, stats = allocate_kernel(prog, reserved_vgprs={0})

        # No virtual register should be allocated to v0
        for vreg_id, phys in mapping.vreg_map.items():
            assert phys != 0, f"Virtual reg {vreg_id} allocated to reserved v0"

    def test_range_allocation(self):
        """Test contiguous range allocation."""
        prog = KernelProgram()

        # Allocate a pair for ds_read_b64
        pair = prog.alloc_vreg_range(2, alignment=2)
        addr = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (addr,), (KImm(0),)))

        # Emit ds_read_b64 that defines the pair
        prog.emit(KInstr(Instruction.DS_READ_B64, (pair,), (addr,)))

        mapping, stats = allocate_kernel(prog)

        # The pair base should be allocated
        base_phys = mapping.vreg_map[pair.base_reg.id]

        # Should be even (2-aligned)
        assert base_phys % 2 == 0, f"Pair base {base_phys} not 2-aligned"

    def test_reg_pool_single(self):
        """Test RegPool single allocation."""
        pool = RegPool(RegClass.VGPR, max_regs=10, reserved={0})

        r1 = pool.alloc_single()
        r2 = pool.alloc_single()

        # Should skip reserved v0
        assert r1 != 0
        assert r2 != 0
        assert r1 != r2

    def test_reg_pool_range(self):
        """Test RegPool range allocation with alignment."""
        pool = RegPool(RegClass.VGPR, max_regs=20, reserved=set())

        # Allocate a pair (2-aligned)
        base = pool.alloc_range(2, alignment=2)
        assert base % 2 == 0

        # Allocate a quad (4-aligned)
        base2 = pool.alloc_range(4, alignment=4)
        assert base2 % 4 == 0

    def test_reg_pool_free_and_reuse(self):
        """Test that freed registers can be reused."""
        pool = RegPool(RegClass.VGPR, max_regs=10, reserved=set())

        r1 = pool.alloc_single()
        r2 = pool.alloc_single()

        # Free r1
        pool.free_single(r1)

        # Next allocation should reuse r1 (lowest available)
        r3 = pool.alloc_single()
        assert r3 == r1

    def test_peak_vgpr_accounting_with_ranges(self):
        """
        Regression test: peak VGPR/SGPR accounting must include register ranges.

        Previously, vreg_map.get(KVReg(base)) was wrong because the map keys are
        integers (virtual reg ids), not KVReg objects. This caused under-reporting
        of peak_vgprs/peak_sgprs when register ranges were allocated.
        """
        prog = KernelProgram()

        # Allocate a single VGPR
        v0 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(1),)))

        # Allocate a quad (4 VGPRs), which should push peak to at least 5
        quad = prog.alloc_vreg_range(4, alignment=4)
        addr = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (addr,), (KImm(0),)))

        # Emit buffer_load_dwordx4 that defines the quad
        prog.emit(
            KInstr(
                Instruction.BUFFER_LOAD_DWORDX4, (quad,), (addr, KPhysSReg(4), KImm(0))
            )
        )

        mapping, stats = allocate_kernel(prog, reserved_vgprs={0})

        # The quad takes 4 registers, plus v0 takes 1, plus addr takes 1
        # peak_vgprs should be at least 6 (accounting for the quad range)
        assert stats.peak_vgprs >= 5, (
            f"peak_vgprs={stats.peak_vgprs} should be >= 5 when allocating "
            f"a quad range. This regression indicates vreg_map key type bug."
        )

        # Verify the quad base is allocated to a physical register
        quad_base_phys = mapping.vreg_map.get(quad.base_reg.id)
        assert quad_base_phys is not None, "Quad base not allocated"

        # Peak should account for quad_base + 3 (the full range)
        expected_min_peak = quad_base_phys + 4  # base + 4 registers in quad
        assert stats.peak_vgprs >= expected_min_peak, (
            f"peak_vgprs={stats.peak_vgprs} should be >= {expected_min_peak} "
            f"(quad at v{quad_base_phys}:v{quad_base_phys+3})"
        )

    def test_allocation_failure(self):
        """Test that allocation fails gracefully with no spilling."""
        prog = KernelProgram(max_vgprs=4)

        # Create many overlapping live ranges to exhaust registers
        regs = []
        for i in range(6):
            v = prog.alloc_vreg()
            prog.emit(KInstr(Instruction.V_MOV_B32, (v,), (KImm(i),)))
            regs.append(v)

        # Use all of them at once to prevent reuse
        result = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_ADD_U32, (result,), (regs[0], regs[1])))
        for r in regs[2:]:
            new_result = prog.alloc_vreg()
            prog.emit(KInstr(Instruction.V_ADD_U32, (new_result,), (result, r)))
            result = new_result

        # With only 4 VGPRs, this should fail
        with pytest.raises(AllocationError):
            allocate_kernel(prog, reserved_vgprs={0})


class TestRenderer:
    """Tests for assembly rendering."""

    def test_simple_render(self):
        """Test rendering a simple program."""
        prog = KernelProgram()

        v0 = prog.alloc_vreg()
        v1 = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (v0,), (KImm(42),)))
        prog.emit(KInstr(Instruction.V_ADD_U32, (v1,), (v0, KImm(1))))

        mapping, _ = allocate_kernel(prog, reserved_vgprs={0})
        asm = render_program(prog, mapping)

        assert "v_mov_b32" in asm
        assert "v_add_u32" in asm
        assert "42" in asm

    def test_render_with_ranges(self):
        """Test rendering instructions with register ranges."""
        prog = KernelProgram()

        addr = prog.alloc_vreg()
        prog.emit(KInstr(Instruction.V_MOV_B32, (addr,), (KImm(0),)))
        pair = prog.alloc_vreg_range(2, alignment=2)
        prog.emit(KInstr(Instruction.DS_READ_B64, (pair,), (addr, KMemOffset(8))))

        mapping, _ = allocate_kernel(prog, reserved_vgprs={0})
        asm = render_program(prog, mapping)

        assert "ds_read_b64" in asm
        assert "offset:8" in asm


class TestKernelCompilationContext:
    """Tests for the compilation context."""

    def test_context_creation(self):
        """Test context creation and basic emission."""
        ctx = KernelCompilationContext()

        v0 = ctx.v_mov_b32(KImm(10))
        v1 = ctx.v_add_u32(v0, KImm(5))

        assert ctx.num_instructions == 2
        assert ctx.num_virtual_vregs == 2

    def test_context_finalization(self):
        """Test context finalization generates valid assembly."""
        ctx = KernelCompilationContext(use_flat_tid=False)

        v0 = ctx.v_mov_b32(KImm(42))
        v1 = ctx.v_add_u32(v0, KImm(1))
        ctx.s_endpgm()

        asm_lines, stats = ctx.finalize()

        assert len(asm_lines) > 0
        assert any("v_mov_b32" in line for line in asm_lines)
        assert any("v_add_u32" in line for line in asm_lines)
        assert any("s_endpgm" in line for line in asm_lines)

    def test_cse_in_context(self):
        """Test CSE functionality in context."""
        ctx = KernelCompilationContext()

        key1 = ("add", 10, 5)
        key2 = ("add", 10, 5)  # Same key

        result1 = ctx.cse_get_or_emit(key1, lambda: ctx.vreg())
        result2 = ctx.cse_get_or_emit(key2, lambda: ctx.vreg())

        # Should return same virtual register
        assert result1 == result2
        assert ctx.cse_hit_count == 1


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_address_calculation_pattern(self):
        """Test a realistic address calculation pattern."""
        ctx = KernelCompilationContext()

        # Simulate: addr = (tid_x * 16) + (tid_y * 256)
        # tid_x, tid_y would come from ABI regs in real code

        tid_x = ctx.v_mov_b32(KImm(5))  # Simulated tid_x
        tid_y = ctx.v_mov_b32(KImm(3))  # Simulated tid_y

        # tid_x * 16 = tid_x << 4
        offset_x = ctx.v_lshlrev_b32(KImm(4), tid_x)

        # tid_y * 256 = tid_y << 8
        offset_y = ctx.v_lshlrev_b32(KImm(8), tid_y)

        # addr = offset_x + offset_y
        addr = ctx.v_add_u32(offset_x, offset_y)

        # Load from LDS
        data = ctx.ds_read_b64(addr)

        ctx.s_barrier()
        ctx.s_endpgm()

        asm_lines, stats = ctx.finalize()

        # Verify all instructions are present
        asm = "\n".join(asm_lines)
        assert "v_lshlrev_b32" in asm
        # v_add_u32 may be fused into v_lshl_add_u32 by peephole optimization
        assert "v_add_u32" in asm or "v_lshl_add_u32" in asm
        assert "ds_read_b64" in asm
        assert "s_barrier" in asm
        assert "s_endpgm" in asm

        # Verify allocation stats are reasonable
        assert stats.peak_vgprs > 0
        assert stats.total_vregs > 0

    def test_multiple_loads_with_reuse(self):
        """Test that register reuse works across multiple operations."""
        ctx = KernelCompilationContext()

        # First load
        addr1 = ctx.v_mov_b32(KImm(0))
        data1 = ctx.ds_read_b64(addr1)

        # Process data1 (makes addr1 dead)
        result1 = ctx.v_add_u32(
            KRegRange(data1.base_reg, 1, 1), KImm(1)  # Use first reg of pair
        )

        # Second load (should be able to reuse addr1's register)
        addr2 = ctx.v_mov_b32(KImm(64))
        data2 = ctx.ds_read_b64(addr2)

        ctx.s_endpgm()

        asm_lines, stats = ctx.finalize()

        # Should have allocated registers
        assert stats.peak_vgprs > 0
        # With good allocation, should reuse dead registers
        # Can't assert exact count since it depends on allocation order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
