// RUN: waveasm-conductor --print-debug-locs-inline %s | FileCheck %s

// CONDUCTOR: move v_add_u32_1 before v_add_u32_0


// Original order: v_add_u32_0, v_add_u32_1, v_lshlrev_b32_0.
// After move: v_add_u32_1, v_add_u32_0, v_lshlrev_b32_0.
// Note: output is in generic form because moves may create use-before-def.

// CHECK: sym_name = "test_move_before"
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_1")
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_0")
// CHECK: waveasm.v_lshlrev_b32{{.*}}loc("v_lshlrev_b32_0")
waveasm.program @test_move_before target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>

  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg
  %a1 = waveasm.v_add_u32 %a0, %c4 : !waveasm.vreg, !waveasm.imm<4> -> !waveasm.vreg
  %s0 = waveasm.v_lshlrev_b32 %c4, %a1 : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
