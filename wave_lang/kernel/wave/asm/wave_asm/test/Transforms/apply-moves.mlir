// RUN: waveasm-conductor --print-debug-locs-inline %s | FileCheck %s

// CONDUCTOR: move v_add_u32_1 before v_add_u32_0

// Two independent adds from the same inputs — reordering is safe.

// CHECK-LABEL: waveasm.program @test_move_before
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_1")
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_0")
waveasm.program @test_move_before target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>

  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg
  %a1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg

  waveasm.s_endpgm
}
