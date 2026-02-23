// RUN: waveasm-conductor --print-debug-locs-inline %s | FileCheck %s

// CONDUCTOR: swap v_add_u32_0 v_lshlrev_b32_0

// Three independent ops (all read from %v0/%c4) — swap is safe.

// CHECK-LABEL: waveasm.program @test_swap
// CHECK: waveasm.v_lshlrev_b32{{.*}}loc("v_lshlrev_b32_0")
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_1")
// CHECK: waveasm.v_add_u32{{.*}}loc("v_add_u32_0")
waveasm.program @test_swap target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>

  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg loc("v_add_u32_0")
  %a1 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg loc("v_add_u32_1")
  %s0 = waveasm.v_lshlrev_b32 %c4, %v0 : !waveasm.imm<4>, !waveasm.pvreg<0> -> !waveasm.vreg loc("v_lshlrev_b32_0")

  waveasm.s_endpgm loc("s_endpgm_0")
}
