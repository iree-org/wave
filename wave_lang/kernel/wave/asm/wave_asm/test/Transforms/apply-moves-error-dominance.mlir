// RUN: not waveasm-conductor %s 2>&1 | FileCheck %s

// CONDUCTOR: move v_add_u32_0 after v_add_u32_1

// v_add_u32_1 uses the result of v_add_u32_0, so moving _0 after _1
// breaks dominance.

// CHECK: does not dominate this use
// CHECK: conductor: verification failed after applying moves

waveasm.program @test_dominance target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>

  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg
  %a1 = waveasm.v_add_u32 %a0, %c4 : !waveasm.vreg, !waveasm.imm<4> -> !waveasm.vreg

  waveasm.s_endpgm
}
