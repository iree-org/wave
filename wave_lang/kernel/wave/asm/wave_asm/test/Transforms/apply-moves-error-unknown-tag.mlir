// RUN: not waveasm-conductor %s 2>&1 | FileCheck %s

// CONDUCTOR: move nonexistent_tag_42 after v_add_u32_0


// CHECK: conductor: command 0: unknown tag 'nonexistent_tag_42'

waveasm.program @test_unknown_tag target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg
  waveasm.s_endpgm
}
