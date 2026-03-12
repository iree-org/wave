// RUN: waveasm-translate --waveasm-m0-redundancy-elim %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Redundant M0 write eliminated.
//
// Two consecutive s_mov_b32_m0 with the same source — second is redundant.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_m0_redundant
waveasm.program @test_m0_redundant target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %lds_size = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  // First M0 write — kept.
  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_size : !waveasm.sreg

  // Second M0 write with same source — eliminated.
  // CHECK-NOT: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_size : !waveasm.sreg

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Different-source M0 writes preserved.
//
// Two s_mov_b32_m0 with different sources — both must survive.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @test_m0_different_sources
waveasm.program @test_m0_different_sources target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1024 : !waveasm.imm<1024>
  %val0 = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %val1 = waveasm.s_mov_b32 %c1 : !waveasm.imm<1024> -> !waveasm.sreg

  // Both M0 writes have different sources — both preserved.
  // CHECK: waveasm.s_mov_b32_m0 %{{.*}} : !waveasm.sreg
  // CHECK: waveasm.s_mov_b32_m0 %{{.*}} : !waveasm.sreg
  waveasm.s_mov_b32_m0 %val0 : !waveasm.sreg
  waveasm.s_mov_b32_m0 %val1 : !waveasm.sreg

  waveasm.s_endpgm
}
