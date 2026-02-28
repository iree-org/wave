// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test v_fma_f32 instruction

// CHECK-LABEL: waveasm.program @test_fma_f32
waveasm.program @test_fma_f32 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Define operand registers
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %c = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  // Test v_fma_f32: dst = a * b + c
  // CHECK: waveasm.v_fma_f32
  %result = waveasm.v_fma_f32 %a, %b, %c : !waveasm.pvreg<0>, !waveasm.pvreg<1>, !waveasm.pvreg<2> -> !waveasm.vreg

  waveasm.s_endpgm
}
