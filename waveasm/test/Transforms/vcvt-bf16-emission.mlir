// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_cvt_bf16_f32 emits as v_cvt_pk_bf16_f32 dst, src, 0

// CHECK-LABEL: vcvt_bf16_test:

waveasm.program @vcvt_bf16_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: v_cvt_pk_bf16_f32 v{{[0-9]+}}, v0, 0
  %r = waveasm.v_cvt_bf16_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
