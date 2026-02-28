// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: FP truncation and extension conversions

// CHECK-LABEL: truncf_extf_test:

waveasm.program @truncf_extf_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // f32 -> bf16: should emit v_cvt_pk_bf16_f32 dst, src, 0
  // CHECK: v_cvt_pk_bf16_f32 v{{[0-9]+}}, v0, 0
  %r0 = waveasm.v_cvt_bf16_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // bf16 -> f32
  // CHECK: v_cvt_f32_bf16 v{{[0-9]+}}, v{{[0-9]+}}
  %r1 = waveasm.v_cvt_f32_bf16 %r0 : !waveasm.vreg -> !waveasm.vreg

  // f32 -> f16
  // CHECK: v_cvt_f16_f32 v{{[0-9]+}}, v0
  %r2 = waveasm.v_cvt_f16_f32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // f16 -> f32
  // CHECK: v_cvt_f32_f16 v{{[0-9]+}}, v{{[0-9]+}}
  %r3 = waveasm.v_cvt_f32_f16 %r2 : !waveasm.vreg -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
