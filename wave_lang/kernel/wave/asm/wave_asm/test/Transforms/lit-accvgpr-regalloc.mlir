// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s

// CHECK-LABEL: waveasm.program @mfma_agpr_tied
waveasm.program @mfma_agpr_tied target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  // CHECK: waveasm.precolored.areg 32, 4 : !waveasm.pareg<32, 4>
  %acc0 = waveasm.precolored.areg 32, 4 : !waveasm.pareg<32, 4>

  // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pareg<32, 4>
  %acc1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.pareg<32, 4> -> !waveasm.areg<4, 4>

  // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pareg<32, 4>
  %acc2 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc1
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.areg<4, 4> -> !waveasm.areg<4, 4>

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @scaled_mfma_agpr_tied
waveasm.program @scaled_mfma_agpr_tied target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %as = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>
  %bs = waveasm.precolored.vreg 9 : !waveasm.pvreg<9>
  %acc0 = waveasm.precolored.areg 40, 4 : !waveasm.pareg<40, 4>

  // CHECK: waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}} -> !waveasm.pareg<40, 4>
  %acc1 = waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 %a, %b, %acc0, %as, %bs
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.pareg<40, 4>, !waveasm.pvreg<8>, !waveasm.pvreg<9> -> !waveasm.areg<4, 4>

  waveasm.s_endpgm
}
