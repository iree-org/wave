// RUN: waveasm-translate --emit-assembly %s | FileCheck %s

// CHECK-LABEL: accvgpr_emit_test:
// CHECK: v_accvgpr_read_b32 v2, a0
// CHECK: v_accvgpr_write_b32 v2, a0
// CHECK: v_mfma_f32_16x16x16_f16 a[4:7], v[0:3], v[0:3], a[4:7]
// CHECK: .amdhsa_accum_offset
// CHECK: .amdhsa_next_free_vgpr
waveasm.program @accvgpr_emit_test target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>
  %a0 = waveasm.precolored.areg 0 : !waveasm.pareg<0>
  %acc = waveasm.precolored.areg 4, 4 : !waveasm.pareg<4, 4>

  %r = waveasm.v_accvgpr_read_b32 %a0 : !waveasm.pareg<0> -> !waveasm.pvreg<2>
  waveasm.v_accvgpr_write_b32 %r, %a0 : !waveasm.pvreg<2>, !waveasm.pareg<0>
  %next = waveasm.v_mfma_f32_16x16x16_f16 %v0, %v0, %acc
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<0, 4>, !waveasm.pareg<4, 4> -> !waveasm.pareg<4, 4>

  waveasm.s_endpgm
}
