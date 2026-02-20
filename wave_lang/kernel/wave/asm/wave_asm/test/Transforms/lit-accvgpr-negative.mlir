// RUN: not waveasm-translate --waveasm-linear-scan --max-agprs=4 %s 2>&1 | FileCheck %s

waveasm.program @accvgpr_limit_fail target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  // Reserve all available AGPRs (limit is 4 in RUN line).
  %acc0 = waveasm.precolored.areg 0, 4 : !waveasm.pareg<0, 4>

  // Any new 4-wide AGPR allocation must fail.
  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %r0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %zero
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.imm<0> -> !waveasm.areg<4, 4>
  waveasm.s_endpgm
}

// CHECK: Failed to allocate AGPR
