// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: memref.atomic_rmw generates buffer_atomic instructions
// Note: This test may need adjustment based on the exact translation pipeline flags available.
// For now, test the WaveASM dialect op directly.

// CHECK-LABEL: atomic_rmw_f32_test:

waveasm.program @atomic_rmw_f32_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %s0 = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // CHECK: buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen
  waveasm.buffer_atomic_add_f32 %v0, %s0, %v1 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
