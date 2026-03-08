// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: buffer_store_short and buffer_store_byte assembly emission

// CHECK-LABEL: subdword_store_test:

waveasm.program @subdword_store_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %s0 = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // CHECK: buffer_store_short v0, v1, s[0:3], 0 offen
  waveasm.buffer_store_short %v0, %s0, %v1 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: buffer_store_byte v0, v1, s[0:3], 0 offen
  waveasm.buffer_store_byte %v0, %s0, %v1 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: buffer_store_short v0, v1, s[0:3], 0 offen offset:64
  waveasm.buffer_store_short %v0, %s0, %v1 offset : 64 : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
