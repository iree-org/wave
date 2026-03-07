// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: vector.extract from AGPR source preserves AGPR type

// CHECK-LABEL: waveasm.program @agpr_extract_test
waveasm.program @agpr_extract_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Use an AGPR register
  %a0 = waveasm.precolored.areg 0, 4 : !waveasm.pareg<0, 4>

  // Extracting from precolored AGPR produces a precolored AGPR element
  // CHECK: waveasm.extract %{{.*}} : !waveasm.pareg<0, 4> -> !waveasm.pareg<0>
  %elem = waveasm.extract %a0[0] : !waveasm.pareg<0, 4> -> !waveasm.pareg<0>

  waveasm.s_endpgm
}
