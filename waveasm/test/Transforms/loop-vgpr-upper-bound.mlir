// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: v_readfirstlane_b32 is inserted for VGPR loop upper bounds

// CHECK-LABEL: waveasm.program @vgpr_loop_bound
waveasm.program @vgpr_loop_bound target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_min_i32
  %ub = waveasm.v_min_i32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
