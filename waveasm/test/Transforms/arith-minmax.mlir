// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: integer min/max operations

// CHECK-LABEL: waveasm.program @int_minmax
waveasm.program @int_minmax target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // CHECK: waveasm.v_min_i32
  %v0 = waveasm.v_min_i32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_max_i32
  %v1 = waveasm.v_max_i32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_min_u32
  %v2 = waveasm.v_min_u32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_max_u32
  %v3 = waveasm.v_max_u32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}
