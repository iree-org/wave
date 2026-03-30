// RUN: waveasm-translate %s --waveasm-linear-scan 2>&1 | FileCheck %s

// Test InsertOp with VGPR operands.
// The mov into the target slot uses v_mov_b32 with PVRegType destination.

// CHECK-LABEL: waveasm.program @insert_vgpr
waveasm.program @insert_vgpr
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  %vec = waveasm.precolored.vreg 0 {size = 4} : !waveasm.vreg<4>
  %val = waveasm.precolored.vreg 10 : !waveasm.vreg

  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<2>
  // CHECK-NEXT: waveasm.dce_protect {{.*}} : !waveasm.pvreg<2>
  // CHECK-NEXT: waveasm.insert {{.*}} into {{.*}}[2] {{.*}} -> !waveasm.pvreg<0, 4>
  %updated = waveasm.insert %val into %vec[2] : !waveasm.vreg, !waveasm.vreg<4> -> !waveasm.vreg<4>
  waveasm.dce_protect %updated : !waveasm.vreg<4>

  waveasm.s_endpgm
}
