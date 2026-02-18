// RUN: waveasm-translate --emit-assembly %s | FileCheck %s

// CHECK-LABEL: accvgpr_move_test:
// CHECK: v_accvgpr_read_b32 v10, a12
// CHECK: v_accvgpr_write_b32 v10, a12
waveasm.program @accvgpr_move_test target = #waveasm.target<#waveasm.gfx950, 5> abi = #waveasm.abi<> {
  %v10 = waveasm.precolored.vreg 10 : !waveasm.pvreg<10>
  %a12 = waveasm.precolored.areg 12 : !waveasm.pareg<12>

  %tmp = waveasm.v_accvgpr_read_b32 %a12 : !waveasm.pareg<12> -> !waveasm.pvreg<10>
  waveasm.v_accvgpr_write_b32 %tmp, %a12 : !waveasm.pvreg<10>, !waveasm.pareg<12>

  waveasm.s_endpgm
}
