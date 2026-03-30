// RUN: not waveasm-translate --emit-assembly %s 2>&1 | FileCheck %s
//
// Verify that assembly emission catches out-of-range physical registers
// and produces a single clear error instead of letting the assembler
// produce many "register index is out of range" errors.
//
// This kernel has physical register indices beyond the gfx1250 target's
// 256-VGPR limit (pvreg<260> is v260, which doesn't exist on gfx1250).

// CHECK: error: kernel 'emit_overflow_test' exceeds hardware register limit: VGPRs: {{[0-9]+}} used, 256 limit{{$}}
// CHECK-NOT: register index is out of range

waveasm.program @emit_overflow_test
  target = #waveasm.target<#waveasm.gfx1250, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0> {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>

  // Physical register index 260 exceeds gfx1250's 256-VGPR limit.
  %big = waveasm.precolored.vreg 260, 4 : !waveasm.pvreg<260, 4>
  %result = waveasm.v_add_u32 %big, %c1 : !waveasm.pvreg<260, 4>, !waveasm.imm<1> -> !waveasm.pvreg<265>

  waveasm.s_endpgm
}
