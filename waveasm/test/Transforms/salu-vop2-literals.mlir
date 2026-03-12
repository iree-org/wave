// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: SALU and VOP2 literal operand handling.
//
// SALU instructions support 32-bit literals natively -- no materialization
// into scratch VGPR needed.
//
// VOP2 instructions require the literal in src0.  For commutative ops with
// the literal in src1, the emitter swaps operands.  For non-commutative ops
// it falls back to scratch VGPR materialization.

// CHECK-LABEL: salu_literal_test:

waveasm.program @salu_literal_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>

  // SALU with large literal: should emit directly, no v_mov_b32
  // CHECK-NOT: v_mov_b32
  // CHECK: s_add_u32 s{{[0-9]+}}, s0, 4096
  %c4096 = waveasm.constant 4096 : !waveasm.imm<4096>
  %r1:2 = waveasm.s_add_u32 %s0, %c4096 : !waveasm.psreg<0>, !waveasm.imm<4096> -> !waveasm.sreg, !waveasm.sreg
  // Another SALU with literal -- should also be direct
  // CHECK-NOT: v_mov_b32
  // CHECK: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 7
  %c7 = waveasm.constant 7 : !waveasm.imm<7>
  %r2 = waveasm.s_lshl_b32 %r1#0, %c7 : !waveasm.sreg, !waveasm.imm<7> -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: vop2_literal_test:

waveasm.program @vop2_literal_test target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // VOP2 commutative op with literal in src1: should swap to put literal in src0
  // v_add_u32 is commutative, so emit literal first
  // CHECK-NOT: v_mov_b32 v15
  // CHECK: v_add_u32 v{{[0-9]+}}, 256, v0
  %c256 = waveasm.constant 256 : !waveasm.imm<256>
  %r1 = waveasm.v_add_u32 %v0, %c256 : !waveasm.pvreg<0>, !waveasm.imm<256> -> !waveasm.vreg

  // VOP2 with literal already in src0: should emit directly
  // CHECK-NOT: v_mov_b32 v15
  // CHECK: v_add_u32 v{{[0-9]+}}, 512, v{{[0-9]+}}
  %c512 = waveasm.constant 512 : !waveasm.imm<512>
  %r2 = waveasm.v_add_u32 %c512, %r1#0 : !waveasm.imm<512>, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
