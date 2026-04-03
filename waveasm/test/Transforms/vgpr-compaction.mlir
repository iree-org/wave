// RUN: waveasm-translate --waveasm-linear-scan='max-vgprs=256 max-agprs=256' --waveasm-vgpr-compaction --emit-assembly %s | FileCheck %s
//
// Tests for the VGPRCompaction pass.
//
// Each test verifies that compaction reduces the .vgpr_count metadata
// below what the linear scan allocator produces.  Without the compaction
// pass, the allocator leaves fragmentation gaps that inflate the peak
// register number.

// -----

// Test 1: Compaction repacks accumulator past precolored gap
//
// The allocator places the 4-wide accumulator at v[16:19] (skipping the
// precolored v[0], v[4:7], v[8:11] and the gap v[12:15] which includes
// scratch v15).  Compaction repacks it to v[0:3] since the precolored
// values are only defined (not live across the MFMA), yielding
// vgpr_count=12 instead of 20.
//
// Without compaction: .vgpr_count: 20
// With compaction:    .vgpr_count: 12
//
// CHECK-LABEL: pinned_compact
// CHECK: .vgpr_count: 12
waveasm.program @pinned_compact
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %mfma = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// -----

// Test 2: v15 (scratch VGPR) must not be used by compaction
//
// Allocate enough 1-wide values that compaction would naturally want to
// place one at v15.  Verify no user operation writes to v15 : it should
// only appear in literal materialization (v_mov_b32 v15, <const>).
//
// CHECK-LABEL: scratch_vgpr_skip
// CHECK-NOT: v_add_u32 {{.*}}v15
// CHECK: s_endpgm
waveasm.program @scratch_vgpr_skip
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %r1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %r1, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r3 = waveasm.v_add_u32 %r2, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r4 = waveasm.v_add_u32 %r3, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r5 = waveasm.v_add_u32 %r4, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r6 = waveasm.v_add_u32 %r5, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r7 = waveasm.v_add_u32 %r6, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
  %r8 = waveasm.v_add_u32 %r7, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// -----

// Test 3: IfOp result and block-argument remapping
//
// Verifies that the compaction pass correctly handles IfOp: the MFMA
// result flows through an if/else and the pass must remap the IfOp
// result type and yield operand types without crashing.  The assembly
// should contain valid branch structure (s_cbranch) and s_endpgm.
//
// CHECK-LABEL: ifop_compact
// CHECK: s_cbranch
// CHECK: s_endpgm
waveasm.program @ifop_compact
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %mfma = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  %s0 = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %cond = waveasm.s_cmp_lt_u32 %s0, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.scc

  %if_result = waveasm.if %cond : !waveasm.scc -> !waveasm.vreg {
    %r1 = waveasm.v_add_u32 %mfma, %c1 : !waveasm.vreg<4, 4>, !waveasm.imm<1> -> !waveasm.vreg
    waveasm.yield %r1 : !waveasm.vreg
  } else {
    %r2 = waveasm.v_add_u32 %mfma, %c0 : !waveasm.vreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg
    waveasm.yield %r2 : !waveasm.vreg
  }

  waveasm.s_endpgm
}

// -----

// Test 4: LoopOp result, block-argument, and iter_arg remapping
//
// An MFMA accumulator is carried through a loop as an iter_arg.
// The precolored gap (v[4:7], v[8:11]) forces the accumulator to
// v[16:19].  Compaction remaps it and must consistently update:
//   - LoopOp block arguments (loop-carried phi)
//   - MFMA result inside the loop body
//   - ConditionOp iter_args and _iterArgPhysRegs attribute
//   - LoopOp results (post-loop value)
//
// The MFMA inside the body and the v_add_u32 after the loop must
// reference the same accumulator register range.  A double-remap
// bug (Walk 1 + Walk 2 both remapping results) would cause them
// to diverge when oldToNew contains chains.
//
// CHECK-LABEL: loopop_compact
// CHECK: v_mfma_f32_16x16x16_f16 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
// CHECK: v_add_u32 v{{[0-9]+}}, v[[[LO]]:[[HI]]]
// CHECK: s_endpgm
waveasm.program @loopop_compact
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c4 : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc iter_args(%next_i#0, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  %use = waveasm.v_add_u32 %acc_out, %c0 : !waveasm.vreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
