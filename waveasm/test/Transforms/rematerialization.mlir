// RUN: waveasm-translate --waveasm-linear-scan='max-vgprs=48 max-agprs=256' %s 2>&1 | FileCheck %s
//
// Tests for the rematerialization pass in LinearScanPass.
//
// The pass clones cheap-to-compute ops (v_mov_b32, s_mov_b32) near their
// use sites to shorten live ranges and reduce peak register pressure.
// Each test is calibrated so that without rematerialization the VGPR budget
// (48) is exceeded (needs 49), but with rematerialization the cloned ops
// have short live ranges and allocation succeeds.
//
// Coverage:
//   @remat_vmov_immediate  — v_mov_b32 from immediate (accumulator zero-init)
//   @remat_vmov_sgpr       — v_mov_b32 from SGPR (scalar-to-vector copy)
//   @remat_no_clone_into_loop — VGPR ops outside loop are NOT cloned inside

// =====================================================================
// Test 1: v_mov_b32 from immediate
//
// A v_mov_b32 %zero, 0 defined early but used 20+ instructions later
// holds a vreg<4,4> (4 VGPRs) live across all the filler ops.
// Without remat: 4 extra VGPRs push past the 48 budget.
// With remat: clone appears right before the MFMA, freeing those 4
// VGPRs for the intervening span.
// =====================================================================

// CHECK-LABEL: waveasm.program @remat_vmov_immediate
// CHECK-NOT: Failed to allocate
// CHECK: v_mfma_f32_16x16x16_f16
waveasm.program @remat_vmov_immediate
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %zero = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %f0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f4 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f5 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f6 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f7 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f8 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f9 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f10 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %u0 = waveasm.v_add_u32 %f0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u1 = waveasm.v_add_u32 %f1, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u2 = waveasm.v_add_u32 %f2, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u3 = waveasm.v_add_u32 %f3, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u4 = waveasm.v_add_u32 %f4, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u5 = waveasm.v_add_u32 %f5, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u6 = waveasm.v_add_u32 %f6, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u7 = waveasm.v_add_u32 %f7, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u8 = waveasm.v_add_u32 %f8, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u9 = waveasm.v_add_u32 %f9, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u10 = waveasm.v_add_u32 %f10, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  %mfma = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %zero : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// =====================================================================
// Test 2: v_mov_b32 from SGPR (scalar-to-vector address copy)
//
// Same pressure pattern as test 1, but the v_mov_b32 source is an SGPR
// instead of an immediate.  The SGPR dominates all use sites so
// cloning is safe.
// =====================================================================

// CHECK-LABEL: waveasm.program @remat_vmov_sgpr
// CHECK-NOT: Failed to allocate
// CHECK: v_mfma_f32_16x16x16_f16
waveasm.program @remat_vmov_sgpr
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %saddr = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %vaddr = waveasm.v_mov_b32 %saddr : !waveasm.sreg -> !waveasm.vreg<4, 4>

  %f0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f4 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f5 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f6 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f7 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f8 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f9 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f10 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %u0 = waveasm.v_add_u32 %f0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u1 = waveasm.v_add_u32 %f1, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u2 = waveasm.v_add_u32 %f2, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u3 = waveasm.v_add_u32 %f3, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u4 = waveasm.v_add_u32 %f4, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u5 = waveasm.v_add_u32 %f5, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u6 = waveasm.v_add_u32 %f6, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u7 = waveasm.v_add_u32 %f7, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u8 = waveasm.v_add_u32 %f8, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u9 = waveasm.v_add_u32 %f9, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u10 = waveasm.v_add_u32 %f10, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  %mfma = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %vaddr : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// =====================================================================
// Test 3: VGPR ops outside loop must NOT be cloned into loop body
//
// A v_mov_b32 defined before a loop has two uses:
//   1. Inside the loop (must NOT be cloned — would add VALU to the
//      critical MFMA path without reducing in-loop pressure)
//   2. After the loop with filler in between (SHOULD be cloned to
//      shorten the live range)
//
// Without remat: %addr is live from definition through the loop and
//   past all post-loop filler → exceeds 48-VGPR budget.
// With remat: the post-loop use gets a clone, the in-loop use is
//   skipped → allocation succeeds.
// =====================================================================

// CHECK-LABEL: waveasm.program @remat_no_clone_into_loop
// CHECK-NOT: Failed to allocate
waveasm.program @remat_no_clone_into_loop
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %addr = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %z0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %r0 = waveasm.loop(
      %i = %init_i, %acc0 = %z0)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    %sum = waveasm.v_add_u32 %acc0, %addr : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    // CHECK: v_mfma_f32_16x16x16_f16
    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %sum : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.scc
    // CHECK: waveasm.condition
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c16 : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.scc
    waveasm.condition %cond : !waveasm.scc iter_args(%next_i#0, %n0)
        : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  %f0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f4 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f5 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f6 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f7 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f8 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %f9 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %u0 = waveasm.v_add_u32 %f0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u1 = waveasm.v_add_u32 %f1, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u2 = waveasm.v_add_u32 %f2, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u3 = waveasm.v_add_u32 %f3, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u4 = waveasm.v_add_u32 %f4, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u5 = waveasm.v_add_u32 %f5, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u6 = waveasm.v_add_u32 %f6, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u7 = waveasm.v_add_u32 %f7, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u8 = waveasm.v_add_u32 %f8, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %u9 = waveasm.v_add_u32 %f9, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  %final = waveasm.v_add_u32 %r0, %addr : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}
