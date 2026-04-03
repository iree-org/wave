// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test bidirectional VGPR allocation:
//
// Multi-register VGPR ranges whose length exceeds 75% of the program span are
// allocated from the top of the expected register usage (tryAllocateFromTop),
// while shorter-lived multi-register ranges are allocated from the bottom
// (tryAllocate). This separates interleaved buffer_load (prefetch, long-lived)
// and ds_read (consumed quickly, short-lived) destinations into contiguous
// regions, reducing fragmentation and peak VGPR count.
//
// Single-register VGPRs always use bottom-up allocation regardless of
// lifetime, because the bidirectional heuristic only applies to size > 1.

//===----------------------------------------------------------------------===//
// Test 1: Long-lived buffer_load vs short-lived ds_read
//
// %buf spans almost the entire program (long-lived), so it should be allocated
// from the top of the expected register usage via tryAllocateFromTop.
// %ds is consumed immediately (short-lived), so it should be allocated from
// the bottom via tryAllocate.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @bidirectional_basic
// CHECK-NOT: Failed to allocate
waveasm.program @bidirectional_basic
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Long-lived 4-wide VGPR: defined early, used at the very end.
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<[[BUF:[0-9]+]], 4>
  %buf = waveasm.buffer_load_dwordx4 %srd, %v0, %c0
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0>
      -> !waveasm.vreg<4, 4>

  // Padding operations to extend the program span so that %buf's range
  // length exceeds 75% of the program span, triggering top allocation.
  %t1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %t2 = waveasm.v_add_u32 %t1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t3 = waveasm.v_add_u32 %t2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t4 = waveasm.v_add_u32 %t3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // Short-lived 4-wide VGPR: defined and consumed within 2 operations.
  // CHECK: waveasm.ds_read_b128 {{.*}} -> !waveasm.pvreg<[[DS:[0-9]+]], 4>
  %ds = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>
  %ds_use = waveasm.v_add_u32 %ds, %v0
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // More padding after ds_read consumption
  %t5 = waveasm.v_add_u32 %t4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t6 = waveasm.v_add_u32 %t5, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t7 = waveasm.v_add_u32 %t6, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t8 = waveasm.v_add_u32 %t7, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // Late use of buffer_load result — keeps %buf alive across the entire program
  %buf_use = waveasm.v_add_u32 %buf, %v0
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: Two long-lived buffer_loads and two short-lived ds_reads in a loop
//
// Models the real double-buffer GEMM pattern: buffer_load prefetch values live
// across almost the entire loop body while ds_read values are consumed by MFMA
// within a few ops. The bidirectional allocator should pack them into disjoint
// regions (top for prefetch, bottom for consumed), reducing fragmentation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @bidirectional_loop_double_buffer
// CHECK-NOT: Failed to allocate
waveasm.program @bidirectional_loop_double_buffer
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: -> !waveasm.pvreg<[[ACC0:[0-9]+]], 4>
  %init_acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  // CHECK: -> !waveasm.pvreg<[[ACC1:[0-9]+]], 4>
  %init_acc1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %r0, %r1 = waveasm.loop(
      %i = %init_i, %acc0 = %init_acc0, %acc1 = %init_acc1)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    // Buffer loads at the top of the loop — results live until the end.
    // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
    %buf0 = waveasm.buffer_load_dwordx4 %srd, %v0, %c0
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0>
        -> !waveasm.vreg<4, 4>
    // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
    %buf1 = waveasm.buffer_load_dwordx4 %srd, %v0, %c1
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<1>
        -> !waveasm.vreg<4, 4>

    // Short-lived ds_reads — consumed immediately by MFMA
    // CHECK: waveasm.ds_read_b128 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
    %ds0 = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>
    // CHECK: waveasm.ds_read_b128 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
    %ds1 = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

    // MFMA using ds_read results — kills ds_read values quickly
    // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[ACC0]], 4>
    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %ds0, %ds1, %acc0
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>
    // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[ACC1]], 4>
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %ds0, %ds1, %acc1
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>

    // Padding — extends buffer_load lifetimes further
    %p1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
    %p2 = waveasm.v_add_u32 %p1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p3 = waveasm.v_add_u32 %p2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p4 = waveasm.v_add_u32 %p3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p5 = waveasm.v_add_u32 %p4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p6 = waveasm.v_add_u32 %p5, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p7 = waveasm.v_add_u32 %p6, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %p8 = waveasm.v_add_u32 %p7, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

    // Late use of buffer_load results — keeps them alive across the loop body
    %buf0_use = waveasm.v_mfma_f32_16x16x16_f16 %buf0, %a, %n0
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>
    %buf1_use = waveasm.v_mfma_f32_16x16x16_f16 %buf1, %b, %n1
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4>
        -> !waveasm.vreg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1
        : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c16
        : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i#0, %buf0_use, %buf1_use)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  %use0 = waveasm.v_add_u32 %r0, %v0
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %use1 = waveasm.v_add_u32 %r1, %v0
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: Single-register VGPRs skip bidirectional even when long-lived
//
// Bidirectional allocation only applies to multi-register (size > 1) ranges.
// Single-register VGPRs always use bottom-up allocation. Verify that a
// long-lived single-register value is still allocated from the bottom.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @single_reg_not_bidirectional
// CHECK-NOT: Failed to allocate
waveasm.program @single_reg_not_bidirectional
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Long-lived single VGPR — should NOT trigger bidirectional (size == 1)
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[SINGLE:[0-9]+]]>
  %long_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %t1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %t2 = waveasm.v_add_u32 %t1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t3 = waveasm.v_add_u32 %t2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t4 = waveasm.v_add_u32 %t3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t5 = waveasm.v_add_u32 %t4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t6 = waveasm.v_add_u32 %t5, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t7 = waveasm.v_add_u32 %t6, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %t8 = waveasm.v_add_u32 %t7, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // Late use — keeps %long_val alive across the program.
  // Single VGPR should still be allocated from the bottom (low register).
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[SINGLE]]>{{.*}} -> !waveasm.pvreg<
  %result = waveasm.v_add_u32 %long_val, %v0
      : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 4: SGPR ranges skip bidirectional entirely
//
// Bidirectional allocation only applies to VGPRs. SGPR ranges always use
// bottom-up allocation regardless of lifetime or size.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @sgpr_not_bidirectional
// CHECK-NOT: Failed to allocate
waveasm.program @sgpr_not_bidirectional
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // Long-lived 4-wide SGPR — bidirectional should NOT apply (not VGPR)
  // CHECK: waveasm.s_load_dwordx4 {{.*}} -> !waveasm.psreg<{{[0-9]+}}, 4>
  %ptr = waveasm.s_load_dwordx4 %srd, %c0
      : !waveasm.psreg<0, 4>, !waveasm.imm<0> -> !waveasm.sreg<4>

  %s1 = waveasm.s_mov_b32 %c1 : !waveasm.imm<1> -> !waveasm.sreg
  %s2:2 = waveasm.s_add_u32 %s1, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s3:2 = waveasm.s_add_u32 %s2#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s4:2 = waveasm.s_add_u32 %s3#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s5:2 = waveasm.s_add_u32 %s4#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s6:2 = waveasm.s_add_u32 %s5#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s7:2 = waveasm.s_add_u32 %s6#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
  %s8:2 = waveasm.s_add_u32 %s7#0, %c1
      : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg

  // Late use of %ptr — long-lived SGPR quad
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %loaded = waveasm.buffer_load_dwordx4 %ptr, %v0, %c0
      : !waveasm.sreg<4>, !waveasm.pvreg<0>, !waveasm.imm<0>
      -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}
