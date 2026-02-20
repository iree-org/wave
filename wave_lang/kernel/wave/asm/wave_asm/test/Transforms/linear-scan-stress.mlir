// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Stress tests for the linear scan register allocator.
// These tests all PASS -- they verify correct behavior under various
// pressure scenarios. See linear-scan-ifop-bug.mlir for known bugs.

//===----------------------------------------------------------------------===//
// Test 1: Register reuse after expiry
//
// When a value dies and its register is freed, a later value should reuse it.
// Verifies expireRanges() correctly returns registers to the pool.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @reuse_after_expiry
// CHECK-NOT: Failed to allocate
waveasm.program @reuse_after_expiry
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // First chain: values die before second chain starts
  %a1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %a2 = waveasm.v_add_u32 %a1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %a3 = waveasm.v_add_u32 %a2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %a4 = waveasm.v_add_u32 %a3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // Second chain: should reuse registers freed from first chain
  %b1 = waveasm.v_add_u32 %a4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %b2 = waveasm.v_add_u32 %b1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %b3 = waveasm.v_add_u32 %b2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %b4 = waveasm.v_add_u32 %b3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: Fragmented multi-register ranges
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @fragmented_multi_reg
// CHECK-NOT: Failed to allocate
waveasm.program @fragmented_multi_reg
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %vec1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %s1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %s2 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %s3 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %kill_s = waveasm.v_add_u32 %s1, %s2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  %kill_s2 = waveasm.v_add_u32 %kill_s, %s3 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // CHECK: -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %vec2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %use1 = waveasm.v_add_u32 %vec1, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %use2 = waveasm.v_add_u32 %vec2, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %use3 = waveasm.v_add_u32 %kill_s2, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: Loop with ds_read results competing with accumulators
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @loop_dsread_acc_competition
// CHECK-NOT: Failed to allocate
waveasm.program @loop_dsread_acc_competition
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_acc1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_acc2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_acc3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %r0, %r1, %r2, %r3 = waveasm.loop(
      %i = %init_i,
      %acc0 = %init_acc0, %acc1 = %init_acc1,
      %acc2 = %init_acc2, %acc3 = %init_acc3)
      : (!waveasm.sreg,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %lds_data0 = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>
    %lds_data1 = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %lds_data0, %lds_data1, %acc0
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %lds_data0, %lds_data1, %acc1
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n2 = waveasm.v_mfma_f32_16x16x16_f16 %lds_data0, %lds_data1, %acc2
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n3 = waveasm.v_mfma_f32_16x16x16_f16 %lds_data0, %lds_data1, %acc3
        : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %n0, %n1, %n2, %n3)
        : !waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  %use0 = waveasm.v_add_u32 %r0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %use1 = waveasm.v_add_u32 %r3, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 4: Value defined before loop, used inside and after
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @value_live_across_loop
// CHECK-NOT: Failed to allocate
waveasm.program @value_live_across_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: -> !waveasm.pvreg<[[OUTSIDE:[0-9]+]]>
  %outside_val = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_sum = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %sum_out = waveasm.loop(%i = %init_i, %sum = %init_sum)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[OUTSIDE]]>
    %tmp = waveasm.v_add_u32 %sum, %outside_val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %tmp2 = waveasm.v_add_u32 %tmp, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %tmp3 = waveasm.v_add_u32 %tmp2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %tmp3) : !waveasm.sreg, !waveasm.vreg
  }

  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[OUTSIDE]]>
  %post = waveasm.v_add_u32 %sum_out, %outside_val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 5: Nested loops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @nested_loops
// CHECK-NOT: Failed to allocate
waveasm.program @nested_loops
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %outer_init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %outer_init_sum = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK: waveasm.loop
  %oi_out, %osum_out = waveasm.loop(%oi = %outer_init_i, %osum = %outer_init_sum)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %inner_init_j = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

    // CHECK: waveasm.loop
    %ij_out, %inner_result = waveasm.loop(%ij = %inner_init_j, %inner_acc = %osum)
        : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
      %inner_tmp = waveasm.v_add_u32 %inner_acc, %v0
          : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
      %next_j = waveasm.s_add_u32 %ij, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
      %inner_cond = waveasm.s_cmp_lt_u32 %next_j, %c4 : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
      waveasm.condition %inner_cond : !waveasm.sreg iter_args(%next_j, %inner_tmp) : !waveasm.sreg, !waveasm.vreg
    }

    %outer_new_sum = waveasm.v_add_u32 %inner_result, %c1
        : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %next_i = waveasm.s_add_u32 %oi, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %outer_cond = waveasm.s_cmp_lt_u32 %next_i, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %outer_cond : !waveasm.sreg iter_args(%next_i, %outer_new_sum) : !waveasm.sreg, !waveasm.vreg
  }

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 6: 8 MFMA chains in loop with ds_read data
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @multi_mfma_chain_loop
// CHECK-NOT: Failed to allocate
waveasm.program @multi_mfma_chain_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %z0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z4 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z5 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z6 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %z7 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7 = waveasm.loop(
      %i = %init_i,
      %acc0 = %z0, %acc1 = %z1, %acc2 = %z2, %acc3 = %z3,
      %acc4 = %z4, %acc5 = %z5, %acc6 = %z6, %acc7 = %z7)
      : (!waveasm.sreg,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
         !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %lds_a = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>
    %lds_b = waveasm.ds_read_b128 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

    %n0 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc0 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc1 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n2 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc2 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n3 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc3 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n4 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc4 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n5 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc5 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n6 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc6 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %n7 = waveasm.v_mfma_f32_16x16x16_f16 %lds_a, %lds_b, %acc7 : !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c16 : !waveasm.sreg, !waveasm.imm<16> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7)
        : !waveasm.sreg,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>,
          !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  %use0 = waveasm.v_add_u32 %r0, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %use7 = waveasm.v_add_u32 %r7, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 7: MFMA result reuse as generic source
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @mfma_result_reuse
// CHECK-NOT: Failed to allocate
waveasm.program @mfma_result_reuse
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %v8 = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>

  // CHECK: -> !waveasm.pvreg<[[ACC:[0-9]+]], 4>
  %acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  // CHECK: -> !waveasm.pvreg<[[ACC]], 4>
  %acc1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[ACC]], 4>
  %readback = waveasm.v_add_u32 %acc1, %v8
      : !waveasm.vreg<4, 4>, !waveasm.pvreg<8> -> !waveasm.vreg
  // CHECK: -> !waveasm.pvreg<[[ACC]], 4>
  %acc2 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc1
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

  %use = waveasm.v_add_u32 %readback, %v8 : !waveasm.vreg, !waveasm.pvreg<8> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 8: Block arg used only as MFMA accumulator (late use)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @blockarg_only_mfma_use
// CHECK-NOT: Failed to allocate
waveasm.program @blockarg_only_mfma_use
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: -> !waveasm.pvreg<[[BA_ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // Pressure-creating work before MFMA
    %t1 = waveasm.v_add_u32 %a, %c1 : !waveasm.pvreg<0, 4>, !waveasm.imm<1> -> !waveasm.vreg
    %t2 = waveasm.v_add_u32 %t1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %t3 = waveasm.v_add_u32 %t2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %t4 = waveasm.v_add_u32 %t3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %t5 = waveasm.v_add_u32 %t4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

    // MFMA at end -- block arg must still be alive
    // CHECK: v_mfma_f32_16x16x16_f16 {{.*}} -> !waveasm.pvreg<[[BA_ACC]], 4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  %v8 = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[BA_ACC]], 4>
  %final = waveasm.v_add_u32 %acc_out, %v8 : !waveasm.vreg<4, 4>, !waveasm.pvreg<8> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 9: Two sequential loops -- register reuse
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @sequential_loops
// CHECK-NOT: Failed to allocate
waveasm.program @sequential_loops
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %a = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %b = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>

  // First loop
  %init_i1 = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %i1_out, %acc1_out = waveasm.loop(%i1 = %init_i1, %loop1_acc = %init_acc1)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {
    %n1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %loop1_acc
        : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %next_i1 = waveasm.s_add_u32 %i1, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond1 = waveasm.s_cmp_lt_u32 %next_i1, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond1 : !waveasm.sreg iter_args(%next_i1, %n1) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  %use1 = waveasm.v_add_u32 %acc1_out, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Second loop -- should reuse registers
  %init_i2 = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %i2_out, %acc2_out = waveasm.loop(%i2 = %init_i2, %loop2_acc = %init_acc2)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {
    %n2 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %loop2_acc
        : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %next_i2 = waveasm.s_add_u32 %i2, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond2 = waveasm.s_cmp_lt_u32 %next_i2, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond2 : !waveasm.sreg iter_args(%next_i2, %n2) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  %use2 = waveasm.v_add_u32 %acc2_out, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 10: Unused block arg forwarded through condition
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @blockarg_cond_only
// CHECK-NOT: Failed to allocate
waveasm.program @blockarg_cond_only
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  // CHECK: -> !waveasm.pvreg<[[FWD:[0-9]+]]>
  %init_sum = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %i_out, %val_out, %sum_out = waveasm.loop(%i = %init_i, %val = %init_val, %sum = %init_sum)
      : (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg, !waveasm.vreg) {

    %new_val = waveasm.v_add_u32 %val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %new_val, %sum) : !waveasm.sreg, !waveasm.vreg, !waveasm.vreg
  }

  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[FWD]]>
  %final = waveasm.v_add_u32 %sum_out, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 11: Long epilogue -- loop result survives
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @long_epilogue
// CHECK-NOT: Failed to allocate
waveasm.program @long_epilogue
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: -> !waveasm.pvreg<[[EPIL_ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>) -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {
    %la = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
    %lb = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>
    %new_acc = waveasm.v_mfma_f32_16x16x16_f16 %la, %lb, %acc
        : !waveasm.pvreg<4, 4>, !waveasm.pvreg<8, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c8 : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i, %new_acc) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  // Long epilogue chain
  %e1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg
  %e2 = waveasm.v_add_u32 %e1, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e3 = waveasm.v_add_u32 %e2, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e4 = waveasm.v_add_u32 %e3, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e5 = waveasm.v_add_u32 %e4, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e6 = waveasm.v_add_u32 %e5, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e7 = waveasm.v_add_u32 %e6, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
  %e8 = waveasm.v_add_u32 %e7, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  // Loop result used AFTER long epilogue -- must not be clobbered
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[EPIL_ACC]], 4>
  %final = waveasm.v_add_u32 %acc_out, %v0 : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 12: Large 16-register MFMA with fragmented precoloring
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @large_alignment
// CHECK-NOT: Failed to allocate
waveasm.program @large_alignment
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %v8 = waveasm.precolored.vreg 8, 4 : !waveasm.pvreg<8, 4>
  %v12 = waveasm.precolored.vreg 12, 4 : !waveasm.pvreg<12, 4>

  // CHECK: -> !waveasm.pvreg<[[BIG:[0-9]+]], 16>
  %big_vec = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<16>
  // CHECK: -> !waveasm.pvreg<[[BIG]], 16>
  %mfma_result = waveasm.v_mfma_f32_32x32x8_f16 %v0, %v4, %big_vec
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<16> -> !waveasm.vreg<16>

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 13: AGPR allocation smoke test
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @agpr_pressure_smoke
// CHECK-NOT: Failed to allocate
waveasm.program @agpr_pressure_smoke
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %acc0 = waveasm.precolored.areg 16, 4 : !waveasm.pareg<16, 4>
  %acc1 = waveasm.precolored.areg 20, 4 : !waveasm.pareg<20, 4>

  // CHECK: -> !waveasm.pareg<16, 4>
  %r0 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc0
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.pareg<16, 4> -> !waveasm.areg<4, 4>
  // CHECK: -> !waveasm.pareg<20, 4>
  %r1 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc1
      : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.pareg<20, 4> -> !waveasm.areg<4, 4>
  waveasm.s_endpgm
}
