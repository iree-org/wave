// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test linear scan register allocation with multi-register ranges and alignment

// Test 1: Basic allocation - single registers
// CHECK-LABEL: waveasm.program @basic_alloc
waveasm.program @basic_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Precolored registers should keep their physical assignment
  // CHECK: waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // CHECK: waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Virtual register should be allocated to physical
  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: Multi-register allocation (4 consecutive VGPRs with 4-alignment)
// CHECK-LABEL: waveasm.program @multi_reg_alloc
waveasm.program @multi_reg_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Vector of 4 registers with 4-alignment - must start at v0, v4, v8, etc.
  // Since v0 is precolored, should allocate at v4
  // CHECK: -> !waveasm.pvreg<4, 4>
  %vec = waveasm.v_mov_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 3: SGPR multi-register allocation (2 consecutive SGPRs for 64-bit pointer)
// CHECK-LABEL: waveasm.program @sgpr_pair_alloc
waveasm.program @sgpr_pair_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Precolored SGPR pair (kernarg pointer at s[0:1])
  // CHECK: waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %karg = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>

  // Another SGPR pair - should be allocated after s[0:1]
  // The s_load returns an SGPR, testing allocation
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<{{[0-9]+}}, 2>
  %ptr = waveasm.s_load_dwordx2 %karg, %c0 : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}

// Test 4: MFMA accumulator (16 consecutive VGPRs)
// CHECK-LABEL: waveasm.program @mfma_acc_alloc
waveasm.program @mfma_acc_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>

  // MFMA with 16-register accumulator
  // CHECK: waveasm.v_mfma_f32_32x32x8_f16 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 16>
  %acc = waveasm.v_mfma_f32_32x32x8_f16 %v0, %v4, %c0 : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.imm<0> -> !waveasm.vreg<16>

  waveasm.s_endpgm
}

// Test 5: Multiple overlapping ranges
// CHECK-LABEL: waveasm.program @overlapping_ranges
waveasm.program @overlapping_ranges target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Create several values with overlapping lifetimes
  // CHECK: -> !waveasm.pvreg<
  %a = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  // CHECK: -> !waveasm.pvreg<
  %b = waveasm.v_mul_lo_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg
  // Both %a and %b are live here - need different registers
  // CHECK: -> !waveasm.pvreg<
  %c = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 6: Loop result allocation - result should get physical register via tying
// CHECK-LABEL: waveasm.program @loop_result_alloc
waveasm.program @loop_result_alloc target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[LR:[0-9]+]]>
  %init_sum = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // Loop result should be allocated to a physical register
  // CHECK: waveasm.loop
  %i_out, %sum_out = waveasm.loop(%i = %init_i, %sum = %init_sum)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {
    // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<[[LR]]>
    %new_sum = waveasm.v_add_u32 %sum, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i#0, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_i#0, %new_sum) : !waveasm.sreg, !waveasm.vreg
  }

  // Post-loop use: loop result should have same physical register as init/body.
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[LR]]>{{.*}} -> !waveasm.pvreg<
  %final = waveasm.v_add_u32 %sum_out, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 7: Init arg with post-loop uses must NOT be coalesced with block arg.
// If the init arg (constant 0) is coalesced with the loop IV block arg,
// the loop body increments the shared register and corrupts the post-loop
// use that expects the original value.
// CHECK-LABEL: waveasm.program @init_arg_post_loop_use
waveasm.program @init_arg_post_loop_use target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // This value is used as both loop init arg AND post-loop operand.
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[INIT:[0-9]+]]>
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  %init_i = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // The loop uses %init_val as init arg for the IV.
  // The block arg must get a different register than [[INIT]].
  // CHECK-NOT: waveasm.v_mul_lo_u32{{.*}}!waveasm.pvreg<[[INIT]]>
  %iv_out = waveasm.loop(%iv = %init_val) : (!waveasm.vreg) -> (!waveasm.vreg) {
    %next_iv = waveasm.v_mul_lo_u32 %iv, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %cond_s = waveasm.v_readfirstlane_b32 %next_iv : !waveasm.vreg -> !waveasm.sreg
    %ub_s = waveasm.v_readfirstlane_b32 %init_i : !waveasm.vreg -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %cond_s, %ub_s : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.vreg
  }

  // Post-loop use of %init_val. It must still hold 0, not the loop's final IV.
  // The register must differ from the IV's block arg register.
  // CHECK: waveasm.v_add_u32 {{.*}}!waveasm.pvreg<[[INIT]]>{{.*}} -> !waveasm.pvreg<
  %result = waveasm.v_add_u32 %init_val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
