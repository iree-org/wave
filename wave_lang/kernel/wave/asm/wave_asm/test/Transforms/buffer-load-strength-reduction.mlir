// RUN: waveasm-translate --waveasm-buffer-load-strength-reduction %s 2>&1 | FileCheck %s
//
// Tests for the BufferLoadStrengthReduction pass.
//
// The pass finds buffer_load ops inside LoopOps whose voffset depends on the
// induction variable, precomputes the voffset at iv=init, and replaces
// per-iteration address recomputation with a single s_add_u32 soffset bump.

// ---- Basic: one buffer_load with IV-dependent voffset ----
// The voffset chain: v_lshlrev_b32(iv, 4) computes byte offset from IV.
// After strength reduction:
//   - voffset is precomputed before the loop (at iv=0).
//   - soffset iter_arg starts at 0 and increments by stride each iteration.
//   - buffer_load uses the precomputed voffset + soffset iter_arg.

// CHECK-LABEL: @basic_strength_reduction
waveasm.program @basic_strength_reduction
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // The new loop should have an extra iter_arg for soffset.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg) {
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // Compute voffset = (tid + iv) << 4.
    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    // Soffset is a loop-carried sreg, bumped each iteration, fed back via condition.
    // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
    // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
    // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_SOFF]]) :
    %val = waveasm.buffer_load_dword %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- No transformation: voffset does not depend on IV ----
// The buffer_load voffset is just %tid (loop-invariant), so the pass
// should leave it untouched.

// CHECK-LABEL: @no_transform_loop_invariant_voffset
waveasm.program @no_transform_loop_invariant_voffset
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Loop should keep exactly 2 iter_args (no extra soffset added).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg) {
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // voffset is just %tid — no IV dependency.
    // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}}, %{{.*}} : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> ->
    %val = waveasm.buffer_load_dword %srd, %tid, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Multiple SRD groups: two buffer_loads with different SRDs ----
// Should create one soffset iter_arg per SRD group.

// CHECK-LABEL: @two_srd_groups
waveasm.program @two_srd_groups
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd_a = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %srd_b = waveasm.precolored.sreg 4, 4 : !waveasm.psreg<4, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Two SRD groups -> 2 extra soffset iter_args, independently carried.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF_A:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF_B:%[a-z0-9]+]] : !waveasm.psreg<4, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_A:%[0-9]+]] = waveasm.s_add_u32 [[SOFF_A]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: [[NEXT_B:%[0-9]+]] = waveasm.s_add_u32 [[SOFF_B]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_A]], [[NEXT_B]]) :
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    %val_a = waveasm.buffer_load_dword %srd_a, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg
    %val_b = waveasm.buffer_load_dword %srd_b, %voff, %soff0
        : !waveasm.psreg<4, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %sum = waveasm.v_add_u32 %val_a, %val_b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %new_acc = waveasm.v_add_u32 %acc, %sum : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Same SRD, same stride: two buffer_loads sharing one SRD ----
// Both voffsets use the same shift (<<4) but different constant addends,
// so the stride is identical. Should create only one soffset iter_arg.

// CHECK-LABEL: @shared_srd_group
waveasm.program @shared_srd_group
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %sixteen = waveasm.constant 16 : !waveasm.imm<16>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // One SRD group (same stride=16) -> 1 extra soffset iter_arg.
  // Both loads share the same loop-carried soffset.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_SOFF]]) :
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // Both use <<4 (same stride), but voff_b has a +16 offset.
    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff_a = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg
    %shifted = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg
    %voff_b = waveasm.v_add_u32 %shifted, %sixteen : !waveasm.vreg, !waveasm.imm<16> -> !waveasm.vreg

    // Both loads share the same SRD and stride -> same soffset.
    %val_a = waveasm.buffer_load_dword %srd, %voff_a, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg
    %val_b = waveasm.buffer_load_dword %srd, %voff_b, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %sum = waveasm.v_add_u32 %val_a, %val_b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %new_acc = waveasm.v_add_u32 %acc, %sum : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Soffset increment: verify s_add_u32 for soffset bumping ----
// After transformation, the loop body should contain an s_add_u32 that
// increments the soffset by the stride.

// CHECK-LABEL: @soffset_increment
waveasm.program @soffset_increment
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    %val = waveasm.buffer_load_dword %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    // Trace the loop-carried soffset: load uses it, s_add bumps it, condition feeds it back.
    // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
    // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
    // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_SOFF]]) :
    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Stride materialization: s_mov_b32 with constant stride before loop ----
// The pass computes stride symbolically (here: ivStep=1, shift=4, so stride=16)
// and emits s_mov_b32 with the constant before the loop.

// CHECK-LABEL: @stride_precompute
waveasm.program @stride_precompute
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Before the loop: stride=16 materialized as constant + s_mov_b32.
  // CHECK: waveasm.constant 16
  // CHECK: [[STRIDE:%[0-9]+]] = waveasm.s_mov_b32 {{.*}} : !waveasm.imm<16> -> !waveasm.sreg
  // CHECK: waveasm.loop
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], [[STRIDE]] : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, [[NEXT_SOFF]]) :
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    %val = waveasm.buffer_load_dword %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- No loop: buffer_load outside a loop is not touched ----
// CHECK-LABEL: @no_loop
waveasm.program @no_loop
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %soff0 = waveasm.constant 0 : !waveasm.imm<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Should remain unchanged — no loop to optimize.
  // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}}, %{{.*}} : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> ->
  %val = waveasm.buffer_load_dword %srd, %voff, %soff0
      : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// ---- Non-uniform stride: v_mul_lo_u32(tid, iv) is skipped ----
// voffset = tid * iv has stride = tid * step, which is thread-dependent.
// computeStaticStride cannot evaluate getConstantValue(tid), so the
// candidate is rejected. The loop should remain unchanged.

// CHECK-LABEL: @non_uniform_stride
waveasm.program @non_uniform_stride
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Loop unchanged: still 2 iter_args (no soffset added).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg) {
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // voffset = tid * iv — stride is tid-dependent (non-uniform).
    %voff = waveasm.v_mul_lo_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg

    %val = waveasm.buffer_load_dword %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Multi-dword load (buffer_load_dwordx4) ----
// Verify the pass handles multi-result loads correctly.

// CHECK-LABEL: @multi_dword_load
waveasm.program @multi_dword_load
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Should add 1 soffset iter_arg (3 total).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_SOFF]]) :
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg
    %v0, %v1, %v2, %v3 = waveasm.buffer_load_dwordx4 %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0>
        -> !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %v0 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- Non-zero instOffset preserved ----
// buffer_load with instOffset should still be transformed; the offset field
// is orthogonal to the soffset/voffset split.

// CHECK-LABEL: @nonzero_inst_offset
waveasm.program @nonzero_inst_offset
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Should still transform — instOffset is independent of soffset.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] offset : 2048 : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, [[NEXT_SOFF]]) :
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg
    %val = waveasm.buffer_load_dword %srd, %voff, %soff0 offset : 2048
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Different strides on same SRD get separate soffset iter_args ----
// Two loads share the same SRD but have different voffset chains (<<2 vs <<4),
// producing different strides. Each should get its own soffset group.

// CHECK-LABEL: @different_strides_same_srd
waveasm.program @different_strides_same_srd
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %two = waveasm.constant 2 : !waveasm.imm<2>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Same SRD, different strides -> 2 independent loop-carried soffsets.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF_A:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF_B:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_A:%[0-9]+]] = waveasm.s_add_u32 [[SOFF_A]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: [[NEXT_B:%[0-9]+]] = waveasm.s_add_u32 [[SOFF_B]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, [[NEXT_A]], [[NEXT_B]]) :
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // voff_a = (tid + iv) << 2  (stride = 4 per IV step).
    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff_a = waveasm.v_lshlrev_b32 %two, %addr : !waveasm.imm<2>, !waveasm.vreg -> !waveasm.vreg

    // voff_b = (tid + iv) << 4  (stride = 16 per IV step).
    %voff_b = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    %val_a = waveasm.buffer_load_dword %srd, %voff_a, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg
    %val_b = waveasm.buffer_load_dword %srd, %voff_b, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %sum = waveasm.v_add_u32 %val_a, %val_b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
    %new_acc = waveasm.v_add_u32 %acc, %sum : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}

// ---- LDS load: buffer_load_dwordx4_lds with IV-dependent voffset ----
// Gather-to-LDS loads have operand order (voffset, srd, soffset) and no VGPR
// results. The pass should still optimize the IV-dependent voffset chain.

// CHECK-LABEL: @lds_load_strength_reduction
waveasm.program @lds_load_strength_reduction
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // New loop should have an extra iter_arg for soffset.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.sreg) {
  // m0 write preserved, soffset is loop-carried sreg.
  // CHECK: waveasm.s_mov_b32_m0
  // CHECK: waveasm.buffer_load_dwordx4_lds {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.sreg
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, [[NEXT_SOFF]]) :
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    // s_mov_b32_m0 sets LDS offset — should be preserved untouched.
    %m0_val = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
    waveasm.s_mov_b32_m0 %m0_val : !waveasm.sreg
    waveasm.buffer_load_dwordx4_lds %voff, %srd, %soff0
        : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.imm<0>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- LDS load: loop-invariant voffset — no transformation ----
// The buffer_load_dword_lds voffset is just %tid (no IV dependency), so the
// pass should leave the loop untouched.

// CHECK-LABEL: @lds_load_no_transform
waveasm.program @lds_load_no_transform
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Loop unchanged: still 1 iter_arg (no soffset added).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> !waveasm.sreg {
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    // voffset is just %tid — no IV dependency. Soffset stays imm<0>.
    // CHECK: waveasm.buffer_load_dword_lds {{.*}}, {{.*}}, {{.*}} : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>
    waveasm.buffer_load_dword_lds %tid, %srd, %soff0
        : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Right-shift in address chain: lshl then lshr ----
// Pattern from MXFP4 preshuffle GEMM: address chain does lshl 7 then lshr 8.
// With IV step=2: delta through lshl = 2*128 = 256, lshr 8 = 256/256 = 1.
// The stride is constant because the delta is exactly divisible by the shift.
// Previously the pass bailed on lshrrev as "nonlinear if IV-dependent".

// CHECK-LABEL: @lshrrev_in_address_chain
waveasm.program @lshrrev_in_address_chain
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %two = waveasm.constant 2 : !waveasm.imm<2>
  %seven = waveasm.constant 7 : !waveasm.imm<7>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %twelve = waveasm.constant 12 : !waveasm.imm<12>
  %limit = waveasm.constant 32 : !waveasm.imm<32>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg

  // Should be transformed despite the lshrrev in the chain.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]] : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.sreg ->
  // CHECK: [[NEXT_SOFF:%[0-9]+]] = waveasm.s_add_u32 [[SOFF]], {{.*}} : !waveasm.sreg, !waveasm.sreg ->
  // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, [[NEXT_SOFF]]) :
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    // Mimic the GEMM pattern: addr = ((IV << 7) + tid) >> 8, then shift up.
    %shifted_iv = waveasm.v_lshlrev_b32 %seven, %iv : !waveasm.imm<7>, !waveasm.sreg -> !waveasm.vreg
    %addr = waveasm.v_add_u32 %shifted_iv, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
    %page = waveasm.v_lshrrev_b32 %eight, %addr : !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %twelve, %page : !waveasm.imm<12>, !waveasm.vreg -> !waveasm.vreg

    %v0, %v1, %v2, %v3 = waveasm.buffer_load_dwordx4 %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0>
        -> !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %two : !waveasm.sreg, !waveasm.imm<2> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<32> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.sreg
  }

  waveasm.s_endpgm
}

// ---- Right-shift with non-divisible delta: no transformation ----
// IV step=1, lshl 3 gives delta=8, lshr 4 = 8/16 = 0.5 — not an integer.
// The pass should reject this candidate.

// CHECK-LABEL: @lshrrev_non_divisible
waveasm.program @lshrrev_non_divisible
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %three = waveasm.constant 3 : !waveasm.imm<3>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // Loop unchanged: delta(lshl 3)=8, lshr 4 gives 8%16!=0, rejected.
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg) {
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // delta(shifted_iv) = 1*8 = 8, lshr 4 -> 8 % 16 != 0 -> bail.
    %shifted_iv = waveasm.v_lshlrev_b32 %three, %iv : !waveasm.imm<3>, !waveasm.sreg -> !waveasm.vreg
    %addr = waveasm.v_add_u32 %shifted_iv, %tid : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg
    %voff = waveasm.v_lshrrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    %val = waveasm.buffer_load_dword %srd, %voff, %soff0
        : !waveasm.psreg<0, 4>, !waveasm.vreg, !waveasm.imm<0> -> !waveasm.vreg

    %new_acc = waveasm.v_add_u32 %acc, %val : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

    %next_iv = waveasm.s_add_u32 %iv, %one : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_iv, %limit : !waveasm.sreg, !waveasm.imm<8> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv, %new_acc) : !waveasm.sreg, !waveasm.vreg
  }

  waveasm.s_endpgm
}
