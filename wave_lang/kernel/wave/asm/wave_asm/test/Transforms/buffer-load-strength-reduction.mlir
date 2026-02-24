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
  // CHECK-SAME: !waveasm.sreg
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    // Compute voffset = (tid + iv) << 4.
    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    // After strength reduction, this buffer_load should use a precomputed
    // voffset (defined before the loop) and an soffset iter_arg (not imm<0>).
    // CHECK: waveasm.buffer_load_dword
    // CHECK-NOT: !waveasm.imm<0> ->
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

  // Two SRD groups -> 2 extra soffset iter_args.
  // Original: 2 iter_args (iv, acc). After: 4 (iv, acc, soff_a, soff_b).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg, !waveasm.sreg) {
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
  // Original: 2 iter_args. After: 3 (iv, acc, soff).
  // Both buffer_loads must use the same soffset (3rd block arg).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg) {
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF:%[a-z0-9]+]]
  // CHECK: waveasm.buffer_load_dword {{.*}}, {{.*}}, [[SOFF]]
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

    // After transformation, the original v_mov_b32/v_add_u32/v_lshlrev_b32
    // VALU chain should still be cloned (dead code), but there should be an
    // s_add_u32 for soffset bumping inside the loop.
    // CHECK: waveasm.s_add_u32
    // CHECK: waveasm.s_add_u32
    // CHECK: waveasm.condition
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
  // CHECK: waveasm.s_mov_b32
  // CHECK: waveasm.loop
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
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    // CHECK: waveasm.buffer_load_dwordx4
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
  %final_iv = waveasm.loop(%iv = %init_iv) : (!waveasm.sreg) -> (!waveasm.sreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

    // instOffset:2048 must be preserved after transformation.
    // CHECK: waveasm.buffer_load_dword
    // CHECK-SAME: offset : 2048
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

  // Same SRD, different strides -> 2 soffset iter_args.
  // Original: 2 iter_args (iv, acc). After: 4 (iv, acc, soff_a, soff_b).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg, !waveasm.sreg) {
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
