// RUN: waveasm-translate %s 2>&1 | FileCheck %s
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

// ---- Same SRD group: two buffer_loads sharing one SRD ----
// Should create only one soffset iter_arg for the shared SRD group.

// CHECK-LABEL: @shared_srd_group
waveasm.program @shared_srd_group
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %zero = waveasm.constant 0 : !waveasm.imm<0>
  %one = waveasm.constant 1 : !waveasm.imm<1>
  %four = waveasm.constant 4 : !waveasm.imm<4>
  %eight = waveasm.constant 8 : !waveasm.imm<8>
  %limit = waveasm.constant 8 : !waveasm.imm<8>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %init_iv = waveasm.s_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %zero : !waveasm.imm<0> -> !waveasm.vreg

  // One SRD group -> 1 extra soffset iter_arg.
  // Original: 2 iter_args. After: 3 (iv, acc, soff).
  // CHECK: waveasm.loop
  // CHECK-SAME: -> (!waveasm.sreg, !waveasm.vreg, !waveasm.sreg) {
  %final_iv, %final_acc = waveasm.loop(%iv = %init_iv, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg) -> (!waveasm.sreg, !waveasm.vreg) {

    %addr = waveasm.v_add_u32 %tid, %iv : !waveasm.pvreg<0>, !waveasm.sreg -> !waveasm.vreg
    %voff_a = waveasm.v_lshlrev_b32 %four, %addr : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg
    %voff_b = waveasm.v_lshlrev_b32 %eight, %addr : !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg

    // Both loads share the same SRD -> same soffset.
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

// ---- Stride precomputation: v_sub and v_readfirstlane before loop ----
// The pass computes stride = voff(iv+step) - voff(iv) and converts to SGPR
// via v_readfirstlane_b32 before the loop.

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

  // Before the loop: stride computation via v_sub_u32 + v_readfirstlane_b32.
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_readfirstlane_b32
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
