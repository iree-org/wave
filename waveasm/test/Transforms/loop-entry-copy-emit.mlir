// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: When a loop init arg is not coalesced with its block arg (because
// the init arg has post-loop uses), the emitter must insert a v_mov_b32
// copy before the loop label to initialize the block arg register.

// CHECK-LABEL: loop_entry_copy:
waveasm.program @loop_entry_copy
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Init value used both as loop init arg and post-loop.
  // CHECK: v_mov_b32 [[INIT:v[0-9]+]], 0
  %init_val = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %init_i = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // Entry copy: init register copied into a different block arg register.
  // CHECK: v_mov_b32 [[IV:v[0-9]+]], [[INIT]]
  // The loop label comes after the copy.
  // CHECK: L_loop_0:
  // The block arg register is used inside the loop, not [[INIT]].
  // CHECK: v_mul_lo_u32 [[IV]], [[IV]]
  %iv_out = waveasm.loop(%iv = %init_val) : (!waveasm.vreg) -> (!waveasm.vreg) {
    %next_iv = waveasm.v_mul_lo_u32 %iv, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg
    %cond_s = waveasm.v_readfirstlane_b32 %next_iv : !waveasm.vreg -> !waveasm.sreg
    %ub_s = waveasm.v_readfirstlane_b32 %init_i : !waveasm.vreg -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %cond_s, %ub_s : !waveasm.sreg, !waveasm.sreg -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg iter_args(%next_iv) : !waveasm.vreg
  }

  // Post-loop use still reads the original register.
  // CHECK: v_add_u32 {{.*}}, [[INIT]], v0
  %result = waveasm.v_add_u32 %init_val, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
