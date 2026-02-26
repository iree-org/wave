// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Assembly emission tests for async memory iter_arg tying.
//
// Verifies that the emitter produces correct back-edge copies when iter_args
// and block args are NOT tied (unsafe async memory ops), and omits copies
// when they ARE tied (safe ordering or synchronous ops).

//===----------------------------------------------------------------------===//
// Test 1: UNSAFE buffer_load — emitter must produce 4 × v_mov_b32 copies
// to move the untied iter_arg into the block arg's register at the back edge.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: vmem_emit_unsafe:
waveasm.program @vmem_emit_unsafe
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_data = waveasm.buffer_load_dwordx4 %srd, %c0, %c0 : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: L_loop_0:
  // CHECK: buffer_load_dwordx4
  // CHECK: v_mfma_f32_16x16x16_f16
  // Back-edge copies: 4 individual v_mov_b32 for the wide register.
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: s_cbranch_scc1 L_loop_0
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %data_next = waveasm.buffer_load_dwordx4 %srd, %c0, %c0
        : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new, %data_next)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: SAFE buffer_load — block arg dead before load, no copies needed.
// The buffer_load writes directly into the block arg's register.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: vmem_emit_safe:
waveasm.program @vmem_emit_safe
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_data = waveasm.buffer_load_dwordx4 %srd, %c0, %c0 : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: L_loop_0:
  // CHECK: v_mfma_f32_16x16x16_f16
  // CHECK: buffer_load_dwordx4
  // No v_mov copies — tied registers, load writes to block arg directly.
  // CHECK-NOT: v_mov_b32
  // CHECK: s_cbranch_scc1 L_loop_0
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>
    %data_next = waveasm.buffer_load_dwordx4 %srd, %c0, %c0
        : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new, %data_next)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 3: UNSAFE ds_read_b128 — same back-edge copy pattern via LDS.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: lds_emit_unsafe:
waveasm.program @lds_emit_unsafe
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %lds_addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  %init_data = waveasm.ds_read_b128 %lds_addr : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  // CHECK: L_loop_0:
  // CHECK: ds_read_b128
  // CHECK: v_mfma_f32_16x16x16_f16
  // Back-edge copies: 4 individual v_mov_b32 for the wide register.
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: v_mov_b32
  // CHECK: s_cbranch_scc1 L_loop_0
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    %data_next = waveasm.ds_read_b128 %lds_addr : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new, %data_next)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}
