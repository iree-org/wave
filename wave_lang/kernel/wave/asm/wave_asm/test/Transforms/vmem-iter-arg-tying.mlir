// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Tests for async memory iter_arg tying in the register allocator.
//
// In software-pipelined loops, buffer_load/ds_read results are passed as
// condition iter_args. Tying them to block args (sharing a physical register)
// is unsafe when the block arg is still live after the async memory op writes
// — the VMEM/LDS write would clobber a value MFMAs are still reading.
//
// NOTE: The type mismatch between untied iter_args and block args causes the
// LoopOp printer to fall back to generic MLIR format for the module.
// CHECK patterns below use generic format accordingly.

//===----------------------------------------------------------------------===//
// Test 1: UNSAFE — block arg used by MFMA after buffer_load writes iter_arg.
//
// Schedule (simplified double-buffer):
//   %data_next = buffer_load ...       ← async VMEM write to iter_arg
//   %acc_new   = mfma %data, ..., %acc ← still reading block arg %data
//
// If %data_next and %data share a register, the buffer_load clobbers %data
// while MFMA is reading it. They MUST get different physical registers.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sym_name = "vmem_tie_unsafe"
waveasm.program @vmem_tie_unsafe
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
  // Prefetch: first iteration's data loaded before the loop.
  // CHECK: buffer_load_dwordx4{{.*}}pvreg<[[INIT_DATA:[0-9]+]], 4>
  %init_data = waveasm.buffer_load_dwordx4 %srd, %c0, %c0 : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // Block arg %data should be tied to %init_data (same register).
  // CHECK: bb0{{.*}}pvreg<[[INIT_DATA]], 4>
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    // Inner buffer_load: the iter_arg for next iteration's data.
    // MUST get a DIFFERENT register from block arg %data (INIT_DATA).
    // CHECK: buffer_load_dwordx4{{.*}}pvreg<[[NEXT_DATA:[0-9]+]], 4>
    %data_next = waveasm.buffer_load_dwordx4 %srd, %c0, %c0
        : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

    // MFMA reads block arg %data AFTER buffer_load writes — the hazard.
    // Block arg %data should still be at INIT_DATA.
    // CHECK: v_mfma_f32_16x16x16_f16{{.*}}pvreg<[[INIT_DATA]], 4>
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    // Condition iter_arg for %data should be at NEXT_DATA (untied).
    // CHECK: waveasm.condition{{.*}}pvreg<[[NEXT_DATA]], 4>
    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new, %data_next)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 2: SAFE — block arg is NOT used after buffer_load.
//
// Schedule:
//   %acc_new   = mfma %data, ..., %acc ← reads block arg %data
//   %data_next = buffer_load ...       ← VMEM write AFTER all %data uses
//
// Since %data is dead before the buffer_load writes, tying is safe.
// They SHOULD share a physical register (no back-edge v_mov needed).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sym_name = "vmem_tie_safe"
waveasm.program @vmem_tie_safe
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  // CHECK: buffer_load_dwordx4{{.*}}pvreg<[[SAFE_INIT:[0-9]+]], 4>
  %init_data = waveasm.buffer_load_dwordx4 %srd, %c0, %c0 : !waveasm.psreg<0, 4>, !waveasm.imm<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // CHECK: waveasm.loop
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    // MFMA reads block arg %data FIRST — %data dies here.
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    // buffer_load writes AFTER all %data uses — safe to tie.
    // Iter_arg should share block arg's register (SAFE_INIT).
    // CHECK: buffer_load_dwordx4{{.*}}pvreg<[[SAFE_INIT]], 4>
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
// Test 3: Non-VMEM iter_arg — always safe to tie.
//
// MFMA results are synchronous VALU ops. Tying iter_arg to block arg
// is always correct because the MFMA writes only after reading all inputs.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sym_name = "non_vmem_iter_arg"
waveasm.program @non_vmem_iter_arg
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %a = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  // CHECK: v_mov_b32{{.*}}pvreg<[[ACC:[0-9]+]], 4>
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  %ri, %racc = waveasm.loop(%i = %init_i, %acc = %init_acc)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>) {

    // MFMA result is iter_arg — synchronous, always safe to tie.
    // CHECK: v_mfma_f32_16x16x16_f16{{.*}}pvreg<[[ACC]], 4>
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new) : !waveasm.sreg, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test 4: UNSAFE — ds_read_b128 (LDS load), same hazard as Test 1.
//
// The MemoryOp trait covers all async memory ops including LDS reads.
// Same pattern: the ds_read writes the iter_arg while MFMA still reads
// the block arg. They MUST get different physical registers.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: sym_name = "lds_tie_unsafe"
waveasm.program @lds_tie_unsafe
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c10 = waveasm.constant 10 : !waveasm.imm<10>
  %b = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %lds_addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %init_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
  // Prefetch: first LDS read before the loop.
  // CHECK: ds_read_b128{{.*}}pvreg<[[LDS_INIT:[0-9]+]], 4>
  %init_data = waveasm.ds_read_b128 %lds_addr : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  // Block arg should be tied to init_data (same register).
  // CHECK: bb0{{.*}}pvreg<[[LDS_INIT]], 4>
  %ri, %racc, %rdata = waveasm.loop(
      %i = %init_i, %acc = %init_acc, %data = %init_data)
      : (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>)
      -> (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>) {

    // Inner ds_read: the iter_arg for next iteration's data.
    // MUST get a DIFFERENT register from block arg %data (LDS_INIT).
    // CHECK: ds_read_b128{{.*}}pvreg<[[LDS_NEXT:[0-9]+]], 4>
    %data_next = waveasm.ds_read_b128 %lds_addr : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

    // MFMA reads block arg %data AFTER ds_read writes — the hazard.
    // CHECK: v_mfma_f32_16x16x16_f16{{.*}}pvreg<[[LDS_INIT]], 4>
    %acc_new = waveasm.v_mfma_f32_16x16x16_f16 %data, %b, %acc
        : !waveasm.vreg<4, 4>, !waveasm.pvreg<4, 4>, !waveasm.vreg<4, 4> -> !waveasm.vreg<4, 4>

    // Condition iter_arg for %data should be at LDS_NEXT (untied).
    // CHECK: waveasm.condition{{.*}}pvreg<[[LDS_NEXT]], 4>
    %next_i = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg
    %cond = waveasm.s_cmp_lt_u32 %next_i, %c10 : !waveasm.sreg, !waveasm.imm<10> -> !waveasm.sreg
    waveasm.condition %cond : !waveasm.sreg
        iter_args(%next_i, %acc_new, %data_next)
        : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.vreg<4, 4>
  }

  waveasm.s_endpgm
}
