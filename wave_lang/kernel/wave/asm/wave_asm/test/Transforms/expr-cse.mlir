// RUN: waveasm-translate --waveasm-scoped-cse %s 2>&1 | FileCheck %s
//
// Test expression CSE (Common Subexpression Elimination)
// The CSE pass should eliminate redundant arithmetic operations

// Test 1: Basic CSE - identical operations
// CHECK-LABEL: waveasm.program @basic_cse
waveasm.program @basic_cse target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // First add
  // CHECK: waveasm.v_add_u32
  %v0 = waveasm.v_add_u32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Second identical add - should be eliminated by CSE
  // CHECK-NOT: waveasm.v_add_u32{{.*}}waveasm.v_add_u32
  %v1 = waveasm.v_add_u32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Third different operation - should not be eliminated
  // CHECK: waveasm.v_mul_lo_u32
  %v2 = waveasm.v_mul_lo_u32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: CSE across operations - only arithmetic ops are CSE-able
// CHECK-LABEL: waveasm.program @cse_arithmetic_only
waveasm.program @cse_arithmetic_only target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %b = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Arithmetic ops are CSE-able (have ArithmeticOp trait)
  // CHECK: waveasm.v_and_b32
  %v0 = waveasm.v_and_b32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Same operation - should be CSE'd (no second v_and_b32 with same operands)
  %v1 = waveasm.v_and_b32 %a, %b : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Different operand order - not the same (AND is commutative but CSE uses operand order)
  // CHECK: waveasm.v_and_b32
  %v2 = waveasm.v_and_b32 %b, %a : !waveasm.pvreg<1>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 3: Memory ops should NOT be CSE'd (have MemoryOp trait)
// CHECK-LABEL: waveasm.program @no_cse_memory_ops
waveasm.program @no_cse_memory_ops target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // First load - should NOT be eliminated
  // CHECK: waveasm.buffer_load_dword
  %v0 = waveasm.buffer_load_dword %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Second identical load - should also remain (memory ops are not CSE-able)
  // CHECK: waveasm.buffer_load_dword
  %v1 = waveasm.buffer_load_dword %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 4: Control flow ops should NOT be CSE'd (have ControlFlowOp trait)
// CHECK-LABEL: waveasm.program @no_cse_control_flow
waveasm.program @no_cse_control_flow target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Waitcnt and barriers are not CSE-able
  // CHECK: waveasm.s_waitcnt_vmcnt 0
  waveasm.s_waitcnt_vmcnt 0
  // CHECK: waveasm.s_waitcnt_vmcnt 0
  waveasm.s_waitcnt_vmcnt 0

  // CHECK: waveasm.s_barrier
  waveasm.s_barrier
  // CHECK: waveasm.s_barrier
  waveasm.s_barrier

  waveasm.s_endpgm
}

// Test 5: SGPR arithmetic should also be CSE'd
// CHECK-LABEL: waveasm.program @cse_sgpr_arith
waveasm.program @cse_sgpr_arith target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %s0 = waveasm.precolored.sreg 0 : !waveasm.psreg<0>
  %s1 = waveasm.precolored.sreg 1 : !waveasm.psreg<1>

  // First SALU add
  // CHECK: waveasm.s_add_u32
  %r0 = waveasm.s_add_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg

  // Same operation - should be CSE'd
  %r1 = waveasm.s_add_u32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg

  // Different operation
  // CHECK: waveasm.s_mul_i32
  %r2 = waveasm.s_mul_i32 %s0, %s1 : !waveasm.psreg<0>, !waveasm.psreg<1> -> !waveasm.sreg

  waveasm.s_endpgm
}
