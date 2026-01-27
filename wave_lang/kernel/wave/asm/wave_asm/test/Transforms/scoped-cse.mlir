// RUN: waveasm-translate --waveasm-scoped-cse %s 2>&1 | FileCheck %s
//
// Test scoped CSE (common subexpression elimination)

// Test 1: Basic CSE - duplicate constants should be eliminated
// CHECK-LABEL: waveasm.program @basic_cse
waveasm.program @basic_cse target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Two identical constants - second should be eliminated
  // Only one constant 42 should remain after CSE
  %c1 = waveasm.constant 42 : !waveasm.imm<42>
  %c2 = waveasm.constant 42 : !waveasm.imm<42>

  // Use the constant (after CSE, both uses refer to same constant)
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // CHECK: waveasm.v_add_u32
  // Both adds become identical after constant CSE, so one is eliminated
  %r1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<42> -> !waveasm.vreg<1>
  %r2 = waveasm.v_add_u32 %v0, %c2 : !waveasm.pvreg<0>, !waveasm.imm<42> -> !waveasm.vreg<1>

  waveasm.s_endpgm
}

// Test 2: Different constants should NOT be eliminated
// CHECK-LABEL: waveasm.program @different_constants
waveasm.program @different_constants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Different constants - both should remain
  // CHECK: waveasm.constant 10
  // CHECK: waveasm.constant 20
  %c1 = waveasm.constant 10 : !waveasm.imm<10>
  %c2 = waveasm.constant 20 : !waveasm.imm<20>

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %r1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<10> -> !waveasm.vreg<1>
  %r2 = waveasm.v_add_u32 %v0, %c2 : !waveasm.pvreg<0>, !waveasm.imm<20> -> !waveasm.vreg<1>

  waveasm.s_endpgm
}

// Test 3: Multiple duplicate constants - should be deduplicated
// CHECK-LABEL: waveasm.program @multiple_duplicates
waveasm.program @multiple_duplicates target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Multiple zero constants - should be deduplicated
  %c0a = waveasm.constant 0 : !waveasm.imm<0>
  %c0b = waveasm.constant 0 : !waveasm.imm<0>
  %c0c = waveasm.constant 0 : !waveasm.imm<0>

  // Multiple 64 constants - should be deduplicated
  %c64a = waveasm.constant 64 : !waveasm.imm<64>
  %c64b = waveasm.constant 64 : !waveasm.imm<64>

  // The pass should report 4 removals (2 zeros + 1 sixty-four)
  waveasm.s_endpgm
}

// Test 4: Memory operations should NOT be CSE'd
// CHECK-LABEL: waveasm.program @no_mem_cse
waveasm.program @no_mem_cse target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Two identical loads - both should remain (memory ops not CSE'd)
  // CHECK: waveasm.buffer_load_dword
  // CHECK: waveasm.buffer_load_dword
  %load1 = waveasm.buffer_load_dword %srd, %voff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %load2 = waveasm.buffer_load_dword %srd, %voff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 5: Precolored registers should NOT be CSE'd
// CHECK-LABEL: waveasm.program @no_precolored_cse
waveasm.program @no_precolored_cse target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Two identical precolored regs - both should remain
  // CHECK: waveasm.precolored.vreg 0
  // CHECK: waveasm.precolored.vreg 0
  %v0a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v0b = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  waveasm.s_endpgm
}
