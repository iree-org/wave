// RUN: waveasm-translate --waveasm-scoped-cse %s | FileCheck %s
//
// Test: Scoped common subexpression elimination

// CHECK-LABEL: waveasm.program @cse_duplicate_constants
waveasm.program @cse_duplicate_constants target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Two identical constants should be deduplicated
  // CHECK: waveasm.constant 42
  // CHECK-NOT: waveasm.constant 42
  %c1 = waveasm.constant 42 : !waveasm.imm<42>
  %c2 = waveasm.constant 42 : !waveasm.imm<42>

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %r1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<42> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %v0, %c2 : !waveasm.pvreg<0>, !waveasm.imm<42> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @cse_duplicate_ops
waveasm.program @cse_duplicate_ops target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c1 = waveasm.constant 10 : !waveasm.imm<10>

  // Two identical v_add_u32 operations should be deduplicated
  // CHECK: waveasm.v_add_u32
  // CHECK-NOT: waveasm.v_add_u32
  %r1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<10> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<10> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @cse_different_values_kept
waveasm.program @cse_different_values_kept target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Different constants should NOT be eliminated
  // CHECK: waveasm.constant 100
  // CHECK: waveasm.constant 200
  %c1 = waveasm.constant 100 : !waveasm.imm<100>
  %c2 = waveasm.constant 200 : !waveasm.imm<200>

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %r1 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<100> -> !waveasm.vreg
  %r2 = waveasm.v_add_u32 %v0, %c2 : !waveasm.pvreg<0>, !waveasm.imm<200> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @cse_memory_not_eliminated
waveasm.program @cse_memory_not_eliminated target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Two identical memory loads should NOT be eliminated (side effects)
  // CHECK: waveasm.buffer_load_dword
  // CHECK: waveasm.buffer_load_dword
  %load1 = waveasm.buffer_load_dword %srd, %voff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  %load2 = waveasm.buffer_load_dword %srd, %voff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @cse_precolored_not_eliminated
waveasm.program @cse_precolored_not_eliminated target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Precolored registers should NOT be eliminated (they're fixed bindings)
  // CHECK: waveasm.precolored.vreg 0
  // CHECK: waveasm.precolored.vreg 0
  %v0a = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v0b = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  waveasm.s_endpgm
}
