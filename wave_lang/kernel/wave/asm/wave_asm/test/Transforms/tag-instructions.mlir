// RUN: waveasm-translate --waveasm-tag-instructions --print-debug-locs-inline %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Each op kind gets its own counter.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @per_kind_counters
waveasm.program @per_kind_counters target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>

  // Two adds should get v_add_u32_0 and v_add_u32_1.
  // CHECK: waveasm.v_add_u32 {{.*}} loc("v_add_u32_0")
  // CHECK: waveasm.v_add_u32 {{.*}} loc("v_add_u32_1")
  %a0 = waveasm.v_add_u32 %v0, %c4 : !waveasm.pvreg<0>, !waveasm.imm<4> -> !waveasm.vreg
  %a1 = waveasm.v_add_u32 %a0, %c4 : !waveasm.vreg, !waveasm.imm<4> -> !waveasm.vreg

  // A shift gets its own counter starting at 0.
  // CHECK: waveasm.v_lshlrev_b32 {{.*}} loc("v_lshlrev_b32_0")
  %s0 = waveasm.v_lshlrev_b32 %c4, %a1 : !waveasm.imm<4>, !waveasm.vreg -> !waveasm.vreg

  // Load and store have independent counters.
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} loc("buffer_load_dwordx4_0")
  // CHECK: waveasm.buffer_store_dword {{.*}} loc("buffer_store_dword_0")
  %ld = waveasm.buffer_load_dwordx4 %srd, %s0 : !waveasm.psreg<0, 4>, !waveasm.vreg -> !waveasm.vreg<4>
  waveasm.buffer_store_dword %a1, %srd, %s0 : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.vreg

  // CHECK: waveasm.s_endpgm loc("s_endpgm_0")
  waveasm.s_endpgm
}

//===----------------------------------------------------------------------===//
// Test: Counters reset per module, not per program.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: waveasm.program @second_program
waveasm.program @second_program target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>

  // Counter continues from first program (module-wide walk).
  // CHECK: waveasm.v_add_u32 {{.*}} loc("v_add_u32_2")
  %a0 = waveasm.v_add_u32 %v0, %c1 : !waveasm.pvreg<0>, !waveasm.imm<1> -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm loc("s_endpgm_1")
  waveasm.s_endpgm
}
