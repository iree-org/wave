// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s | FileCheck %s
//
// Test: v_permlane16_swap_b32 → v_cndmask_b32 → buffer_store_dwordx4
// assembly sequence for coalesced bf16 wide stores.
//
// The WaveASM epilogue for wide bf16 stores works as follows:
//   1. v_permlane16_swap_b32 exchanges packed bf16 data between partner
//      lanes (16 apart) so each lane sees the other's data.
//   2. v_cndmask_b32 selects between own and partner data based on VCC
//      (set by a prior lane-position comparison).
//   3. The 4 selected registers are packed and written via
//      buffer_store_dwordx4 (128-bit store).

// CHECK-LABEL: permlane_cndmask_dwordx4_store:
waveasm.program @permlane_cndmask_dwordx4_store
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %v2 = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>
  %v3 = waveasm.precolored.vreg 3 : !waveasm.pvreg<3>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 4 : !waveasm.pvreg<4>

  %vcc = waveasm.precolored.sreg 106, 2 : !waveasm.psreg<106, 2>
  %lane_id = waveasm.precolored.vreg 5 : !waveasm.pvreg<5>
  %c16 = waveasm.constant 16 : !waveasm.imm<16>

  // CHECK: v_cmp_lt_u32 vcc, v5, 16
  waveasm.v_cmp_lt_u32 %lane_id, %c16 : !waveasm.pvreg<5>, !waveasm.imm<16>

  // CHECK: v_permlane16_swap_b32 v{{[0-9]+}}, v0
  %swap0 = waveasm.v_permlane16_swap_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg
  // CHECK: v_permlane16_swap_b32 v{{[0-9]+}}, v1
  %swap1 = waveasm.v_permlane16_swap_b32 %v1 : !waveasm.pvreg<1> -> !waveasm.vreg
  // CHECK: v_permlane16_swap_b32 v{{[0-9]+}}, v2
  %swap2 = waveasm.v_permlane16_swap_b32 %v2 : !waveasm.pvreg<2> -> !waveasm.vreg
  // CHECK: v_permlane16_swap_b32 v{{[0-9]+}}, v3
  %swap3 = waveasm.v_permlane16_swap_b32 %v3 : !waveasm.pvreg<3> -> !waveasm.vreg

  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v{{[0-9]+}}, v0
  %sel0 = waveasm.v_cndmask_b32 %swap0, %v0, %vcc
      : !waveasm.vreg, !waveasm.pvreg<0>, !waveasm.psreg<106, 2> -> !waveasm.vreg
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v{{[0-9]+}}, v1
  %sel1 = waveasm.v_cndmask_b32 %swap1, %v1, %vcc
      : !waveasm.vreg, !waveasm.pvreg<1>, !waveasm.psreg<106, 2> -> !waveasm.vreg
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v{{[0-9]+}}, v2
  %sel2 = waveasm.v_cndmask_b32 %swap2, %v2, %vcc
      : !waveasm.vreg, !waveasm.pvreg<2>, !waveasm.psreg<106, 2> -> !waveasm.vreg
  // CHECK: v_cndmask_b32 v{{[0-9]+}}, v{{[0-9]+}}, v3
  %sel3 = waveasm.v_cndmask_b32 %swap3, %v3, %vcc
      : !waveasm.vreg, !waveasm.pvreg<3>, !waveasm.psreg<106, 2> -> !waveasm.vreg

  %packed = waveasm.pack %sel0, %sel1, %sel2, %sel3
      : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<4, 4>

  // CHECK: buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], v4, s[0:3], 0 offen
  waveasm.buffer_store_dwordx4 %packed, %srd, %voff
      : !waveasm.vreg<4, 4>, !waveasm.psreg<0, 4>, !waveasm.pvreg<4>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}

// -----

// Test 2: dst==src fallback path for v_permlane16_swap_b32.
// When the allocator assigns the same register for dst and src, the emitter
// uses scratch VGPRs to avoid clobbering.
// This test uses a tight register budget to encourage dst==src allocation.
// CHECK-LABEL: permlane_dstsrc_fallback:
waveasm.program @permlane_dstsrc_fallback
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<>
  attributes {vgprs = 8 : i64, sgprs = 8 : i64} {

  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Whether the allocator chooses dst==src or dst!=src, we should see
  // a valid v_permlane16_swap_b32 in the output.
  // CHECK: v_permlane16_swap_b32
  %swap = waveasm.v_permlane16_swap_b32 %v0 : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: buffer_store_dword v{{[0-9]+}}, v1, s[0:3], 0 offen
  waveasm.buffer_store_dword %swap, %srd, %voff
      : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<1>

  // CHECK: s_endpgm
  waveasm.s_endpgm
}
