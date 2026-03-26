// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: When a single SSA value is used as input to multiple PackOps (e.g.,
// after ScopedCSE merges identical S_MOV_B32 constants), the liveness pass
// inserts S_MOV_B32/V_MOV_B32 copies so each PackOp slot gets a distinct
// value with its own physical register assignment.
//
// Without the fix, the last PackOp's assignment would win for the shared
// value, leaving earlier PackOps with the wrong physical register for that
// slot (corrupting SRD words).

// -----------------------------------------------------------------------
// Test 1: Same VGPR value shared across two PackOps.
// The second PackOp must get a copy of the shared input.
// -----------------------------------------------------------------------

// CHECK-LABEL: waveasm.program @pack_shared_vgpr_across_packs
waveasm.program @pack_shared_vgpr_across_packs target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %imm1 = waveasm.constant 1 : !waveasm.imm<1>
  %imm2 = waveasm.constant 2 : !waveasm.imm<2>

  %v0 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg
  %v1 = waveasm.v_mov_b32 %imm1 : !waveasm.imm<1> -> !waveasm.vreg

  // %v0 is shared between both packs. The first pack claims it; the second
  // must get a copy so both packs have distinct physical registers.
  // CHECK: [[V0:%.*]] = waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[R0:[0-9]+]]>
  // CHECK: waveasm.pack [[V0]], {{.*}} -> !waveasm.pvreg<[[R0]], 2>
  %pack_a = waveasm.pack %v0, %v1 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  %v2 = waveasm.v_mov_b32 %imm2 : !waveasm.imm<2> -> !waveasm.vreg

  // A copy of %v0 should be inserted for this pack.
  // CHECK: waveasm.v_mov_b32 [[V0]]
  // CHECK: waveasm.pack
  %pack_b = waveasm.pack %v0, %v2 : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  waveasm.buffer_store_dwordx2 %pack_a, %srd, %voff : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.buffer_store_dwordx2 %pack_b, %srd, %voff : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

// -----------------------------------------------------------------------
// Test 2: Same VGPR value duplicated three times within a single PackOp.
// Each slot must get its own distinct physical register.
// -----------------------------------------------------------------------

// CHECK-LABEL: waveasm.program @pack_dup_vgpr_within_pack
waveasm.program @pack_dup_vgpr_within_pack target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %v0 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg

  // %v0 used three times in the same pack. The first use claims it,
  // remaining uses get copies. All three must land in contiguous registers.
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[BASE:[0-9]+]]>
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg
  // CHECK: waveasm.pack {{.*}} -> !waveasm.pvreg<[[BASE]], 3>
  %packed = waveasm.pack %v0, %v0, %v0 : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<3>

  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  waveasm.buffer_store_dwordx3 %packed, %srd, %voff : !waveasm.vreg<3>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}
