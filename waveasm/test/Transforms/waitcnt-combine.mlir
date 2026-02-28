// RUN: waveasm-translate --waveasm-insert-waitcnt %s | FileCheck %s
//
// Test: Adjacent waitcnt combining and double-free safety.
// The combineAdjacentWaitcnts pass merges consecutive s_waitcnt_vmcnt and
// s_waitcnt_lgkmcnt into a single s_waitcnt.  When three consecutive waits
// form a vmcnt/lgkmcnt/vmcnt triple, the first pair shares an element with
// the second pair -- the erased-set tracking prevents a double-free crash.

// CHECK-LABEL: waveasm.program @combine_vmcnt_lgkmcnt
waveasm.program @combine_vmcnt_lgkmcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %addr = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // Issue both VMEM and LDS loads
  %vmem = waveasm.buffer_load_dword %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg
  %lds  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<1> -> !waveasm.vreg

  // Using both results forces vmcnt and lgkmcnt waits.
  // The pass inserts s_waitcnt_vmcnt + s_waitcnt_lgkmcnt, then the
  // combine pass should merge them into a single s_waitcnt.
  // CHECK: waveasm.s_waitcnt vmcnt({{[0-9]+}}) lgkmcnt({{[0-9]+}})
  // CHECK-NEXT: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %vmem, %lds : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test: pre-existing separate waits followed by barrier.
// The schedule places s_waitcnt_vmcnt, then the pass adds s_waitcnt_lgkmcnt
// for the barrier, and the combine pass merges the adjacent pair.
// This exercises the path where consecutive toCombine pairs may share ops.
// CHECK-LABEL: waveasm.program @combine_with_barrier
waveasm.program @combine_with_barrier target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %addr = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %vmem = waveasm.buffer_load_dword %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg
  %lds  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<1> -> !waveasm.vreg

  // Schedule-placed vmcnt wait
  waveasm.s_waitcnt_vmcnt 0

  // Barrier needs lgkmcnt(0) too; the pass adds it and combine merges
  // CHECK: waveasm.s_waitcnt vmcnt(0) lgkmcnt(0)
  // CHECK-NEXT: waveasm.s_barrier
  waveasm.s_barrier

  waveasm.s_endpgm
}
