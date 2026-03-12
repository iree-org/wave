// RUN: waveasm-translate --waveasm-insert-waitcnt %s | FileCheck %s
//
// Test: Merging S_WAITCNT with adjacent individual S_WAITCNT_VMCNT or
// S_WAITCNT_LGKMCNT.  The combineAdjacentWaitcnts pass should fold the
// individual counter into the combined waitcnt, taking the minimum of
// each field.

// --- Pattern: S_WAITCNT followed by S_WAITCNT_VMCNT -------------------------
// CHECK-LABEL: waveasm.program @combined_then_vmcnt
waveasm.program @combined_then_vmcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %addr = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %vmem = waveasm.buffer_load_dword %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg
  %lds  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<1> -> !waveasm.vreg

  // Pre-existing combined wait + individual vmcnt should merge.
  // CHECK: waveasm.s_waitcnt vmcnt(0) lgkmcnt(0)
  // CHECK-NOT: waveasm.s_waitcnt_vmcnt
  // CHECK-NEXT: waveasm.s_barrier
  waveasm.s_waitcnt vmcnt(0) lgkmcnt(0)
  waveasm.s_waitcnt_vmcnt 0
  waveasm.s_barrier

  waveasm.s_endpgm
}

// --- Pattern: S_WAITCNT_LGKMCNT followed by S_WAITCNT ------------------------
// CHECK-LABEL: waveasm.program @lgkmcnt_then_combined
waveasm.program @lgkmcnt_then_combined target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %addr = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  %vmem = waveasm.buffer_load_dword %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg
  %lds  = waveasm.ds_read_b32 %addr : !waveasm.pvreg<1> -> !waveasm.vreg

  // Individual lgkmcnt followed by combined wait should merge.
  // CHECK: waveasm.s_waitcnt vmcnt(0) lgkmcnt(0)
  // CHECK-NOT: waveasm.s_waitcnt_lgkmcnt
  // CHECK-NEXT: waveasm.s_barrier
  waveasm.s_waitcnt_lgkmcnt 0
  waveasm.s_waitcnt vmcnt(0) lgkmcnt(0)
  waveasm.s_barrier

  waveasm.s_endpgm
}
