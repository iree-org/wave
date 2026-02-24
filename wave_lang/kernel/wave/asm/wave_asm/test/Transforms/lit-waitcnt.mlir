// RUN: waveasm-translate --waveasm-insert-waitcnt %s | FileCheck %s
//
// Test: Waitcnt insertion pass for memory operations

// CHECK-LABEL: waveasm.program @vmem_waitcnt
waveasm.program @vmem_waitcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // VMEM load
  %loaded = waveasm.buffer_load_dword %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Waitcnt should be inserted before using the loaded value
  // CHECK: waveasm.buffer_load_dword
  // CHECK: waveasm.s_waitcnt
  // CHECK: waveasm.v_add_u32
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %result = waveasm.v_add_u32 %loaded, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @smem_waitcnt
waveasm.program @smem_waitcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %offset = waveasm.constant 0 : !waveasm.imm<0>

  // SMEM load
  %loaded = waveasm.s_load_dword %sbase, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg

  // Waitcnt should be inserted for lgkmcnt
  // CHECK: waveasm.s_load_dword
  // CHECK: waveasm.s_waitcnt
  %s1 = waveasm.precolored.sreg 4 : !waveasm.psreg<4>
  %result = waveasm.s_add_u32 %loaded, %s1 : !waveasm.sreg, !waveasm.psreg<4> -> !waveasm.sreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @lds_waitcnt
waveasm.program @lds_waitcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // LDS read
  %loaded = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg

  // Waitcnt should be inserted for lgkmcnt
  // CHECK: waveasm.ds_read_b32
  // CHECK: waveasm.s_waitcnt
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %result = waveasm.v_add_u32 %loaded, %c1 : !waveasm.vreg, !waveasm.imm<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @multiple_loads
waveasm.program @multiple_loads target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %voff1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // Multiple VMEM loads in flight
  // CHECK: waveasm.buffer_load_dword
  // CHECK: waveasm.buffer_load_dword
  %load1 = waveasm.buffer_load_dword %srd, %voff0, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg
  %load2 = waveasm.buffer_load_dword %srd, %voff1, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<1>, !waveasm.imm<0> -> !waveasm.vreg

  // Waitcnt before using results
  // CHECK: waveasm.s_waitcnt
  %result = waveasm.v_add_u32 %load1, %load2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @barrier_waitcnt_insertion
waveasm.program @barrier_waitcnt_insertion target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // VMEM load followed by barrier
  // CHECK: waveasm.buffer_load_dword
  %load1 = waveasm.buffer_load_dword %srd, %voff0, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Ticketing pass should insert s_waitcnt before the barrier
  // to ensure all memory operations are complete before synchronization
  // CHECK: waveasm.s_waitcnt
  // CHECK-NEXT: waveasm.s_barrier
  waveasm.s_barrier

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @existing_waitcnt_observed
waveasm.program @existing_waitcnt_observed target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // VMEM load
  // CHECK: waveasm.buffer_load_dword
  %load1 = waveasm.buffer_load_dword %srd, %voff0, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg

  // Pre-existing waitcnt - pass observes this and knows VMEM is drained.
  // No LGKM ops were issued, so barrier needs no additional lgkmcnt wait.
  // CHECK: waveasm.s_waitcnt_vmcnt 0
  waveasm.s_waitcnt_vmcnt 0

  // Barrier follows - no additional waitcnt needed since vmcnt(0) already
  // observed and no LGKM operations are outstanding.
  // CHECK-NOT: waveasm.s_waitcnt
  // CHECK: waveasm.s_barrier
  waveasm.s_barrier

  waveasm.s_endpgm
}
