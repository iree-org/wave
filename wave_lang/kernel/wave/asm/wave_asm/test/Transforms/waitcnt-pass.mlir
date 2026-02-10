// RUN: waveasm-translate --waveasm-insert-waitcnt %s 2>&1 | FileCheck %s
//
// Test the waitcnt insertion pass on WaveASM IR
// This tests that memory operations are properly tracked and waitcnts inserted

// CHECK-LABEL: waveasm.program @simple_load_use
waveasm.program @simple_load_use target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Define address register
  %addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // LDS load (LGKM counter)
  // CHECK: waveasm.ds_read_b32
  %loaded = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg

  // Waitcnt should be inserted before the use of the loaded value
  // CHECK-NEXT: waveasm.s_waitcnt_lgkmcnt 0
  // CHECK-NEXT: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %loaded, %loaded : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.s_endpgm
  waveasm.s_endpgm
}

// Test VMEM load with vmcnt insertion
// CHECK-LABEL: waveasm.program @vmem_load_use
waveasm.program @vmem_load_use target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %saddr = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %voffset = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Global load (VMEM counter)
  // CHECK: waveasm.global_load_dword
  %loaded = waveasm.global_load_dword %saddr, %voffset : !waveasm.psreg<0, 2>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Waitcnt should be inserted before use
  // CHECK-NEXT: waveasm.s_waitcnt_vmcnt 0
  // CHECK-NEXT: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %loaded, %loaded : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test multiple loads with coalesced waitcnt
// CHECK-LABEL: waveasm.program @coalesced_waitcnt
waveasm.program @coalesced_waitcnt target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %addr = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Two LDS loads
  // CHECK: waveasm.ds_read_b32
  %load1 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg
  // CHECK: waveasm.ds_read_b32
  %load2 = waveasm.ds_read_b32 %addr : !waveasm.pvreg<0> -> !waveasm.vreg

  // Use both - should wait for both with lgkmcnt(0)
  // CHECK: waveasm.s_waitcnt_lgkmcnt 0
  // CHECK-NEXT: waveasm.v_add_u32
  %result = waveasm.v_add_u32 %load1, %load2 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
