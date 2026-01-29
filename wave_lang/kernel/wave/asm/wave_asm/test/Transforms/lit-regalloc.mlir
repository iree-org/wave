// RUN: waveasm-translate --waveasm-linear-scan %s | FileCheck %s
//
// Test: Linear scan register allocation

// CHECK-LABEL: waveasm.program @regalloc_basic
waveasm.program @regalloc_basic target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: Virtual registers are allocated to physical registers
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Virtual register result should be allocated to physical register
  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<
  %sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @regalloc_chain
waveasm.program @regalloc_chain target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %v0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %v1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>

  // Test: Chain of operations with register reuse
  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<2>
  %r1 = waveasm.v_add_u32 %v0, %v1 : !waveasm.pvreg<0>, !waveasm.pvreg<1> -> !waveasm.vreg

  // CHECK: waveasm.v_mul_lo_u32 {{.*}} -> !waveasm.pvreg<
  %r2 = waveasm.v_mul_lo_u32 %r1, %v0 : !waveasm.vreg, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @regalloc_multi_size
waveasm.program @regalloc_multi_size target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: Multi-register allocations (4-wide VGPR)
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // 4-wide load should be allocated to 4 consecutive VGPRs
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %loaded = waveasm.buffer_load_dwordx4 %srd, %voff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @regalloc_sgpr
waveasm.program @regalloc_sgpr target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: SGPR allocation
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %offset = waveasm.constant 0 : !waveasm.imm<0>

  // SGPR result from s_load
  // CHECK: waveasm.s_load_dwordx2 {{.*}} -> !waveasm.psreg<
  %loaded = waveasm.s_load_dwordx2 %sbase, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  waveasm.s_endpgm
}
