// RUN: not waveasm-translate --waveasm-linear-scan="max-vgprs=4 max-agprs=8" %s 2>&1 | FileCheck %s
//
// Test: Multi-register (size > 1) values are NOT spill candidates.
// When the only live values are multi-register, eviction is not attempted
// and allocation fails.

// CHECK: error: Failed to allocate VGPR
waveasm.program @multi_reg_no_spill target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %soff = waveasm.constant 0 : !waveasm.imm<0>

  // A 4-wide load consumes v2..v5 (v0 precolored, v14/v15 reserved).
  // That exhausts the pool.  There are no size-1 eviction candidates.
  %wide = waveasm.buffer_load_dwordx4 %srd, %voff, %soff : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

  // This needs another VGPR but nothing can be evicted (wide is size 4).
  %extra = waveasm.v_add_u32 %voff, %voff : !waveasm.pvreg<0>, !waveasm.pvreg<0> -> !waveasm.vreg

  // Keep wide live past extra to create the pressure.
  %elem = waveasm.extract %wide[0] : !waveasm.vreg<4, 4> -> !waveasm.vreg
  %sum = waveasm.v_add_u32 %elem, %extra : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
