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
  %soff0 = waveasm.constant 0 : !waveasm.imm<0>

  // 4-wide load should be allocated to 4 consecutive VGPRs
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %loaded = waveasm.buffer_load_dwordx4 %srd, %voff, %soff0 : !waveasm.psreg<0, 4>, !waveasm.pvreg<0>, !waveasm.imm<0> -> !waveasm.vreg<4, 4>

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

// CHECK-LABEL: waveasm.program @regalloc_pack_store
waveasm.program @regalloc_pack_store target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: Pack inputs get contiguous physical registers matching the pack result.
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>

  %lo = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg
  %hi = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg

  // Pack result should get a contiguous 2-wide allocation.
  // Pack inputs should get result.physReg + 0 and result.physReg + 1.
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[LO:[0-9]+]]>
  // CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.pvreg<[[HI:[0-9]+]]>
  // CHECK: waveasm.pack {{.*}} -> !waveasm.pvreg<[[LO]], 2>
  %packed = waveasm.pack %lo, %hi : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  waveasm.buffer_store_dwordx2 %packed, %srd, %voff : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @regalloc_pack_input_post_use
waveasm.program @regalloc_pack_input_post_use target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: Pack input used independently after the pack op.
  // The pack result's live range must extend to cover the input's post-pack use,
  // otherwise the allocator could reuse the register prematurely.
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>

  %lo = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg
  %hi = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg

  // CHECK: waveasm.pack {{.*}} -> !waveasm.pvreg<[[BASE:[0-9]+]], 2>
  %packed = waveasm.pack %lo, %hi : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2>

  waveasm.buffer_store_dwordx2 %packed, %srd, %voff : !waveasm.vreg<2>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  // %lo is used again after the pack. The pack result's live range must cover
  // this point so that %lo's physical register (BASE+0) is not reallocated.
  // CHECK: waveasm.buffer_store_dword {{.*}} : !waveasm.pvreg<[[BASE]]>,
  waveasm.buffer_store_dword %lo, %srd, %voff : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}

// CHECK-LABEL: waveasm.program @regalloc_extract_of_pack
waveasm.program @regalloc_extract_of_pack target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Test: Extract from pack result gets correct sub-register.
  %imm0 = waveasm.constant 0 : !waveasm.imm<0>
  %imm1 = waveasm.constant 1 : !waveasm.imm<1>
  %imm2 = waveasm.constant 2 : !waveasm.imm<2>
  %imm3 = waveasm.constant 3 : !waveasm.imm<3>

  %v0 = waveasm.v_mov_b32 %imm0 : !waveasm.imm<0> -> !waveasm.vreg
  %v1 = waveasm.v_mov_b32 %imm1 : !waveasm.imm<1> -> !waveasm.vreg
  %v2 = waveasm.v_mov_b32 %imm2 : !waveasm.imm<2> -> !waveasm.vreg
  %v3 = waveasm.v_mov_b32 %imm3 : !waveasm.imm<3> -> !waveasm.vreg

  // CHECK: waveasm.pack {{.*}} -> !waveasm.pvreg<[[BASE:[0-9]+]], 4>
  %packed = waveasm.pack %v0, %v1, %v2, %v3 : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<4, 4>

  // Extract element 2 should yield physReg = pack result base + 2.
  // CHECK: waveasm.extract {{.*}}[2] : !waveasm.pvreg<[[BASE]], 4> -> !waveasm.pvreg<[[ELEM:[0-9]+]]>
  %elem = waveasm.extract %packed[2] : !waveasm.vreg<4, 4> -> !waveasm.vreg

  // Verify the extracted element's physical register is used in the store.
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // CHECK: waveasm.buffer_store_dword {{.*}} : !waveasm.pvreg<[[ELEM]]>,
  waveasm.buffer_store_dword %elem, %srd, %voff : !waveasm.vreg, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>
  waveasm.s_endpgm
}
