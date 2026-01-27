// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test vector load/store operations - buffer, global, LDS (DS)

// Test 1: Buffer loads - various sizes
// CHECK-LABEL: waveasm.program @buffer_loads
waveasm.program @buffer_loads target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // SRD (buffer descriptor) in SGPRs
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  // Offset in VGPR
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.buffer_load_dword
  %v0 = waveasm.buffer_load_dword %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.buffer_load_dwordx2
  %v1 = waveasm.buffer_load_dwordx2 %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg<2>

  // CHECK: waveasm.buffer_load_dwordx4
  %v2 = waveasm.buffer_load_dwordx4 %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 2: Buffer stores
// CHECK-LABEL: waveasm.program @buffer_stores
waveasm.program @buffer_stores target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %data = waveasm.precolored.vreg 1, 4 : !waveasm.pvreg<1, 4>

  // CHECK: waveasm.buffer_store_dwordx4
  waveasm.buffer_store_dwordx4 %data, %srd, %off : !waveasm.pvreg<1, 4>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

// Test 3: Global loads (saddr + voffset format)
// CHECK-LABEL: waveasm.program @global_loads
waveasm.program @global_loads target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  // Base address in SGPRs (64-bit pointer)
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  // Per-lane offset in VGPR
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.global_load_dword
  %v0 = waveasm.global_load_dword %sbase, %voff : !waveasm.psreg<0, 2>, !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.global_load_dwordx4
  %v1 = waveasm.global_load_dwordx4 %sbase, %voff : !waveasm.psreg<0, 2>, !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 4: Global stores
// CHECK-LABEL: waveasm.program @global_stores
waveasm.program @global_stores target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %data = waveasm.precolored.vreg 2, 4 : !waveasm.pvreg<2, 4>

  // CHECK: waveasm.global_store_dwordx4
  waveasm.global_store_dwordx4 %data, %sbase, %voff : !waveasm.pvreg<2, 4>, !waveasm.psreg<0, 2>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

// Test 5: LDS reads (DS operations)
// CHECK-LABEL: waveasm.program @lds_reads
waveasm.program @lds_reads target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %ldsoff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // CHECK: waveasm.ds_read_b32
  %v0 = waveasm.ds_read_b32 %ldsoff : !waveasm.pvreg<0> -> !waveasm.vreg

  // CHECK: waveasm.ds_read_b64
  %v1 = waveasm.ds_read_b64 %ldsoff : !waveasm.pvreg<0> -> !waveasm.vreg<2>

  // CHECK: waveasm.ds_read_b128
  %v2 = waveasm.ds_read_b128 %ldsoff : !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  waveasm.s_endpgm
}

// Test 6: LDS writes (DS operations)
// CHECK-LABEL: waveasm.program @lds_writes
waveasm.program @lds_writes target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %ldsoff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %data32 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %data64 = waveasm.precolored.vreg 2, 2 : !waveasm.pvreg<2, 2>

  // CHECK: waveasm.ds_write_b32
  waveasm.ds_write_b32 %data32, %ldsoff : !waveasm.pvreg<1>, !waveasm.pvreg<0>

  // CHECK: waveasm.ds_write_b64
  waveasm.ds_write_b64 %data64, %ldsoff : !waveasm.pvreg<2, 2>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

// Test 7: SGPR loads (scalar memory)
// CHECK-LABEL: waveasm.program @smem_loads
waveasm.program @smem_loads target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %offset = waveasm.constant 0 : !waveasm.imm<0>

  // CHECK: waveasm.s_load_dword
  %s0 = waveasm.s_load_dword %sbase, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.s_load_dwordx2
  %s1 = waveasm.s_load_dwordx2 %sbase, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<2>

  // CHECK: waveasm.s_load_dwordx4
  %s2 = waveasm.s_load_dwordx4 %sbase, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}

// Test 8: Register allocation for memory ops
// CHECK-LABEL: waveasm.program @mem_regalloc
waveasm.program @mem_regalloc target = #waveasm.target<"gfx942", 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Load into virtual reg - should get allocated
  // CHECK: waveasm.buffer_load_dwordx4 {{.*}} -> !waveasm.pvreg<{{[0-9]+}}, 4>
  %loaded = waveasm.buffer_load_dwordx4 %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg<4, 4>

  // Use the loaded data in computation
  // CHECK: waveasm.v_add_u32 {{.*}} -> !waveasm.pvreg<
  %sum = waveasm.v_add_u32 %loaded, %off : !waveasm.vreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
