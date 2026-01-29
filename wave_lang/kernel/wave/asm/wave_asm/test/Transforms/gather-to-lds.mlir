// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test gather-to-LDS operations (buffer_load directly to LDS via M0)

// Test 1: Basic gather-to-LDS using buffer_load_dword_lds
// CHECK-LABEL: waveasm.program @gather_to_lds_basic
waveasm.program @gather_to_lds_basic target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // SRD (buffer descriptor) for global memory
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  // Per-lane offset in VGPR
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  // LDS offset (scalar)
  %lds_off = waveasm.constant 256 : !waveasm.imm<256>
  // Scalar offset for buffer (usually 0)
  %soff = waveasm.constant 0 : !waveasm.imm<0>

  // Set M0 to LDS destination offset
  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_off : !waveasm.imm<256>

  // Gather 4 bytes from global to LDS
  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %voff, %srd, %soff : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

// Test 2: Gather-to-LDS with dwordx4 (16 bytes)
// CHECK-LABEL: waveasm.program @gather_to_lds_x4
waveasm.program @gather_to_lds_x4 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %lds_off = waveasm.constant 512 : !waveasm.imm<512>
  %soff = waveasm.constant 0 : !waveasm.imm<0>

  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_off : !waveasm.imm<512>

  // Gather 16 bytes from global to LDS
  // CHECK: waveasm.buffer_load_dwordx4_lds
  waveasm.buffer_load_dwordx4_lds %voff, %srd, %soff : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

// Test 3: Dynamic M0 from SGPR
// CHECK-LABEL: waveasm.program @gather_dynamic_m0
waveasm.program @gather_dynamic_m0 target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %soff = waveasm.constant 0 : !waveasm.imm<0>

  // Compute LDS offset dynamically (e.g., row * cols + col) * elem_bytes
  // Simulated here with precolored SGPR
  %dynamic_lds_off = waveasm.precolored.sreg 4 : !waveasm.psreg<4>

  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %dynamic_lds_off : !waveasm.psreg<4>

  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %voff, %srd, %soff : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}

// Test 4: Multiple gather operations in sequence
// CHECK-LABEL: waveasm.program @gather_sequence
waveasm.program @gather_sequence target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %voff0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %voff1 = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %lds_off0 = waveasm.constant 0 : !waveasm.imm<0>
  %lds_off1 = waveasm.constant 64 : !waveasm.imm<64>
  %soff = waveasm.constant 0 : !waveasm.imm<0>

  // First gather at LDS offset 0
  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_off0 : !waveasm.imm<0>
  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %voff0, %srd, %soff : !waveasm.pvreg<0>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  // Second gather at LDS offset 64
  // CHECK: waveasm.s_mov_b32_m0
  waveasm.s_mov_b32_m0 %lds_off1 : !waveasm.imm<64>
  // CHECK: waveasm.buffer_load_dword_lds
  waveasm.buffer_load_dword_lds %voff1, %srd, %soff : !waveasm.pvreg<1>, !waveasm.psreg<0, 4>, !waveasm.imm<0>

  waveasm.s_endpgm
}
