// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test SRD (Shader Resource Descriptor) management for buffer access

// Test 1: Basic SRD setup and buffer load
// CHECK-LABEL: waveasm.program @srd_basic
waveasm.program @srd_basic target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // SRD is a 4-SGPR descriptor: base ptr (s[0:1]) + size + stride (s[2:3])
  // CHECK: waveasm.precolored.sreg 0, 4
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // VGPR offset for per-lane addressing
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Buffer load using SRD
  // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}} -> !waveasm.pvreg
  %data = waveasm.buffer_load_dword %srd, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 2: Multiple SRDs for different buffers
// CHECK-LABEL: waveasm.program @srd_multiple_buffers
waveasm.program @srd_multiple_buffers target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // SRD for buffer A
  // CHECK: waveasm.precolored.sreg 0, 4
  %srd_a = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>

  // SRD for buffer B
  // CHECK: waveasm.precolored.sreg 4, 4
  %srd_b = waveasm.precolored.sreg 4, 4 : !waveasm.psreg<4, 4>

  // SRD for buffer C
  // CHECK: waveasm.precolored.sreg 8, 4
  %srd_c = waveasm.precolored.sreg 8, 4 : !waveasm.psreg<8, 4>

  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Load from different buffers
  // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}} -> !waveasm.pvreg
  %data_a = waveasm.buffer_load_dword %srd_a, %off : !waveasm.psreg<0, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}} -> !waveasm.pvreg
  %data_b = waveasm.buffer_load_dword %srd_b, %off : !waveasm.psreg<4, 4>, !waveasm.pvreg<0> -> !waveasm.vreg
  // CHECK: waveasm.buffer_load_dword %{{.*}}, %{{.*}} -> !waveasm.pvreg
  %data_c = waveasm.buffer_load_dword %srd_c, %off : !waveasm.psreg<8, 4>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 3: SRD with scalar memory load (s_load)
// CHECK-LABEL: waveasm.program @srd_scalar_load
waveasm.program @srd_scalar_load target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Kernarg pointer in first 2 SGPRs
  %kernarg = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  %offset = waveasm.constant 0 : !waveasm.imm<0>

  // Load SRD from kernarg (e.g., buffer descriptor passed as kernel arg)
  // CHECK: waveasm.s_load_dwordx4
  %srd = waveasm.s_load_dwordx4 %kernarg, %offset : !waveasm.psreg<0, 2>, !waveasm.imm<0> -> !waveasm.sreg<4, 4>

  waveasm.s_endpgm
}

// Test 4: Buffer store with SRD
// CHECK-LABEL: waveasm.program @srd_buffer_store
waveasm.program @srd_buffer_store target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>
  %data = waveasm.precolored.vreg 1, 4 : !waveasm.pvreg<1, 4>

  // Buffer store using SRD
  // CHECK: waveasm.buffer_store_dwordx4
  waveasm.buffer_store_dwordx4 %data, %srd, %off : !waveasm.pvreg<1, 4>, !waveasm.psreg<0, 4>, !waveasm.pvreg<0>

  waveasm.s_endpgm
}

// Test 5: SRD with swizzling (cache swizzle stride)
// CHECK-LABEL: waveasm.program @srd_with_swizzle
waveasm.program @srd_with_swizzle target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  %srd = waveasm.precolored.sreg 0, 4 : !waveasm.psreg<0, 4>
  %base_off = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Swizzle offset computation: offset + (lane_id & swizzle_mask) * stride
  // This is typically handled by cache_swizzle_stride in the context
  %stride = waveasm.constant 16 : !waveasm.imm<16>
  // CHECK: waveasm.v_add_u32
  %swizzled_off = waveasm.v_add_u32 %base_off, %stride : !waveasm.pvreg<0>, !waveasm.imm<16> -> !waveasm.vreg

  // Load with swizzled offset
  // CHECK: waveasm.buffer_load_dword
  %data = waveasm.buffer_load_dword %srd, %swizzled_off : !waveasm.psreg<0, 4>, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}

// Test 6: Global memory access (saddr + voffset format)
// CHECK-LABEL: waveasm.program @global_memory_access
waveasm.program @global_memory_access target = #waveasm.target<#waveasm.gfx942, 5> abi = #waveasm.abi<> {
  // Base address in SGPRs (64-bit pointer)
  %sbase = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>
  // Per-lane offset in VGPR
  %voff = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>

  // Global load: base[lane_offset]
  // CHECK: waveasm.global_load_dword
  %data = waveasm.global_load_dword %sbase, %voff : !waveasm.psreg<0, 2>, !waveasm.pvreg<0> -> !waveasm.vreg

  waveasm.s_endpgm
}
