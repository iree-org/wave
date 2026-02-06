// RUN: waveasm-translate %s | FileCheck %s
// Test memory effect annotations on operations

// CHECK-LABEL: @memory_load_operations
waveasm.program @memory_load_operations
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %srd = waveasm.precolored.sreg 8 : !waveasm.sreg<4>
  %offset = waveasm.precolored.vreg 0 : !waveasm.vreg

  // Buffer loads have Read effect on global memory
  // CHECK: waveasm.buffer_load_dword
  %val1 = waveasm.buffer_load_dword %srd, %offset
      : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.buffer_load_dwordx4
  %val2 = waveasm.buffer_load_dwordx4 %srd, %offset
      : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg<4>

  waveasm.s_endpgm
}

// CHECK-LABEL: @memory_store_operations
waveasm.program @memory_store_operations
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  %srd = waveasm.precolored.sreg 8 : !waveasm.sreg<4>
  %offset = waveasm.precolored.vreg 0 : !waveasm.vreg
  %data = waveasm.precolored.vreg 1 : !waveasm.vreg

  // Buffer stores have Write effect on global memory
  // CHECK: waveasm.buffer_store_dword
  waveasm.buffer_store_dword %data, %srd, %offset
      : !waveasm.vreg, !waveasm.sreg<4>, !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: @lds_operations
waveasm.program @lds_operations
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64, lds_size = 4096 : i64} {

  %addr = waveasm.precolored.vreg 0 : !waveasm.vreg
  %data = waveasm.precolored.vreg 1 : !waveasm.vreg

  // LDS operations have Read/Write effects on workgroup memory
  // CHECK: waveasm.ds_read_b32
  %loaded = waveasm.ds_read_b32 %addr : !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.ds_write_b32
  waveasm.ds_write_b32 %data, %addr : !waveasm.vreg, !waveasm.vreg

  waveasm.s_endpgm
}

// CHECK-LABEL: @synchronization_operations
waveasm.program @synchronization_operations
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 32 : i64, sgprs = 32 : i64} {

  // Barrier synchronizes all memory accesses
  // Has Read and Write effects on all memory resources
  // CHECK: waveasm.s_barrier
  waveasm.s_barrier

  // Waitcnt creates a memory fence
  // CHECK: waveasm.s_waitcnt
  waveasm.s_waitcnt lgkmcnt (0)

  waveasm.s_endpgm
}

// CHECK-LABEL: @pure_arithmetic
waveasm.program @pure_arithmetic
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 16 : i64, sgprs = 16 : i64} {

  %a = waveasm.precolored.vreg 0 : !waveasm.vreg
  %b = waveasm.precolored.vreg 1 : !waveasm.vreg

  // Arithmetic operations are Pure (no side effects)
  // CHECK: waveasm.v_add_u32
  %sum = waveasm.v_add_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // CHECK: waveasm.v_mul_lo_u32
  %product = waveasm.v_mul_lo_u32 %a, %b : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  waveasm.s_endpgm
}
