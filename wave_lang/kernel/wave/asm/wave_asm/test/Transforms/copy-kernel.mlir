// RUN: waveasm-translate %s
// Verify copy kernel translates without errors
//
// This test matches the Python test_copy_kernel_asm_backend test pattern.
// It verifies that a simple copy kernel generates correct assembly.

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .text
// CHECK: .globl copy_kernel
// CHECK: copy_kernel:
// CHECK-DAG: global_load
// CHECK-DAG: global_store
// CHECK: s_endpgm
// CHECK: .amdhsa_kernel copy_kernel

waveasm.program @copy_kernel
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Physical registers for ABI
  %tid = waveasm.precolored.vreg 0 : !waveasm.pvreg<0>  // Thread ID in v0
  %kptr_lo = waveasm.precolored.sreg 0 : !waveasm.psreg<0>  // Kernarg ptr low
  %kptr_hi = waveasm.precolored.sreg 1 : !waveasm.psreg<1>  // Kernarg ptr high

  // Working registers
  %addr = waveasm.precolored.vreg 1 : !waveasm.pvreg<1>
  %data = waveasm.precolored.vreg 2 : !waveasm.pvreg<2>

  waveasm.comment "Calculate address from thread ID"

  // Load from global memory (using saddr, voffset form)
  %saddr = waveasm.precolored.sreg 4, 2 : !waveasm.psreg<4, 2>
  %loaded = waveasm.global_load_dword %saddr, %addr : !waveasm.psreg<4, 2>, !waveasm.pvreg<1> -> !waveasm.vreg

  // Store to global memory
  waveasm.global_store_dword %loaded, %saddr, %addr : !waveasm.vreg, !waveasm.psreg<4, 2>, !waveasm.pvreg<1>

  waveasm.s_endpgm
}
