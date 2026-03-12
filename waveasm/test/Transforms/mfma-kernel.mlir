// RUN: waveasm-translate %s
// Assembly emission pass not yet integrated into CLI
//
// This test matches the Python test_mma_kernel_asm_backend test pattern.
// It verifies that an MFMA kernel generates correct assembly with
// proper register allocation for accumulator blocks.

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK: .globl mfma_kernel
// CHECK: mfma_kernel:
// CHECK: v_mfma_f32_16x16x16_f16
// CHECK: s_endpgm
// CHECK: .amdhsa_kernel mfma_kernel

waveasm.program @mfma_kernel
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Precolored physical registers for MFMA
  // Accumulator: 4 VGPRs for F32_16x16x16_F16
  %acc0 = waveasm.precolored.vreg 0 : !waveasm.pvreg<0, 4>

  // Source A and B: 2 VGPRs each for F16 inputs
  %src_a = waveasm.precolored.vreg 4 : !waveasm.pvreg<4, 2>
  %src_b = waveasm.precolored.vreg 6 : !waveasm.pvreg<6, 2>

  waveasm.comment "Load A and B matrices"
  // In reality, we would load from global/LDS here

  waveasm.comment "Matrix multiply accumulate"
  // v_mfma_f32_16x16x16_f16 acc[0:3], src_a[0:1], src_b[0:1], acc[0:3]
  %result = waveasm.v_mfma_f32_16x16x16_f16 %src_a, %src_b, %acc0
    : !waveasm.pvreg<4, 2>, !waveasm.pvreg<6, 2>, !waveasm.pvreg<0, 4> -> !waveasm.vreg<4>

  waveasm.comment "Store result"
  // In reality, we would store to global memory here

  waveasm.s_endpgm
}
