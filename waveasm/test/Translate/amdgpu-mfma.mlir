// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test amdgpu.mfma translation to v_mfma instructions

module {
  gpu.module @test_mfma {
    // CHECK: waveasm.program @mfma_kernel
    gpu.func @mfma_kernel(%arg0: memref<16x16xf16, #gpu.address_space<workgroup>>,
                          %arg1: memref<16x16xf16, #gpu.address_space<workgroup>>) kernel {
      %c0 = arith.constant 0 : index
      %zero = arith.constant dense<0.0> : vector<4xf32>

      // Load operands from LDS
      // CHECK: waveasm.ds_read_b64
      %a = vector.load %arg0[%c0, %c0] : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>
      // CHECK: waveasm.ds_read_b64
      %b = vector.load %arg1[%c0, %c0] : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<4xf16>

      // MFMA instruction with zero accumulator (inline constant)
      // CHECK: waveasm.v_mfma_f32_16x16x16_f16 {{.*}} !waveasm.imm<0> ->
      %result = amdgpu.mfma 16x16x16 %a * %b + %zero { blocks = 1 : i32 } blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
