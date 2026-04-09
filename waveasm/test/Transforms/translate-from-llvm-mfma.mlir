// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify MFMA and shufflevector (element extraction) translation.
// Dense vector constant -> wide v_mov_b32; mfma -> v_mfma; shuffle -> extract.

// CHECK: waveasm.program @test__waveasm
// Zero-init accumulator (dense<0.0> : vector<4xf32>).
// CHECK: waveasm.v_mov_b32 {{.*}} -> !waveasm.vreg<4, 4>
// MFMA instruction.
// CHECK: waveasm.v_mfma_f32_16x16x16_f16
// Extract elements from MFMA result.
// CHECK: waveasm.extract {{.*}}[0] : !waveasm.vreg<4, 4> -> !waveasm.vreg
// CHECK: waveasm.extract {{.*}}[1] : !waveasm.vreg<4, 4> -> !waveasm.vreg
// CHECK: waveasm.extract {{.*}}[2] : !waveasm.vreg<4, 4> -> !waveasm.vreg
// CHECK: waveasm.extract {{.*}}[3] : !waveasm.vreg<4, 4> -> !waveasm.vreg
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %c0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf32>) : vector<4xf32>
    // Use zero for A and B inputs (vector<4xf16>).
    %a = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf16>) : vector<4xf16>
    %b = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf16>) : vector<4xf16>
    %mfma = rocdl.mfma.f32.16x16x16f16 %a, %b, %c0, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
    // Extract individual elements.
    %e0 = llvm.shufflevector %mfma, %mfma [0] : vector<4xf32>
    %e1 = llvm.shufflevector %mfma, %mfma [1] : vector<4xf32>
    %e2 = llvm.shufflevector %mfma, %mfma [2] : vector<4xf32>
    %e3 = llvm.shufflevector %mfma, %mfma [3] : vector<4xf32>
    llvm.return
  }
}
