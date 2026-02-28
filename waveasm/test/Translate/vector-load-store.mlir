// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test vector.load and vector.store translation to WAVEASM instructions

module {
  gpu.module @test_kernel {
    // CHECK: waveasm.program @vector_ops
    gpu.func @vector_ops(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      // CHECK: waveasm.buffer_load_dwordx2
      %v = vector.load %arg0[%c0, %c0] : memref<16x16xf16>, vector<4xf16>

      // CHECK: waveasm.buffer_store_dwordx2
      vector.store %v, %arg1[%c0, %c0] : memref<16x16xf16>, vector<4xf16>

      gpu.return
    }
  }
}
