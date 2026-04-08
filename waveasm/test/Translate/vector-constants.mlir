// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: end-to-end translation after extract scalarization rewrites the
// broadcast+dense-const chain to scalar ops.

module {
  gpu.module @test_vec_constants {

    // The scalarization pass rewrites extract[2](addi(broadcast(base),
    // dense<[10,11,12,13]>)) into scalar arith.addi(base, 12).
    // The translator emits a single v_add_u32 with the constant 12.
    //
    // CHECK-LABEL: waveasm.program @scalarized_kernel
    // CHECK: waveasm.constant 12
    // CHECK: waveasm.s_add_u32
    // CHECK: waveasm.s_endpgm
    gpu.func @scalarized_kernel(%base: i32) kernel {
      %bcast = vector.broadcast %base : i32 to vector<4xi32>
      %offsets = arith.constant dense<[10, 11, 12, 13]> : vector<4xi32>
      %sum = arith.addi %bcast, %offsets : vector<4xi32>
      %lane = vector.extract %sum[2] : i32 from vector<4xi32>
      gpu.return
    }
  }
}
