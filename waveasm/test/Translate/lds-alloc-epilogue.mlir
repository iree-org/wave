// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: LDS base offset propagation through scf.for results to epilogue.
//
// In pipelined MXFP4 GEMM, the epilogue code (after the scf.for) reads from
// memref results of the loop.  The LDS base offset must be propagated from the
// loop's init args through the yield/results so the epilogue vector.load can
// find the correct LDS address.

module {
  gpu.module @test_lds_alloc_epilogue {

    // CHECK-LABEL: waveasm.program @lds_alloc_epilogue
    gpu.func @lds_alloc_epilogue(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index

      // Two LDS allocations at different offsets
      %bufA = memref.alloc() : memref<16x128xi8, 3>
      %bufB = memref.alloc() : memref<16x128xi8, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // CHECK: waveasm.loop
      %result:3 = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %acc_init, %curA = %bufA, %curB = %bufB)
          -> (vector<4xf32>, memref<16x128xi8, 3>, memref<16x128xi8, 3>) {
        %data = vector.load %curA[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        scf.yield %acc, %curB, %curA : vector<4xf32>, memref<16x128xi8, 3>, memref<16x128xi8, 3>
      }

      // Epilogue: load from the loop result memref (result #1).
      // The LDS base offset should be propagated from the loop result.
      // CHECK: v_add_u32
      // CHECK: ds_read_b128
      %epilogue_data = vector.load %result#1[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
