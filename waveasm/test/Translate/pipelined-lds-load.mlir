// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: vector.load from LDS memref iter_args inside a pipelined loop.
// The vector.load should use the SGPR-carried LDS base offset from the
// memref iter_arg, not a static offset.

module {
  gpu.module @test_pipelined_lds_load {

    // CHECK-LABEL: waveasm.program @pipelined_lds_read
    gpu.func @pipelined_lds_read(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index

      // Allocate LDS with two buffers
      %lds = memref.alloc() : memref<4096xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c2048_byte = arith.constant 2048 : index
      %viewA = memref.view %lds[%c0_byte][] : memref<4096xi8, 3> to memref<64x16xf16, 3>
      %viewB = memref.view %lds[%c2048_byte][] : memref<4096xi8, 3> to memref<64x16xf16, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // CHECK: waveasm.loop
      %result:3 = scf.for %i = %c0 to %c7 step %c1
          iter_args(%acc = %acc_init, %curRead = %viewA, %curWrite = %viewB)
          -> (vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>) {

        // vector.load from the current read buffer (memref iter_arg)
        // The SGPR-carried LDS base offset should be added to the address
        // via v_add_u32 (not v_mov_b32 -- SGPRs go directly into VALU src).
        //
        // CHECK:      v_add_u32 {{.*}}, %arg2 : !waveasm.vreg, !waveasm.sreg
        // CHECK-NEXT: ds_read_b64
        %data = vector.load %curRead[%tidx, %c0] : memref<64x16xf16, 3>, vector<4xf16>

        // Yield with swapped buffers
        scf.yield %acc, %curWrite, %curRead : vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>
      }

      // Condition swaps: %arg3, %arg2 (B then A)
      // CHECK: waveasm.condition {{.*}} iter_args({{.*}}, {{.*}}, %arg3, %arg2)

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
