// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s 2>&1 | FileCheck %s
//
// Test: Full assembly output for a pipelined double-buffer loop.
// Verifies the SGPR rotation (swap) pattern in the generated assembly.
//
// The loop carries two memref iter_args (LDS buffers A and B) and swaps them
// each iteration. The assembly must contain:
// 1. s_mov_b32 initialization of the two SGPR offsets
// 2. A loop label
// 3. ds_read using one of the SGPR offsets
// 4. s_mov_b32 rotation (3-instruction swap using a temporary)
// 5. s_cbranch_scc1 back to the loop label

module {
  gpu.module @test_pipelined_double_buffer {

    // CHECK-LABEL: pipelined_double_buffer:
    gpu.func @pipelined_double_buffer(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c15 = arith.constant 15 : index
      %c1 = arith.constant 1 : index

      %lds = memref.alloc() : memref<8192xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c4096_byte = arith.constant 4096 : index
      %viewA = memref.view %lds[%c0_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>
      %viewB = memref.view %lds[%c4096_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // Init: s_mov_b32 for both LDS offsets
      // CHECK-DAG:  s_mov_b32 s{{[0-9]+}}, 0
      // CHECK-DAG:  s_mov_b32 s{{[0-9]+}}, 4096
      //
      // Loop with ds_read using SGPR offset
      // CHECK:      L_loop_0:
      // CHECK:      ds_read_b64
      //
      // SGPR rotation: 3-instruction swap pattern
      // CHECK:      s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_cbranch_scc1 L_loop_0
      %result:3 = scf.for %i = %c0 to %c15 step %c1
          iter_args(%acc = %acc_init, %curA = %viewA, %curB = %viewB)
          -> (vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>) {
        %data = vector.load %curA[%tidx, %c0] : memref<64x16xf16, 3>, vector<4xf16>
        scf.yield %acc, %curB, %curA : vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>
      }

      // CHECK: s_endpgm
      gpu.return
    }
  }
}
