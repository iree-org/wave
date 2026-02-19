// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: scf.for with memref iter_args (LDS double-buffering) translated to
// waveasm.loop with SGPR-carried LDS offsets.
//
// In pipelined GEMM, scf.for carries memref iter_args that swap each iteration
// (ping-pong double buffering). The C++ backend must:
// 1. Resolve each memref init arg to its LDS base offset (from memref.view)
// 2. Materialize the offset as an SGPR (s_mov_b32)
// 3. Pass the SGPR through the loop as an iter_arg
// 4. At yield, resolve the swapped memref's offset for the condition iter_args

module {
  gpu.module @test_memref_iter_args {

    // --- Two memref iter_args: double-buffer ping-pong ---
    // The scf.for swaps viewA <-> viewB each iteration.
    // CHECK-LABEL: waveasm.program @memref_double_buffer
    gpu.func @memref_double_buffer() kernel {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index

      // Allocate LDS and create two views at different offsets
      %lds = memref.alloc() : memref<8192xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c4096_byte = arith.constant 4096 : index
      %viewA = memref.view %lds[%c0_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>
      %viewB = memref.view %lds[%c4096_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>

      // MFMA accumulator init
      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // Loop with 3 iter_args: accumulator (vector), viewA (memref), viewB (memref)
      // The memref iter_args should become SGPR-carried LDS offsets.
      //
      // Init: [IV=0, acc=vreg<4>, offsetA=sreg(0), offsetB=sreg(4096)]
      //
      // CHECK:      %[[IV_INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK:      %[[ACC_INIT:.*]] = waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.vreg<4, 4>
      // Memref A (offset 0) -> s_mov_b32 with immediate 0
      // CHECK:      %[[OA_INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // Memref B (offset 4096) -> s_mov_b32 with immediate 4096
      // CHECK:      %[[OB_INIT:.*]] = waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<4096> -> !waveasm.sreg
      //
      // Loop with 4 iter_args: [IV, acc, offsetA, offsetB]
      // CHECK:      %{{.*}}:4 = waveasm.loop (%[[IV:.*]] = %[[IV_INIT]], %[[ACC:.*]] = %[[ACC_INIT]], %[[OA:.*]] = %[[OA_INIT]], %[[OB:.*]] = %[[OB_INIT]])
      // CHECK-SAME:   (!waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.sreg, !waveasm.sreg)
      %result:3 = scf.for %i = %c0 to %c7 step %c1
          iter_args(%acc = %acc_init, %curA = %viewA, %curB = %viewB)
          -> (vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>) {
        // Loop body would normally have vector.load, mfma, gather_to_lds, etc.
        // For this test, just yield with swapped memrefs.

        // Yield: swap viewA <-> viewB (double-buffer ping-pong)
        scf.yield %acc, %curB, %curA : vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>
      }

      // Condition should carry: [nextIV, acc, offsetB, offsetA] (swapped!)
      // CHECK:      waveasm.condition %{{.*}} : !waveasm.sreg iter_args(%{{.*}}, %[[ACC]], %[[OB]], %[[OA]]) : !waveasm.sreg, !waveasm.vreg<4, 4>, !waveasm.sreg, !waveasm.sreg

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
