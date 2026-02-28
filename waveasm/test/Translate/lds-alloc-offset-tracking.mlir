// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: LDS base offset tracking for direct memref.alloc (without memref.view).
//
// In MXFP4 scheduled GEMM kernels, LDS buffers are allocated as separate typed
// memrefs (e.g. memref<256x128xi8, 3>) rather than as views into a single raw
// byte buffer.  The backend must assign cumulative byte offsets so that each
// buffer occupies a distinct LDS region.
//
// This test verifies that:
// 1. Each memref.alloc gets a cumulative LDS base offset
// 2. When these memrefs are used as scf.for iter_args, the offsets are
//    materialized as SGPRs and carried through the loop
// 3. vector.load from iter_arg memrefs uses the SGPR-carried offset

module {
  gpu.module @test_lds_alloc_offset {

    // CHECK-LABEL: waveasm.program @lds_alloc_double_buffer
    gpu.func @lds_alloc_double_buffer(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index

      // Two separate LDS allocations (no memref.view):
      //   bufA: 256x128 bytes = 32768 bytes at offset 0
      //   bufB: 256x128 bytes = 32768 bytes at offset 32768
      %bufA = memref.alloc() : memref<256x128xi8, 3>
      %bufB = memref.alloc() : memref<256x128xi8, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // The loop carries bufA and bufB as memref iter_args.
      // They should be resolved to SGPR offsets (0 and 32768).
      //
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<32768> -> !waveasm.sreg
      //
      // Loop with iter_args including the SGPR offsets:
      // CHECK: waveasm.loop
      %result:3 = scf.for %i = %c0 to %c7 step %c1
          iter_args(%acc = %acc_init, %curA = %bufA, %curB = %bufB)
          -> (vector<4xf32>, memref<256x128xi8, 3>, memref<256x128xi8, 3>) {

        // vector.load from iter_arg memref should use SGPR-carried offset
        // CHECK: v_add_u32
        // CHECK: ds_read_b128
        %data = vector.load %curA[%tidx, %c0] : memref<256x128xi8, 3>, vector<16xi8>

        scf.yield %acc, %curB, %curA : vector<4xf32>, memref<256x128xi8, 3>, memref<256x128xi8, 3>
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
