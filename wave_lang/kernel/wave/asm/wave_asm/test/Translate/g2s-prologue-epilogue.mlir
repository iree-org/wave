// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: amdgpu.gather_to_lds outside of scf.for (prologue and epilogue).
// In pipelined GEMM, the prologue loads the first tile into LDS before the
// loop starts, and the epilogue may process the last tile after the loop.

module {
  gpu.module @test_g2s_prologue_epilogue {

    // CHECK-LABEL: waveasm.program @g2s_prologue
    gpu.func @g2s_prologue(
        %src_buf: memref<?xf16>,
        %tidx: index {llvm.mlir.workitem_id_x}
    ) kernel {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index

      // Allocate LDS with two buffers
      %lds = memref.alloc() : memref<8192xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c4096_byte = arith.constant 4096 : index
      %viewA = memref.view %lds[%c0_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>
      %viewB = memref.view %lds[%c4096_byte][] : memref<8192xi8, 3> to memref<64x16xf16, 3>

      // === PROLOGUE: gather_to_lds before the loop ===
      // Load first tile into viewA (offset 0)
      // CHECK: waveasm.s_mov_b32_m0 %{{.*}} : !waveasm.imm<0>
      // CHECK: waveasm.buffer_load_dword_lds
      "amdgpu.gather_to_lds"(%src_buf, %tidx, %viewA, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, transferType = vector<2xf16>}> : (memref<?xf16>, index, memref<64x16xf16, 3>, index, index) -> ()

      // Load second prologue tile into viewB (offset 4096)
      // CHECK: waveasm.s_mov_b32_m0 %{{.*}} : !waveasm.imm<4096>
      // CHECK: waveasm.buffer_load_dword_lds
      "amdgpu.gather_to_lds"(%src_buf, %tidx, %viewB, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, transferType = vector<2xf16>}> : (memref<?xf16>, index, memref<64x16xf16, 3>, index, index) -> ()

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // Main loop (just yield for simplicity)
      // CHECK: waveasm.loop
      %result:3 = scf.for %i = %c0 to %c7 step %c1
          iter_args(%acc = %acc_init, %curA = %viewA, %curB = %viewB)
          -> (vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>) {
        scf.yield %acc, %curB, %curA : vector<4xf32>, memref<64x16xf16, 3>, memref<64x16xf16, 3>
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
