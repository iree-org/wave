// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s 2>&1 | FileCheck %s
//
// End-to-end integration test: pipelined GEMM with g2s double-buffering.
//
// This test exercises the complete pipeline from MLIR to AMDGCN assembly:
// 1. Prologue: SRD setup, first gather_to_lds loads
// 2. Loop: interleaved ds_read (from "current" buffer via SGPR offset)
//          + MFMA, with memref iter_args for ping-pong swap
// 3. SGPR rotation (3-instruction swap per pair) before s_cbranch_scc1
// 4. Epilogue: s_endpgm

module {
  gpu.module @test_pipelined_gemm_g2s {

    // CHECK-LABEL: pipelined_gemm_g2s:
    gpu.func @pipelined_gemm_g2s(
        %src_a: memref<?xf16>,
        %src_b: memref<?xf16>,
        %tidx: index {llvm.mlir.workitem_id_x}
    ) kernel {
      %c0 = arith.constant 0 : index
      %c15 = arith.constant 15 : index
      %c1 = arith.constant 1 : index

      // LDS allocation: two buffers for A, two for B
      %lds = memref.alloc() : memref<8192xi8, 3>
      %c0_byte = arith.constant 0 : index
      %c2048_byte = arith.constant 2048 : index
      %c4096_byte = arith.constant 4096 : index
      %c6144_byte = arith.constant 6144 : index
      %lds_a0 = memref.view %lds[%c0_byte][] : memref<8192xi8, 3> to memref<32x16xf16, 3>
      %lds_a1 = memref.view %lds[%c2048_byte][] : memref<8192xi8, 3> to memref<32x16xf16, 3>
      %lds_b0 = memref.view %lds[%c4096_byte][] : memref<8192xi8, 3> to memref<32x16xf16, 3>
      %lds_b1 = memref.view %lds[%c6144_byte][] : memref<8192xi8, 3> to memref<32x16xf16, 3>

      // === PROLOGUE: load first tile into buffer 0 ===
      // CHECK: s_mov_b32 m0, 0
      // CHECK: buffer_load_dword {{.*}} offen lds
      "amdgpu.gather_to_lds"(%src_a, %tidx, %lds_a0, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, transferType = vector<2xf16>}> : (memref<?xf16>, index, memref<32x16xf16, 3>, index, index) -> ()

      // CHECK: s_mov_b32 m0, 4096
      // CHECK: buffer_load_dword {{.*}} offen lds
      "amdgpu.gather_to_lds"(%src_b, %tidx, %lds_b0, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>, transferType = vector<2xf16>}> : (memref<?xf16>, index, memref<32x16xf16, 3>, index, index) -> ()

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // === MAIN LOOP with double-buffered g2s ===
      //
      // CHECK: L_loop_0:
      //
      // ds_read from current A buffer (SGPR-carried offset)
      // CHECK: ds_read_b64
      //
      // ds_read from current B buffer (SGPR-carried offset)
      // CHECK: ds_read_b64
      //
      // MFMA
      // CHECK: v_mfma_f32_16x16x16_f16
      //
      // SGPR rotation: two independent swap pairs (A pair + B pair)
      // First swap pair (A buffers): tmp=s_a, s_a=s_a', s_a'=tmp
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // Second swap pair (B buffers): tmp=s_b, s_b=s_b', s_b'=tmp
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK-NEXT: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      //
      // Branch back
      // CHECK-NEXT: s_cbranch_scc1 L_loop_0
      %result:5 = scf.for %i = %c0 to %c15 step %c1
          iter_args(%acc = %acc_init, %cur_a = %lds_a0, %next_a = %lds_a1,
                    %cur_b = %lds_b0, %next_b = %lds_b1)
          -> (vector<4xf32>, memref<32x16xf16, 3>, memref<32x16xf16, 3>,
              memref<32x16xf16, 3>, memref<32x16xf16, 3>) {

        // Read from current buffers
        %data_a = vector.load %cur_a[%tidx, %c0] : memref<32x16xf16, 3>, vector<4xf16>
        %data_b = vector.load %cur_b[%tidx, %c0] : memref<32x16xf16, 3>, vector<4xf16>

        // MFMA 16x16x16 f16: src is vector<4xf16>, acc is vector<4xf32>
        %new_acc = "amdgpu.mfma"(%data_b, %data_a, %acc) <{abid = 0 : i32, blgp = #amdgpu<mfma_perm_b none>, blocks = 1 : i32, cbsz = 0 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32}> : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>

        // Swap: next iteration reads from "next", writes to "current"
        scf.yield %new_acc, %next_a, %cur_a, %next_b, %cur_b
            : vector<4xf32>, memref<32x16xf16, 3>, memref<32x16xf16, 3>,
              memref<32x16xf16, 3>, memref<32x16xf16, 3>
      }

      // CHECK: s_endpgm
      gpu.return
    }
  }
}
