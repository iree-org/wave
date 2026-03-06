// RUN: waveasm-translate --waveasm-linear-scan --emit-assembly %s 2>&1 | FileCheck %s
//
// Test: Full assembly output for scaled MFMA with double-buffered LDS allocs.
//
// This verifies the end-to-end pipeline from MLIR to AMDGCN assembly for the
// MXFP4 pattern: direct memref.alloc (no memref.view), memref iter_args for
// double-buffering, vector.bitcast + vector.extract for scale factors, and
// amdgpu.scaled_mfma for the scaled matrix multiply.
//
// Expected assembly features:
// 1. s_mov_b32 for LDS offset SGPR initialization
// 2. v_add_u32 with SGPR offset for LDS address computation
// 3. ds_read for data and scale loads
// 4. v_mfma_scale_f32_16x16x128_f8f6f4 with cbsz/blgp attributes
// 5. s_mov_b32 SGPR rotation (swap) at loop tail

module {
  gpu.module @test_scaled_mfma_double_buffer {

    // CHECK-LABEL: scaled_mfma_dbuf_asm:
    gpu.func @scaled_mfma_dbuf_asm(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index

      // Two data buffers + two scale buffers for double-buffering
      %data0 = memref.alloc() : memref<16x128xi8, 3>
      %data1 = memref.alloc() : memref<16x128xi8, 3>
      %scale0 = memref.alloc() : memref<16x8xi8, 3>
      %scale1 = memref.alloc() : memref<16x8xi8, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // SGPR init for LDS offsets
      // CHECK-DAG: s_mov_b32 s{{[0-9]+}}, 0
      // CHECK-DAG: s_mov_b32 s{{[0-9]+}}, 2048
      // CHECK-DAG: s_mov_b32 s{{[0-9]+}}, 4096
      // CHECK-DAG: s_mov_b32 s{{[0-9]+}}, 4224

      // CHECK: L_loop_0:
      %result:5 = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %acc_init,
                    %curData = %data0, %nextData = %data1,
                    %curScale = %scale0, %nextScale = %scale1)
          -> (vector<4xf32>,
              memref<16x128xi8, 3>, memref<16x128xi8, 3>,
              memref<16x8xi8, 3>, memref<16x8xi8, 3>) {

        // Data load with SGPR-carried LDS offset
        // CHECK: v_add_u32
        // CHECK: ds_read_b128
        %raw_data = vector.load %curData[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %data_a = "vector.bitcast"(%raw_data) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        %raw_data2 = vector.load %curData[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %data_b = "vector.bitcast"(%raw_data2) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        // Scale load
        // CHECK: ds_read_u8
        %raw_sa = vector.load %curScale[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %svec_a = "vector.bitcast"(%raw_sa) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %sa = "vector.extract"(%svec_a) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        %raw_sb = vector.load %curScale[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %svec_b = "vector.bitcast"(%raw_sb) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %sb = "vector.extract"(%svec_b) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        // Scaled MFMA instruction
        // CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}} cbsz:4 blgp:4
        %new_acc = "amdgpu.scaled_mfma"(%data_a, %data_b, %acc, %sa, %sb) <{
          k = 128 : i32, m = 16 : i32, n = 16 : i32,
          scalesIdxA = 0 : i32, scalesIdxB = 0 : i32
        }> : (vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<4xf32>, f8E8M0FNU, f8E8M0FNU) -> vector<4xf32>

        scf.yield %new_acc, %nextData, %curData, %nextScale, %curScale
            : vector<4xf32>,
              memref<16x128xi8, 3>, memref<16x128xi8, 3>,
              memref<16x8xi8, 3>, memref<16x8xi8, 3>
      }

      // SGPR rotation for 2 swap pairs (data + scale)
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
      // CHECK: s_cbranch_scc1 L_loop_0

      // CHECK: s_endpgm
      gpu.return
    }
  }
}
