// RUN: waveasm-translate --target=gfx950 --waveasm-linear-scan --emit-assembly %s 2>&1 | FileCheck %s
//
// Test: bf16 store from AGPR accumulator inserts v_accvgpr_read_b32.
//
// On gfx950, MFMA accumulators live in AGPRs. When the loop result is
// truncated to bf16 (arith.truncf) and stored to a bf16 memref, the
// translation must move the data from AGPR to VGPR via v_accvgpr_read_b32
// before the v_cvt_pk_bf16_f32 conversion, since VALU instructions cannot
// read AGPRs directly.
//
// This test verifies the epilogue sequence:
//   1. v_accvgpr_read_b32 (AGPR -> VGPR)
//   2. v_cvt_pk_bf16_f32  (f32 pair -> packed bf16)
//   3. buffer_store         (write packed bf16 to global memory)

module {
  gpu.module @test_agpr_bf16_store {

    // CHECK-LABEL: agpr_bf16_store:
    gpu.func @agpr_bf16_store(
        %out: memref<1024xbf16> {llvm.noalias},
        %tidx: index {llvm.mlir.workitem_id_x}
    ) kernel {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index

      %data_buf = memref.alloc() : memref<16x128xi8, 3>
      %scale_buf = memref.alloc() : memref<16x8xi8, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // Loop with scaled MFMA — accumulator lives in AGPRs on gfx950.
      // CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 a[{{.*}}]
      %result = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %acc_init) -> (vector<4xf32>) {

        %raw_a = vector.load %data_buf[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %a = "vector.bitcast"(%raw_a) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        %raw_b = vector.load %data_buf[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %b = "vector.bitcast"(%raw_b) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        %raw_sa = vector.load %scale_buf[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %svec_a = "vector.bitcast"(%raw_sa) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %sa = "vector.extract"(%svec_a) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        %raw_sb = vector.load %scale_buf[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %svec_b = "vector.bitcast"(%raw_sb) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %sb = "vector.extract"(%svec_b) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        %new_acc = "amdgpu.scaled_mfma"(%a, %b, %acc, %sa, %sb) <{
          k = 128 : i32, m = 16 : i32, n = 16 : i32,
          scalesIdxA = 0 : i32, scalesIdxB = 0 : i32
        }> : (vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<4xf32>, f8E8M0FNU, f8E8M0FNU) -> vector<4xf32>

        scf.yield %new_acc : vector<4xf32>
      }

      // Epilogue: truncate f32 accumulator to bf16 and store.
      // The AGPR data must be moved to VGPR before the VALU bf16 conversion.
      %bf16_result = arith.truncf %result : vector<4xf32> to vector<4xbf16>

      // CHECK: v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
      // CHECK: v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
      // CHECK: v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
      // CHECK: v_accvgpr_read_b32 v{{[0-9]+}}, a{{[0-9]+}}
      // CHECK: v_cvt_pk_bf16_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
      // CHECK: v_cvt_pk_bf16_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
      // CHECK: buffer_store_dwordx2
      vector.store %bf16_result, %out[%tidx] : memref<1024xbf16>, vector<4xbf16>

      // CHECK: s_endpgm
      gpu.return
    }
  }
}
