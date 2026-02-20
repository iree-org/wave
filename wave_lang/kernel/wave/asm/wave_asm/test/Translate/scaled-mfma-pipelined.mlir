// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: amdgpu.scaled_mfma with pipelined double-buffer iter_args.
//
// This exercises the MXFP4 GEMM pattern where:
// 1. Data and scale buffers are separate memref.alloc (not memref.view)
// 2. scf.for carries them as memref iter_args for double-buffering
// 3. vector.load reads data + scales from the iter_arg memrefs
// 4. vector.bitcast reinterprets i8 data as f4E2M1FN and f8E8M0FNU
// 5. vector.extract extracts scalar scales
// 6. amdgpu.scaled_mfma consumes data + scales + accumulator
//
// The backend must:
// - Assign cumulative LDS offsets to each memref.alloc
// - Materialize offsets as SGPRs for iter_args
// - Map all scaled_mfma operands through the value mapper

module {
  gpu.module @test_scaled_mfma_pipelined {

    // CHECK-LABEL: waveasm.program @scaled_mfma_dbuf
    gpu.func @scaled_mfma_dbuf(%tidx: index {llvm.mlir.workitem_id_x}) kernel {
      %c0 = arith.constant 0 : index
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index

      // 4 LDS allocations: data_A0, data_A1 (ping-pong), scale_A0, scale_A1
      //   data_A0:  16x128 bytes = 2048 bytes at offset 0
      //   data_A1:  16x128 bytes = 2048 bytes at offset 2048
      //   scale_A0: 16x8 bytes   = 128 bytes  at offset 4096
      //   scale_A1: 16x8 bytes   = 128 bytes  at offset 4224
      %data0 = memref.alloc() : memref<16x128xi8, 3>
      %data1 = memref.alloc() : memref<16x128xi8, 3>
      %scale0 = memref.alloc() : memref<16x8xi8, 3>
      %scale1 = memref.alloc() : memref<16x8xi8, 3>

      %acc_init = arith.constant dense<0.0> : vector<4xf32>

      // SGPR materialization for all 4 LDS offsets:
      // CHECK-DAG: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<0> -> !waveasm.sreg
      // CHECK-DAG: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<2048> -> !waveasm.sreg
      // CHECK-DAG: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<4096> -> !waveasm.sreg
      // CHECK-DAG: waveasm.s_mov_b32 %{{.*}} : !waveasm.imm<4224> -> !waveasm.sreg
      //
      // CHECK: waveasm.loop
      %result:5 = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %acc_init,
                    %curData = %data0, %nextData = %data1,
                    %curScale = %scale0, %nextScale = %scale1)
          -> (vector<4xf32>,
              memref<16x128xi8, 3>, memref<16x128xi8, 3>,
              memref<16x8xi8, 3>, memref<16x8xi8, 3>) {

        // Load data (16xi8 -> bitcast to 32xf4E2M1FN)
        // CHECK: ds_read_b128
        %raw_data_a = vector.load %curData[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %data_a = "vector.bitcast"(%raw_data_a) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        // Load same data as srcB (simplified: reuse same buffer for both A and B)
        %raw_data_b = vector.load %curData[%tidx, %c0] : memref<16x128xi8, 3>, vector<16xi8>
        %data_b = "vector.bitcast"(%raw_data_b) : (vector<16xi8>) -> vector<32xf4E2M1FN>

        // Load scales (1xi8 -> bitcast to 1xf8E8M0FNU -> extract scalar)
        // CHECK: ds_read_u8
        %raw_scale_a = vector.load %curScale[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %scale_vec_a = "vector.bitcast"(%raw_scale_a) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %scale_a = "vector.extract"(%scale_vec_a) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        %raw_scale_b = vector.load %curScale[%tidx, %c0] : memref<16x8xi8, 3>, vector<1xi8>
        %scale_vec_b = "vector.bitcast"(%raw_scale_b) : (vector<1xi8>) -> vector<1xf8E8M0FNU>
        %scale_b = "vector.extract"(%scale_vec_b) <{static_position = array<i64: 0>}> : (vector<1xf8E8M0FNU>) -> f8E8M0FNU

        // Scaled MFMA: all 5 operands should be mapped
        // CHECK: waveasm.v_mfma_scale_f32_16x16x128_f8f6f4
        %new_acc = "amdgpu.scaled_mfma"(%data_a, %data_b, %acc, %scale_a, %scale_b) <{
          k = 128 : i32, m = 16 : i32, n = 16 : i32,
          scalesIdxA = 0 : i32, scalesIdxB = 0 : i32
        }> : (vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<4xf32>, f8E8M0FNU, f8E8M0FNU) -> vector<4xf32>

        // Swap ping-pong buffers
        scf.yield %new_acc, %nextData, %curData, %nextScale, %curScale
            : vector<4xf32>,
              memref<16x128xi8, 3>, memref<16x128xi8, 3>,
              memref<16x8xi8, 3>, memref<16x8xi8, 3>
      }

      // CHECK: waveasm.s_endpgm
      gpu.return
    }
  }
}
