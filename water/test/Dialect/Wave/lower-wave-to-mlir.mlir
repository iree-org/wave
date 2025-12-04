// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir --mlir-print-local-scope --split-input-file --verify-diagnostics | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @no_hyperparams() {
    %cst = arith.constant 0.0 : f32
    // expected-error @below {{wave dialect operation with no hyperparameters provided by any ancestor}}
    wave.register %cst : vector<4xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_exp2
  func.func @lower_exp2() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.exp2
    // CHECK:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK:     %[[REG:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     math.exp2 %[[REG]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %0 = wave.register %cst : vector<4xf32>
    %1 = wave.exp2 %0 : (vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_mma_f16_f32
  func.func @lower_mma_f16_f32() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst_f16 = arith.constant 0.0 : f16
    // CHECK: %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    %lhs = wave.register %cst_f16 : vector<4xf16>

    %cst_f16_b = arith.constant 1.0 : f16
    // CHECK: %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf16>
    %rhs = wave.register %cst_f16_b : vector<4xf16>

    %cst_f32 = arith.constant 0.0 : f32
    // CHECK: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %acc = wave.register %cst_f32 : vector<4xf32>

    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma 16x16x16 %[[LHS]] * %[[RHS]] + %[[ACC]]
    // CHECK-SAME: blgp =  none
    // CHECK-SAME: vector<4xf16>, vector<4xf16>, vector<4xf32>
    %res = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_all_mmas
  func.func @lower_all_mmas() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // Common scalars
    %cst_f16_0 = arith.constant 0.0 : f16
    %cst_f16_1 = arith.constant 1.0 : f16
    %cst_bf16_0 = arith.constant 0.0 : bf16
    %cst_bf16_1 = arith.constant 1.0 : bf16
    %cst_f32_0 = arith.constant 0.0 : f32
    %cst_i8_0 = arith.constant 0 : i8
    %cst_i8_1 = arith.constant 1 : i8
    %cst_i32_0 = arith.constant 0 : i32
    %cst_f8_0 = arith.constant 0.0 : f8E4M3FNUZ
    %cst_f8_1 = arith.constant 1.0 : f8E4M3FNUZ

    // Vectors per element type
    %lhs_f16 = wave.register %cst_f16_0 : vector<4xf16>
    %rhs_f16 = wave.register %cst_f16_1 : vector<4xf16>
    %lhs_f16_w8 = wave.register %cst_f16_0 : vector<8xf16>
    %rhs_f16_w8 = wave.register %cst_f16_1 : vector<8xf16>
    %lhs_bf16 = wave.register %cst_bf16_0 : vector<8xbf16>
    %rhs_bf16 = wave.register %cst_bf16_1 : vector<8xbf16>
    %lhs_f8 = wave.register %cst_f8_0 : vector<8xf8E4M3FNUZ>
    %rhs_f8 = wave.register %cst_f8_1 : vector<8xf8E4M3FNUZ>
    %lhs_i8 = wave.register %cst_i8_0 : vector<4xi8>
    %rhs_i8 = wave.register %cst_i8_1 : vector<4xi8>
    %lhs_i8_w8 = wave.register %cst_i8_0 : vector<8xi8>
    %rhs_i8_w8 = wave.register %cst_i8_1 : vector<8xi8>

    // Accumulators
    %acc_f32_4 = wave.register %cst_f32_0 : vector<4xf32>
    %acc_f32_16 = wave.register %cst_f32_0 : vector<16xf32>
    %acc_i32_4 = wave.register %cst_i32_0 : vector<4xi32>
    %acc_i32_16 = wave.register %cst_i32_0 : vector<16xi32>

    // f16 kinds
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x16
    %0 = wave.mma %lhs_f16, %rhs_f16, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x8
    %1 = wave.mma %lhs_f16, %rhs_f16, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x8_f16>}
      : (vector<4xf16>, vector<4xf16>, vector<16xf32>) -> vector<16xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %2 = wave.mma %lhs_f16_w8, %rhs_f16_w8, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x32_k8_f16>}
      : (vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %3 = wave.mma %lhs_f16_w8, %rhs_f16_w8, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x16_k8_f16>}
      : (vector<8xf16>, vector<8xf16>, vector<16xf32>) -> vector<16xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %4 = wave.mma %lhs_f16_w8, %rhs_f16_w8, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x16_f16>}
      : (vector<8xf16>, vector<8xf16>, vector<16xf32>) -> vector<16xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %5 = wave.mma %lhs_f16_w8, %rhs_f16_w8, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x32_f16>}
      : (vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>

    // bf16 kinds
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %6 = wave.mma %lhs_bf16, %rhs_bf16, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x16_bf16>}
      : (vector<8xbf16>, vector<8xbf16>, vector<16xf32>) -> vector<16xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %7 = wave.mma %lhs_bf16, %rhs_bf16, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x32_bf16>}
      : (vector<8xbf16>, vector<8xbf16>, vector<4xf32>) -> vector<4xf32>

    // f8 kinds
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %8 = wave.mma %lhs_f8, %rhs_f8, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x32_f8>}
      : (vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %9 = wave.mma %lhs_f8, %rhs_f8, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x16_f8>}
      : (vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>) -> vector<16xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %10 = wave.mma %lhs_f8, %rhs_f8, %acc_f32_4 {kind = #wave.mma_kind<f32_16x16x32_k4_f8>}
      : (vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>) -> vector<4xf32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %11 = wave.mma %lhs_f8, %rhs_f8, %acc_f32_16 {kind = #wave.mma_kind<f32_32x32x16_k4_f8>}
      : (vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>) -> vector<16xf32>

    // i8 kinds
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x16
    %12 = wave.mma %lhs_i8, %rhs_i8, %acc_i32_4 {kind = #wave.mma_kind<i32_16x16x16_i8>}
      : (vector<4xi8>, vector<4xi8>, vector<4xi32>) -> vector<4xi32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x8
    %13 = wave.mma %lhs_i8, %rhs_i8, %acc_i32_16 {kind = #wave.mma_kind<i32_32x32x8_i8>}
      : (vector<4xi8>, vector<4xi8>, vector<16xi32>) -> vector<16xi32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 16x16x32
    %14 = wave.mma %lhs_i8_w8, %rhs_i8_w8, %acc_i32_4 {kind = #wave.mma_kind<i32_16x16x32_i8>}
      : (vector<8xi8>, vector<8xi8>, vector<4xi32>) -> vector<4xi32>
    // CHECK-NOT: wave.mma
    // CHECK: amdgpu.mfma
    // CHECK-SAME: 32x32x16
    %15 = wave.mma %lhs_i8_w8, %rhs_i8_w8, %acc_i32_16 {kind = #wave.mma_kind<i32_32x32x16_i8>}
      : (vector<8xi8>, vector<8xi8>, vector<16xi32>) -> vector<16xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_register
  func.func @lower_register() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.register
    // CHECK:     arith.constant dense<0.000000e+00> : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    wave.register %cst : vector<4xf32>

    // CHECK:     arith.constant dense<1.000000e+00> : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    wave.register %cst1 : vector<4xf32>

    // CHECK:     arith.constant dense<2> : vector<8xi32>
    %cst2 = arith.constant 2 : i32
    wave.register %cst2 : vector<8xi32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_register_with_unrealized_cast
  func.func @lower_register_with_unrealized_cast() attributes {wave.hyperparameters = #wave.hyperparameters<{Y = 10, Z = 1, BLOCK_M = 1, BLOCK_K = 2}>} {
    // CHECK-NOT: wave.register
    // CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<1x6xf32, #gpu.address_space<workgroup>>
    // CHECK:     %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<1x6xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@Y, @Z] of f32, <shared>>
    // CHECK:     "test.foo"(%[[CAST]])
    %0 = wave.allocate
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@Y, @Z] of f32, <shared>>
    "test.foo"(%0) : (!wave.tensor<[@Y, @Z] of f32, <shared>>) -> ()
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_nested_register
  func.func @lower_nested_register(%cond: i1) attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.register
    scf.if %cond {
      // CHECK:     arith.constant dense<0.000000e+00> : vector<1xf32>
      %cst = arith.constant 0.0 : f32
      wave.register %cst : vector<1xf32>
      scf.yield
    } else {
      // CHECK:     arith.constant dense<1.000000e+00> : vector<1xf32>
      %cst1 = arith.constant 1.0 : f32
      wave.register %cst1 : vector<1xf32>
      scf.yield
    }
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_add
  func.func @lower_add() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.add
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.addf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %addf = wave.add %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<2xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<2xi32>
    // CHECK:     arith.addi %[[LHSI]], %[[RHSI]] : vector<2xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : vector<2xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<2xi32>
    %addi = wave.add %lhsi, %rhsi : (vector<2xi32>, vector<2xi32>) -> vector<2xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_mul
  func.func @lower_mul() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.mul
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.mulf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %mulf = wave.mul %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<2> : vector<4xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<4xi32>
    // CHECK:     arith.muli %[[LHSI]], %[[RHSI]] : vector<4xi32>
    %cst2 = arith.constant 2 : i32
    %lhsi = wave.register %cst2 : vector<4xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<4xi32>
    %muli = wave.mul %lhsi, %rhsi : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_div
  func.func @lower_div() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.div
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.divf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %divf = wave.div %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<-2> : vector<4xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<4xi32>
    // CHECK:     arith.divsi %[[LHSI]], %[[RHSI]] : vector<4xi32>
    %cst2 = arith.constant -2 : i32
    %lhsi = wave.register %cst2 : vector<4xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<4xi32>
    %divsi = wave.div %lhsi, %rhsi : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: func.func @lower_alloc_view
func.func @lower_alloc_view() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, N=128, K= 128}>}  {
  // CHECK: %[[BUFF:.*]] = memref.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %parent = wave.allocate { distributed_shape = #wave.expr_list<[] -> (256)> }
    : !wave.tensor<[@M,@N,@K] of i8, <shared>>

  // CHECK: %[[OFF:.*]] = arith.constant 128 : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[BUFF]][%[[OFF]]][]
  // CHECK-SAME: : memref<256xi8, #gpu.address_space<workgroup>> to memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate in %parent : !wave.tensor<[@M,@N,@K] of i8, <shared>>
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>, offset = 128}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_alloc
func.func @lower_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, K= 128}>}  {
  // CHECK: memref.alloc() : memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_read
func.func @lower_read(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %0 = wave.read %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, BLOCK_N ceildiv 8, 1)}]
     : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
     // CHECK: %[[VEC:.+]] = vector.load {{.*}}[%[[ROW]], %[[COL]]] : memref<{{.*}}xf16{{.*}}>, vector<8xf16>

  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_read_non_innermost_dim
func.func @lower_read_non_innermost_dim(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %0 = wave.read %mem index [{
    // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
    // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
    // Note: BLOCK_M = 64
    // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 * 8)>()[%[[BIDX_X]], %[[TIDX_X]]]
    M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 * 8 , 8, 64),
    // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
    // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
    // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
    N : [#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, 1, 1)}]
    : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
    // CHECK: %[[PAD:.*]] = arith.constant {{.*}} : f16
    // CHECK: vector.transfer_read {{.*}}[%[[ROW]], %[[COL]]], %[[PAD]] {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<{{.*}}xf16{{.*}}>, vector<8xf16>
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @lower_read_masked
  func.func @lower_read_masked(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 100, N = 50}>} {

    %v = wave.read %mem index [{
        // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
        // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
        // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1)>()[%[[BIDX_X]], %[[TIDX_X]]]
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * BLOCK_M + T0, 1, 64),
        // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
        // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
        // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 32)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
        N : [#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + T1 * 32, 4, 1)
      }] { bounds = #wave.read_write_bounds<{
        M = #wave.expr_list<[#wave.symbol<"M">] -> (M)>,
        N = #wave.expr_list<[#wave.symbol<"N">] -> (N)>}>}
      : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<4xf16>
      // Bounds for dim 0.
      // CHECK: %[[DIM0_SIZE:.+]] = affine.apply affine_map<() -> (100)>()
      // CHECK: %[[DIM0_CMP:.+]] = arith.cmpi slt, %[[ROW]], %[[DIM0_SIZE]]
      // CHECK: %[[DIM0_CMP_VEC:.+]] = vector.broadcast %[[DIM0_CMP]] : i1 to vector<4xi1>

      // Bounds for dim 1 (vectorized).
      // CHECK: %[[DIM1_SIZE:.+]] = affine.apply affine_map<() -> (50)>()
      // CHECK: %[[IOTA:.+]] = vector.step : vector<4xindex>
      // CHECK: %[[COL_VEC_BASE:.+]] = vector.broadcast %[[COL]] : index to vector<4xindex>
      // CHECK: %[[COL_VEC:.+]] = arith.addi %[[COL_VEC_BASE]], %[[IOTA]]
      // CHECK: %[[DIM1_SIZE_VEC:.+]] = vector.broadcast %[[DIM1_SIZE]] : index to vector<4xindex>
      // CHECK: %[[DIM1_CMP_VEC:.+]] = arith.cmpi slt, %[[COL_VEC]], %[[DIM1_SIZE_VEC]]

      // Masked load.
      // CHECK: %[[MASK:.+]] = arith.andi %[[DIM0_CMP_VEC]], %[[DIM1_CMP_VEC]]
      // CHECK: %[[PASSTHRU:.+]] = arith.constant dense<0.000000e+00> : vector<4xf16>
      // CHECK: vector.maskedload %{{.*}}[%[[ROW]], %[[COL]]], %[[MASK]], %[[PASSTHRU]]

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @lower_read_masked_non_innermost_dim
  func.func @lower_read_masked_non_innermost_dim(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 100, N = 50}>} {
    %v = wave.read %mem index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (WG0 * BLOCK_M + T0, 8, 64),
        N : [#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + T1 * 32, 1, 1)
      }] { bounds = #wave.read_write_bounds<{
        M = #wave.expr_list<[#wave.symbol<"M">] -> (M)>,
        N = #wave.expr_list<[#wave.symbol<"N">] -> (N)>}>}
      : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
      // CHECK: %[[MASK:.+]] = arith.andi {{.*}}, {{.*}}
      // CHECK: %[[PAD:.*]] = arith.constant {{.*}} : f16
      // CHECK: vector.transfer_read {{.*}}[{{.*}}, {{.*}}], %[[PAD]], %[[MASK]] {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<{{.*}}xf16{{.*}}>, vector<8xf16>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_write
func.func @lower_write(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %cst = arith.constant 0.0 : f16
  %reg = wave.register %cst : vector<8xf16>
  wave.write %reg, %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, BLOCK_N ceildiv 8, 1)}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.store {{.*}}[%[[ROW]], %[[COL]]] : memref<{{.*}}xf16{{.*}}>, vector<8xf16>
     // CHECK-NOT: vector.transfer_write
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
// CHECK-LABEL: @lower_write_non_innermost
func.func @lower_write_non_innermost(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %cst = arith.constant 0.0 : f16
  %reg = wave.register %cst : vector<8xf16>
  wave.write %reg, %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 8, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, 1, 1)}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.transfer_write {{.*}}, {{.*}}[{{.*}}, {{.*}}] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<8xf16>, memref<{{.*}}xf16{{.*}}>

  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_float_extension
  func.func @lower_cast_float_extension() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<0.000000e+00> : vector<4xf16>
    // CHECK:     arith.extf %[[INPUT]] : vector<4xf16> to vector<4xf32>
    %cst = arith.constant 0.0 : f16
    %input = wave.register %cst : vector<4xf16>
    %result = wave.cast %input : vector<4xf16> to vector<4xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_float_truncation
  func.func @lower_cast_float_truncation() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
    // CHECK:     arith.truncf %[[INPUT]] : vector<8xf32> to vector<8xf16>
    %cst = arith.constant 1.0 : f32
    %input = wave.register %cst : vector<8xf32>
    %result = wave.cast %input : vector<8xf32> to vector<8xf16>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_integer_extension
  func.func @lower_cast_integer_extension() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<42> : vector<4xi8>
    // CHECK:     arith.extsi %[[INPUT]] : vector<4xi8> to vector<4xi32>
    %cst = arith.constant 42 : i8
    %input = wave.register %cst : vector<4xi8>
    %result = wave.cast %input : vector<4xi8> to vector<4xi32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_integer_truncation
  func.func @lower_cast_integer_truncation() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<1000> : vector<4xi32>
    // CHECK:     arith.trunci %[[INPUT]] : vector<4xi32> to vector<4xi16>
    %cst = arith.constant 1000 : i32
    %input = wave.register %cst : vector<4xi32>
    %result = wave.cast %input : vector<4xi32> to vector<4xi16>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_float_to_integer
  func.func @lower_cast_float_to_integer() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<3.140000e+00> : vector<4xf32>
    // CHECK:     arith.fptosi %[[INPUT]] : vector<4xf32> to vector<4xi32>
    %cst = arith.constant 3.14 : f32
    %input = wave.register %cst : vector<4xf32>
    %result = wave.cast %input : vector<4xf32> to vector<4xi32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_integer_to_float
  func.func @lower_cast_integer_to_float() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.cast
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<-5> : vector<8xi32>
    // CHECK:     arith.sitofp %[[INPUT]] : vector<8xi32> to vector<8xf32>
    %cst = arith.constant -5 : i32
    %input = wave.register %cst : vector<8xi32>
    %result = wave.cast %input : vector<8xi32> to vector<8xf32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_cast_mixed_types
  func.func @lower_cast_mixed_types() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // Test f16 -> f32 extension
    // CHECK:     %[[F16:.*]] = arith.constant dense<1.000000e+00> : vector<4xf16>
    %cst_f16 = arith.constant 1.0 : f16
    %f16_reg = wave.register %cst_f16 : vector<4xf16>
    // CHECK:     %[[F16_TO_F32:.*]] = arith.extf %[[F16]] : vector<4xf16> to vector<4xf32>
    %f16_to_f32 = wave.cast %f16_reg : vector<4xf16> to vector<4xf32>

    // Test f32 -> i32 conversion
    // CHECK:     arith.fptosi %[[F16_TO_F32]] : vector<4xf32> to vector<4xi32>
    %f32_to_i32 = wave.cast %f16_to_f32 : vector<4xf32> to vector<4xi32>

    // Test i32 -> i64 extension
    // CHECK:     arith.extsi {{.*}} : vector<4xi32> to vector<4xi64>
    %i32_to_i64 = wave.cast %f32_to_i32 : vector<4xi32> to vector<4xi64>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_extract_slice_constants
  func.func @lower_extract_slice_constants() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    %cst = arith.constant 0.0 : f32
    %input = wave.register %cst : vector<16xf32>

    // CHECK-NOT: wave.extract_slice
    // CHECK:     vector.extract_strided_slice %[[INPUT]] {offsets = [4], sizes = [8], strides = [1]} : vector<16xf32> to vector<8xf32>
    %result = wave.extract_slice %input {
      offset = #wave.expr_list<[] -> (4)>,
      size = #wave.expr_list<[] -> (8)>,
      stride = #wave.expr_list<[] -> (1)>
    } : (vector<16xf32>) -> vector<8xf32>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_iterate
  func.func @lower_iterate() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 128, BLOCK_K = 32, M = 64}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
      : !wave.tensor<[@M] of f32, <shared>>

    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (memref<64xf32, #gpu.address_space<workgroup>>) {
    // CHECK:       scf.yield %{{.*}} : memref<64xf32, #gpu.address_space<workgroup>>
    // CHECK:     }
    %result = wave.iterate @K iter_args(%alloc) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> !wave.tensor<[@M] of f32, <shared>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: func.func @lower_iterate_with_operations
  func.func @lower_iterate_with_operations() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, BLOCK_K = 16, M = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
      : !wave.tensor<[@M] of f32, <shared>>

    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (memref<32xf32, #gpu.address_space<workgroup>>) {
    // CHECK:       %[[TEMP:.*]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    // CHECK:       scf.yield %{{.*}} : memref<32xf32, #gpu.address_space<workgroup>>
    // CHECK:     }
    %result = wave.iterate @K iter_args(%alloc) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      %temp = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
        : !wave.tensor<[@M] of f32, <shared>>
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> !wave.tensor<[@M] of f32, <shared>>

    return
  }
}
