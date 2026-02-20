// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir -lower-normalform-module --mlir-print-local-scope --split-input-file --verify-diagnostics | FileCheck %s

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  func.func @no_hyperparams() {
    %cst = arith.constant 0.0 : f32
    // expected-error @below {{wave dialect operation with no hyperparameters provided by any ancestor}}
    wave.register %cst : vector<4xf32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_reciprocal
  func.func @lower_reciprocal() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.reciprocal
    // CHECK:     %[[CST:.*]] = arith.constant 2.000000e+00 : f32
    // CHECK:     %[[REG:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    // CHECK:     %[[ONE:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.divf %[[ONE]], %[[REG]] : vector<4xf32>
    %cst = arith.constant 2.0 : f32
    %0 = wave.register %cst : vector<4xf32>
    %1 = wave.reciprocal %0 : (vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_allocate_memref
  // Test lowering when wave.allocate already has MemRefType result
  // (after ResolveDistributedAllocations pass).
  func.func @lower_allocate_memref() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1, BLOCK_K = 2}>} {
    // CHECK:     %[[ALLOC:.*]] = memref.alloc() : memref<1x6xf32, #gpu.address_space<workgroup>>
    // CHECK:     "test.foo"(%[[ALLOC]])
    %0 = wave.allocate
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    : memref<1x6xf32, #gpu.address_space<workgroup>>
    "test.foo"(%0) : (memref<1x6xf32, #gpu.address_space<workgroup>>) -> ()
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_max
  func.func @lower_max() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.max
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.maximumf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %maxf = wave.max %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_min
  func.func @lower_min() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.min
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.minimumf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %minf = wave.min %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_sub
  func.func @lower_sub() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.sub
    // CHECK:     %[[LHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
    // CHECK:     arith.subf %[[LHS]], %[[RHS]] : vector<4xf32>
    %cst = arith.constant 1.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 2.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %subf = wave.sub %lhs, %rhs : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

    // CHECK:     %[[LHSI:.*]] = arith.constant dense<5> : vector<2xi32>
    // CHECK:     %[[RHSI:.*]] = arith.constant dense<3> : vector<2xi32>
    // CHECK:     arith.subi %[[LHSI]], %[[RHSI]] : vector<2xi32>
    %cst2 = arith.constant 5 : i32
    %lhsi = wave.register %cst2 : vector<2xi32>
    %cst3 = arith.constant 3 : i32
    %rhsi = wave.register %cst3 : vector<2xi32>
    %subi = wave.sub %lhsi, %rhsi : (vector<2xi32>, vector<2xi32>) -> vector<2xi32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_apply_expr
  func.func @lower_apply_expr() -> vector<4xi32> attributes {wave.hyperparameters = #wave.hyperparameters<{N = 10}>} {
    %cst = arith.constant 42 : i32
    %input = wave.register %cst : vector<4xi32>
    %cst3 = arith.constant 3 : i32
    %input2 = wave.register %cst3 : vector<4xi32>
    // CHECK: %[[CST_42:.+]] = arith.constant dense<42> : vector<4xi32>
    // CHECK: %[[CST_3:.+]] = arith.constant dense<3> : vector<4xi32>
    // CHECK: %[[CST_10:.+]] = arith.constant dense<10> : vector<4xi32>
    // CHECK: %[[CST_2:.+]] = arith.constant dense<2> : vector<4xi32>
    // CHECK: %[[MUL:.+]] = arith.muli %[[CST_10]], %[[CST_2]] overflow<nsw, nuw> : vector<4xi32>
    // CHECK: %[[ADD:.+]] = arith.addi %[[CST_42]], %[[MUL]] overflow<nsw, nuw> : vector<4xi32>
    // CHECK: %[[CST_N1:.+]] = arith.constant dense<-1> : vector<4xi32>
    // CHECK: %[[MUL2:.+]] = arith.muli %[[CST_3]], %[[CST_N1]] overflow<nsw, nuw> : vector<4xi32>
    // CHECK: %[[ADD2:.+]] = arith.addi %[[ADD]], %[[MUL2]] overflow<nsw, nuw> : vector<4xi32>
    // CHECK: %[[CST_1:.+]] = arith.constant dense<1> : vector<4xi32>
    // CHECK: %[[ADD3:.+]] = arith.addi %[[ADD2]], %[[CST_1]] overflow<nsw, nuw> : vector<4xi32>
    %result = wave.apply_expr(%input, %input2)
      <[#wave.operand<1>, #wave.symbol<"N">, #wave.operand<0>] ->
      (_Operand_0 + 1 + 2 * N - _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    return %result : vector<4xi32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_apply_comparisons
  func.func @lower_apply_comparisons() -> vector<4xi1> attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 42 : i32
    %input = wave.register %cst : vector<4xi32>
    %cst2 = arith.constant 2 : i32
    %input2 = wave.register %cst2 : vector<4xi32>
    // CHECK: %[[CST_42:.+]] = arith.constant dense<42> : vector<4xi32>
    // CHECK: %[[CST_2:.+]] = arith.constant dense<2> : vector<4xi32>
    // CHECK: %[[LT:.+]] = arith.cmpi slt, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[GT:.+]] = arith.cmpi sgt, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[LE:.+]] = arith.cmpi sle, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[EXT_LE:.+]] = arith.extui %[[LE]] : vector<4xi1> to vector<4xi32>
    // CHECK: %[[GE:.+]] = arith.cmpi sge, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[EQ:.+]] = arith.cmpi eq, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[EXT_EQ:.+]] = arith.extui %[[EQ]] : vector<4xi1> to vector<4xi64>
    // CHECK: %[[NE:.+]] = arith.cmpi ne, %[[CST_42]], %[[CST_2]] : vector<4xi32>
    %result1 = wave.apply_expr(%input, %input2) lt <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi1>
    %result2 = wave.apply_expr(%input, %input2) gt <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi1>
    %result3 = wave.apply_expr(%input, %input2) le <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %result4 = wave.apply_expr(%input, %input2) ge <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi1>
    %result5 = wave.apply_expr(%input, %input2) eq <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi64>
    %result6 = wave.apply_expr(%input, %input2) ne <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi1>
    return %result1 : vector<4xi1>
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_apply_expr_minmax
  func.func @lower_apply_expr_minmax() -> vector<4xi32> attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 42 : i32
    %input = wave.register %cst : vector<4xi32>
    %cst2 = arith.constant 2 : i32
    %input2 = wave.register %cst2 : vector<4xi32>
    // CHECK: %[[CST_42:.+]] = arith.constant dense<42> : vector<4xi32>
    // CHECK: %[[CST_2:.+]] = arith.constant dense<2> : vector<4xi32>
    // CHECK: %[[MIN:.+]] = arith.minsi %[[CST_42]], %[[CST_2]] : vector<4xi32>
    // CHECK: %[[MAX:.+]] = arith.maxsi %[[CST_42]], %[[CST_2]] : vector<4xi32>
    %result1 = wave.apply_expr(%input, %input2) min <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %result2 = wave.apply_expr(%input, %input2) max <[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0, _Operand_1)> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    return %result1 : vector<4xi32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_apply_expr_div
  func.func @lower_apply_expr_div() -> vector<4xi64> attributes {wave.hyperparameters = #wave.hyperparameters<{A = 15, B = 4}>} {
    // CHECK: %[[CST_A:.+]] = arith.constant dense<15> : vector<4xi64>
    // CHECK: %[[CST_B:.+]] = arith.constant dense<4> : vector<4xi64>
    // CHECK: %[[FLOORDIV:.+]] = arith.floordivsi %[[CST_A]], %[[CST_B]] : vector<4xi64>
    // CHECK: %[[CEILDIV:.+]] = arith.ceildivsi %{{.*}}, %{{.*}} : vector<4xi64>
    // CHECK: %[[ADD:.+]] = arith.addi %[[FLOORDIV]], %[[CEILDIV]] overflow<nsw, nuw> : vector<4xi64>
    // CHECK: %[[REM:.+]] = arith.remsi %{{.*}}, %{{.*}} : vector<4xi64>
    // CHECK: %[[ADD2:.+]] = arith.addi %[[ADD]], %[[REM]] overflow<nsw, nuw> : vector<4xi64>
    // CHECK: return %[[ADD2]] : vector<4xi64>
    %result = wave.apply_expr()
      <[#wave.symbol<"A">, #wave.symbol<"B">] ->
      ((A floordiv B) + (A ceildiv B) + (A mod B))>
      : () -> vector<4xi64>
    return %result : vector<4xi64>
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_select
  func.func @lower_select() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-NOT: wave.select
    // CHECK:     %[[COND:.*]] = arith.constant dense<true> : vector<4xi1>
    // CHECK:     %[[LHS:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    // CHECK:     %[[RHS:.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
    // CHECK:     arith.select %[[COND]], %[[LHS]], %[[RHS]] : vector<4xi1>, vector<4xf32>
    %cond_a = arith.constant 1 : i1
    %cond = wave.register %cond_a : vector<4xi1>
    %cst = arith.constant 0.0 : f32
    %lhs = wave.register %cst : vector<4xf32>
    %cst1 = arith.constant 1.0 : f32
    %rhs = wave.register %cst1 : vector<4xf32>
    %select = wave.select %cond, %lhs, %rhs : (vector<4xi1>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    return
  }
}
// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: func.func @lower_alloc_view
func.func @lower_alloc_view() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28}>}  {
  // CHECK: %[[BUFF:.*]] = memref.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %parent = wave.allocate { distributed_shape = #wave.expr_list<[] -> (256)> }
    : memref<256xi8, #gpu.address_space<workgroup>>

  // CHECK: %[[OFF:.*]] = arith.constant 128 : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[BUFF]][%[[OFF]]][]
  // CHECK-SAME: : memref<256xi8, #gpu.address_space<workgroup>> to memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate in %parent : memref<256xi8, #gpu.address_space<workgroup>>
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>, offset = 128}
    : memref<4x32xbf16, #gpu.address_space<workgroup>>
  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: @lower_alloc
func.func @lower_alloc() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28}>}  {
  // CHECK: memref.alloc() : memref<4x32xbf16, #gpu.address_space<workgroup>>
  %buf = wave.allocate
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>}
    : memref<4x32xbf16, #gpu.address_space<workgroup>>
  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: @lower_read
func.func @lower_read(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %0 = wave.read %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64)>,
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
      N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, BLOCK_N ceildiv 8, 1)>}]
     : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
     // CHECK: %[[VEC:.+]] = vector.load {{.*}}[%[[ROW]], %[[COL]]] : memref<{{.*}}xf16{{.*}}>, vector<8xf16>

  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: @lower_read_non_innermost_dim
func.func @lower_read_non_innermost_dim(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %0 = wave.read %mem index [{
    // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
    // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
    // Note: BLOCK_M = 64
    // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 * 8)>()[%[[BIDX_X]], %[[TIDX_X]]]
    M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 * 8 , 8, 64)>,
    // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
    // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
    // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
    N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, 1, 1)>}]
    : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>
    // CHECK: %[[PAD:.*]] = arith.constant {{.*}} : f16
    // CHECK: vector.transfer_read {{.*}}[%[[ROW]], %[[COL]]], %[[PAD]] {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<{{.*}}xf16{{.*}}>, vector<8xf16>
  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_read_masked
  func.func @lower_read_masked(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 100, N = 50}>} {

    %v = wave.read %mem index [{
        // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
        // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
        // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1)>()[%[[BIDX_X]], %[[TIDX_X]]]
        M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0, 1, 64)>,
        // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
        // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
        // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 32)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
        N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + T1 * 32, 4, 1)>
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_read_masked_non_innermost_dim
  func.func @lower_read_masked_non_innermost_dim(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 100, N = 50}>} {
    %v = wave.read %mem index [{
        M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0, 8, 64)>,
        N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + T1 * 32, 1, 1)>
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

// Sparse bounds: only M needs masking, N is fully tiled (no entry).
// The mask should only check the M dimension — no arith.andi.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_read_sparse_bounds
  func.func @lower_read_sparse_bounds(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 100, N = 64}>} {
    %v = wave.read %mem index [{
        // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
        // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
        // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1)>()[%[[BIDX_X]], %[[TIDX_X]]]
        M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0, 1, 64)>,
        N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + T1 * 32, 4, 1)>
      }] { bounds = #wave.read_write_bounds<{
        M = #wave.expr_list<[#wave.symbol<"M">] -> (M)>}>}
      : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<4xf16>
      // Only M dimension produces a mask — no andi needed.
      // CHECK: %[[DIM0_SIZE:.+]] = affine.apply affine_map<() -> (100)>()
      // CHECK: %[[DIM0_CMP:.+]] = arith.cmpi slt, %[[ROW]], %[[DIM0_SIZE]]
      // CHECK: %[[MASK:.+]] = vector.broadcast %[[DIM0_CMP]] : i1 to vector<4xi1>
      // CHECK-NOT: arith.andi
      // CHECK: %[[CST0:.+]] = arith.constant dense<0.000000e+00> : vector<4xf16>
      // CHECK: vector.maskedload %{{.*}}[%[[ROW]], %{{.*}}], %[[MASK]], %[[CST0]]
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @read_with_vector_result
  func.func @read_with_vector_result(%mem: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 128}>} {
    // Test ReadOp lowering when result is already a vector type
    // (simulates after PropagateElementsPerThread pass)
    %0 = wave.read %mem index [{
        M : <[#wave.index_symbol<WG0>] -> (WG0, 1, 1)>,
        N : <[#wave.index_symbol<T0>] -> (T0, 8, 1)>
      }]
      : (!wave.tensor<[@M, @N] of f16, <global>>) -> vector<8xf16>

    // CHECK: %[[READ:.*]] = vector.load
    // CHECK-SAME: : memref<128x128xf16, #gpu.address_space<global>>, vector<8xf16>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: @lower_write
func.func @lower_write(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %cst = arith.constant 0.0 : f16
  %reg = wave.register %cst : vector<8xf16>
  wave.write %reg, %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64)>,
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
      N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, BLOCK_N ceildiv 8, 1)>}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.store {{.*}}[%[[ROW]], %[[COL]]] : memref<{{.*}}xf16{{.*}}>, vector<8xf16>
     // CHECK-NOT: vector.transfer_write
  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
// CHECK-LABEL: @lower_write_non_innermost
func.func @lower_write_non_innermost(%mem: !wave.tensor<[@M, @N] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128, N = 128}>}  {
  %cst = arith.constant 0.0 : f16
  %reg = wave.register %cst : vector<8xf16>
  wave.write %reg, %mem index [{
      // CHECK: %[[BIDX_X:.*]] = gpu.block_id x
      // CHECK: %[[TIDX_X:.*]] = gpu.thread_id x
      // Note: BLOCK_M = 64
      // CHECK: %[[ROW:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + s1 mod 64)>()[%[[BIDX_X]], %[[TIDX_X]]]
      M : <[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 8, 64)>,
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
      N : <[#wave.index_symbol<WG1>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 8) * T1, 1, 1)>}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.transfer_write {{.*}}, {{.*}}[{{.*}}, {{.*}}] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<8xf16>, memref<{{.*}}xf16{{.*}}>

  return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_extract_static
  func.func @lower_extract_static() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK:     %[[INPUT:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
    %cst = arith.constant 1.0 : f32
    %input = wave.register %cst : vector<8xf32>

    // CHECK-NOT: wave.extract
    // CHECK:     %[[POS:.*]] = affine.apply affine_map<() -> (2)>()
    // CHECK:     %[[ELEM:.*]] = vector.extract %[[INPUT]][%[[POS]]] : f32 from vector<8xf32>
    // CHECK:     vector.broadcast %[[ELEM]] : f32 to vector<1xf32>
    %result = wave.extract %input[#wave.expr_list<[] -> (2)>] : (vector<8xf32>) -> vector<1xf32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_extract_dynamic
  func.func @lower_extract_dynamic() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    // CHECK-DAG: %[[INPUT:.*]] = arith.constant dense<2.000000e+00> : vector<8xf32>
    %cst = arith.constant 2.0 : f32
    %input = wave.register %cst : vector<8xf32>

    // CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id x
    // CHECK:     %[[POS:.*]] = affine.apply affine_map<()[s0] -> (s0 mod 4)>()[%[[TIDX]]]
    // CHECK:     %[[ELEM:.*]] = vector.extract %[[INPUT]][%[[POS]]] : f32 from vector<8xf32>
    // CHECK:     vector.broadcast %[[ELEM]] : f32 to vector<1xf32>
    %result = wave.extract %input[#wave.expr_list<[#wave.index_symbol<T0>] -> (T0 mod 4)>] : (vector<8xf32>) -> vector<1xf32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
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

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_iterate
  func.func @lower_iterate(%init: vector<8xf32>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 128, BLOCK_K = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>) {
    // CHECK:       scf.yield %{{.*}} : vector<8xf32>
    // CHECK:     }
    %result = wave.iterate @K iter_args(%init) {
    ^bb0(%arg0: vector<8xf32>):
      wave.yield %arg0 : vector<8xf32>
    } : (vector<8xf32>) -> vector<8xf32>

    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_iterate_with_operations
  func.func @lower_iterate_with_operations(%init: vector<4xf32>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, BLOCK_K = 16, M = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (vector<4xf32>) {
    // CHECK:       %[[TEMP:.*]] = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    // CHECK:       scf.yield %{{.*}} : vector<4xf32>
    // CHECK:     }
    %result = wave.iterate @K iter_args(%init) {
    ^bb0(%arg0: vector<4xf32>):
      %temp = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
        : memref<32xf32, #gpu.address_space<workgroup>>
      wave.yield %arg0 : vector<4xf32>
    } : (vector<4xf32>) -> vector<4xf32>

    return
  }
}

// -----

// CHECK: module {
// CHECK-NOT: normalform.module
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @test_normalform_module_lowered
  func.func @test_normalform_module_lowered() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @func_with_wave_input
  // CHECK-SAME: (%[[ARG0:.*]]: memref<32x32xf16, #gpu.address_space<global>>)
  func.func @func_with_wave_input(%arg0: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 32, N = 32}>} {
    func.return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @func_with_multiple_wave_inputs
  // CHECK-SAME: (%[[ARG0:.*]]: memref<32x32xf16, #gpu.address_space<global>>, %[[ARG1:.*]]: memref<32x32xf16, #gpu.address_space<global>>, %[[ARG2:.*]]: memref<32x32xf16, #gpu.address_space<global>>)
  func.func @func_with_multiple_wave_inputs(
      %arg0: !wave.tensor<[@M, @N] of f16, <global>>,
      %arg1: !wave.tensor<[@M, @N] of f16, <global>>,
      %arg2: !wave.tensor<[@M, @N] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 32, N = 32}>} {
    func.return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @func_with_mixed_input_types
  // CHECK-SAME: (%[[ARG0:.*]]: memref<64x64xf32, #gpu.address_space<global>>, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: memref<16x16xf16, #gpu.address_space<global>>)
  func.func @func_with_mixed_input_types(
      %arg0: !wave.tensor<[@M, @N] of f32, <global>>,
      %arg1: i32,
      %arg2: !wave.tensor<[@K, @L] of f16, <global>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, N = 64, K = 16, L = 16}>} {
    func.return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @func_without_wave_tensors
  // CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: i64) -> memref<32x32xf32>
  func.func @func_without_wave_tensors(%arg0: f32, %arg1: i64) -> memref<32x32xf32>
      attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %alloc = memref.alloc() : memref<32x32xf32>
    func.return %alloc : memref<32x32xf32>
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_iterate_with_vector_iter_args
  func.func @lower_iterate_with_vector_iter_args() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 128, BLOCK_K = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %cst = arith.constant 0.0 : f16
    %init = wave.register %cst : vector<32xf16>

    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (vector<32xf16>)
    %result = wave.iterate @K iter_args(%init) {
    ^bb0(%arg: vector<32xf16>):
      // Test vector operations within iterate body
      // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<32xf16>
      %add = wave.add %arg, %arg : (vector<32xf16>, vector<32xf16>) -> vector<32xf16>
      // CHECK: scf.yield %{{.*}} : vector<32xf16>
      wave.yield %add : vector<32xf16>
    } : (vector<32xf16>) -> vector<32xf16>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_iterate_multiple_vector_iter_args
  func.func @lower_iterate_multiple_vector_iter_args() attributes {
    wave.hyperparameters = #wave.hyperparameters<{I = 8}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"I">, tile_size = <[#wave.symbol<"I">] -> (I)>>
    ]
  } {
    %cst_f32 = arith.constant 1.0 : f32
    %cst_i32 = arith.constant 42 : i32
    %init1 = wave.register %cst_f32 : vector<4xf32>
    %init2 = wave.register %cst_i32 : vector<8xi32>

    // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (vector<4xf32>, vector<8xi32>)
    %result:2 = wave.iterate @I iter_args(%init1, %init2) {
    ^bb0(%arg1: vector<4xf32>, %arg2: vector<8xi32>):
      // CHECK: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
      %add = wave.add %arg1, %arg1 : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
      // CHECK: arith.addi %{{.*}}, %{{.*}} : vector<8xi32>
      %add2 = wave.add %arg2, %arg2 : (vector<8xi32>, vector<8xi32>) -> vector<8xi32>
      // CHECK: scf.yield %{{.*}}, %{{.*}} : vector<4xf32>, vector<8xi32>
      wave.yield %add, %add2 : vector<4xf32>, vector<8xi32>
    } : (vector<4xf32>, vector<8xi32>) -> (vector<4xf32>, vector<8xi32>)
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_iterate_with_vector_captures
  func.func @lower_iterate_with_vector_captures() attributes {
    wave.hyperparameters = #wave.hyperparameters<{I = 4}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"I">, tile_size = <[#wave.symbol<"I">] -> (I)>>
    ]
  } {
    %cst = arith.constant 2.0 : f32
    %cst2 = arith.constant 3.0 : f16
    %iter_arg = wave.register %cst : vector<8xf32>
    %capture = wave.register %cst2 : vector<4xf16>

    // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>)
    %result = wave.iterate @I iter_args(%iter_arg) captures(%capture) {
    ^bb0(%arg: vector<8xf32>, %cap: vector<4xf16>):
      // Test that vector captures work correctly - should use original SSA value
      // CHECK: arith.extf %{{.*}} : vector<4xf16> to vector<4xf32>
      %cast = wave.cast %cap : vector<4xf16> to vector<4xf32>
      // CHECK: scf.yield %{{.*}} : vector<8xf32>
      wave.yield %arg : vector<8xf32>
    } : (vector<8xf32>, vector<4xf16>) -> vector<8xf32>
    return
  }
}

// -----

// Test permute lowering with read/write - permute should be a pass-through.
// This simulates reading from memory, permuting dimensions, and writing back.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_permute_with_read_write
  func.func @lower_permute_with_read_write(
      %src: memref<64x64xf16, #gpu.address_space<workgroup>>,
      %dst: memref<64x64xf16, #gpu.address_space<workgroup>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64}>} {
    // CHECK: %[[READ:.*]] = vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %0 = wave.read %src index [{
        BLOCK_M : <[#wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 mod 64, 1, 64)>,
        BLOCK_N : <[#wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * 8, 8, 1)>
      }] {ordered_syms = [#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">]}
      : (memref<64x64xf16, #gpu.address_space<workgroup>>) -> vector<8xf16>

    // Permute is a pass-through at lowering time - just swaps dimension interpretation
    // CHECK-NOT: wave.permute
    %1 = wave.permute %0 : vector<8xf16> to vector<8xf16>

    // CHECK: vector.store %[[READ]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    wave.write %1, %dst index [{
        BLOCK_N : <[#wave.index_symbol<T0>, #wave.symbol<"BLOCK_N">] -> (T0 mod 64, 1, 64)>,
        BLOCK_M : <[#wave.index_symbol<T1>, #wave.symbol<"BLOCK_M">] -> (T1 * 8, 8, 1)>
      }] {ordered_syms = [#wave.symbol<"BLOCK_N">, #wave.symbol<"BLOCK_M">]}
      : vector<8xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Test read/write lowering with MemRefType memory operand.
// This simulates the state after ResolveDistributedAllocations pass has run.
// Dimension ordering is provided via ordered_syms attribute.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @lower_read_write_memref
  func.func @lower_read_write_memref(%mem: memref<64x64xf16, #gpu.address_space<workgroup>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64}>} {
    // CHECK: %[[READ:.*]] = vector.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %0 = wave.read %mem index [{
        BLOCK_M : <[#wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 mod 64, 1, 64)>,
        BLOCK_N : <[#wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * 8, 8, 1)>
      }] {ordered_syms = [#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">]}
      : (memref<64x64xf16, #gpu.address_space<workgroup>>) -> vector<8xf16>

    // CHECK: vector.store %[[READ]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    wave.write %0, %mem index [{
        BLOCK_M : <[#wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (T0 mod 64, 1, 64)>,
        BLOCK_N : <[#wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * 8, 8, 1)>
      }] {ordered_syms = [#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">]}
      : vector<8xf16>, memref<64x64xf16, #gpu.address_space<workgroup>>
    return
  }
}

// -----

// Test that ordered_syms correctly determines dimension ordering (not alphabetical).
// The dimensions are named M, K, N intentionally - alphabetically sorted would be K, M, N.
// But ordered_syms = [M, K, N] means the correct order is: dim0=M, dim1=K, dim2=N.
// If we used alphabetical ordering from DictionaryAttr, we'd get wrong index order.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: @ordered_syms_determines_dim_order
  func.func @ordered_syms_determines_dim_order(%mem: memref<64x32x128xf16, #gpu.address_space<workgroup>>)
      attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, K = 32, N = 128}>} {
    // The index expressions are designed to produce distinguishable affine expressions:
    // - M dimension: T0 (just thread_id x) -> affine_map<()[s0] -> (s0)>
    // - K dimension: T1 * 2 (scaled thread_id y) -> affine_map<()[s0] -> (s0 * 2)>
    // - N dimension: T2 * 4 (scaled by 4) -> affine_map<()[s0] -> (s0 * 4)>
    // The indices should be in M, K, N order (as specified by ordered_syms), NOT alphabetical (K, M, N).
    // If alphabetical ordering was used, we'd see: s0*2 first, then s0, then s0*4.
    // With correct ordering (M, K, N), we see: s0 first, then s0*2, then s0*4.
    // CHECK: %[[IDX_M:.*]] = affine.apply affine_map<()[s0] -> (s0)>
    // CHECK: %[[IDX_K:.*]] = affine.apply affine_map<()[s0] -> (s0 * 2)>
    // CHECK: %[[IDX_N:.*]] = affine.apply affine_map<()[s0] -> (s0 * 4)>
    // CHECK: vector.load %{{.*}}[%[[IDX_M]], %[[IDX_K]], %[[IDX_N]]] : memref<64x32x128xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %0 = wave.read %mem index [{
        M : <[#wave.index_symbol<T0>] -> (T0, 1, 1)>,
        K : <[#wave.index_symbol<T1>] -> (T1 * 2, 1, 1)>,
        N : <[#wave.index_symbol<T2>] -> (T2 * 4, 8, 1)>
      }] {ordered_syms = [#wave.symbol<"M">, #wave.symbol<"K">, #wave.symbol<"N">]}
      : (memref<64x32x128xf16, #gpu.address_space<workgroup>>) -> vector<8xf16>
      return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_shuffle_xor
  func.func @lower_shuffle_xor() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 1.0 : f16
    // CHECK: %[[REG:.*]] = arith.constant dense<1.000000e+00> : vector<1xf16>
    %0 = wave.register %cst : vector<1xf16>
    // CHECK-NOT: wave.shuffle
    // CHECK: %[[OFFSET:.*]] = arith.constant 1 : i32
    // CHECK: %[[WIDTH:.*]] = arith.constant 64 : i32
    // CHECK: %[[RESULT:.*]], %{{.*}} = gpu.shuffle xor %[[REG]], %[[OFFSET]], %[[WIDTH]] : vector<1xf16>
    %1 = wave.shuffle xor %0, 1, 64 : (vector<1xf16>) -> vector<1xf16>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_shuffle_down
  func.func @lower_shuffle_down() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 0.0 : f32
    // CHECK: %[[REG:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    %0 = wave.register %cst : vector<4xf32>
    // CHECK-NOT: wave.shuffle
    // CHECK: %[[OFFSET:.*]] = arith.constant 4 : i32
    // CHECK: %[[WIDTH:.*]] = arith.constant 32 : i32
    // CHECK: %[[RESULT:.*]], %{{.*}} = gpu.shuffle down %[[REG]], %[[OFFSET]], %[[WIDTH]] : vector<4xf32>
    %1 = wave.shuffle down %0, 4, 32 : (vector<4xf32>) -> vector<4xf32>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_shuffle_up
  func.func @lower_shuffle_up() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 2.0 : bf16
    // CHECK: %[[REG:.*]] = arith.constant dense<2.000000e+00> : vector<1xbf16>
    %0 = wave.register %cst : vector<1xbf16>
    // CHECK-NOT: wave.shuffle
    // CHECK: %[[OFFSET:.*]] = arith.constant 2 : i32
    // CHECK: %[[WIDTH:.*]] = arith.constant 64 : i32
    // CHECK: %[[RESULT:.*]], %{{.*}} = gpu.shuffle up %[[REG]], %[[OFFSET]], %[[WIDTH]] : vector<1xbf16>
    %1 = wave.shuffle up %0, 2, 64 : (vector<1xbf16>) -> vector<1xbf16>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_shuffle_idx
  func.func @lower_shuffle_idx() attributes {wave.hyperparameters = #wave.hyperparameters<{}>} {
    %cst = arith.constant 42 : i32
    // CHECK: %[[REG:.*]] = arith.constant dense<42> : vector<1xi32>
    %0 = wave.register %cst : vector<1xi32>
    // CHECK-NOT: wave.shuffle
    // CHECK: %[[OFFSET:.*]] = arith.constant 8 : i32
    // CHECK: %[[WIDTH:.*]] = arith.constant 64 : i32
    // CHECK: %[[RESULT:.*]], %{{.*}} = gpu.shuffle idx %[[REG]], %[[OFFSET]], %[[WIDTH]] : vector<1xi32>
    %1 = wave.shuffle idx %0, 8, 64 : (vector<1xi32>) -> vector<1xi32>
    return
  }
}

// -----

// Test wave.sum lowering - reduces input vector to scalar via thread-local
// reduction followed by subgroup reduction.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_sum
  // expected-warning @below {{unused hyperparameter: N}}
  func.func @lower_sum() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, N = 32}>} {
    %cst_input = arith.constant 1.0 : f32
    %cst_init = arith.constant 0.0 : f32
    // CHECK: %[[INPUT:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
    %input = wave.register %cst_input : vector<8xf32>
    // CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // CHECK-NOT: wave.sum
    // CHECK: %[[INIT_ELEM:.*]] = vector.extract %[[INIT]][0] : f32 from vector<1xf32>
    // CHECK: %[[THREAD_REDUCE:.*]] = vector.reduction <add>, %[[INPUT]], %[[INIT_ELEM]] : vector<8xf32> into f32
    // CHECK: gpu.subgroup_reduce add %[[THREAD_REDUCE]] : (f32) -> f32
    %result = wave.sum %input init(%init) along @M <warp> : (vector<8xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test wave.max_element lowering - reduces input vector to scalar via
// thread-local max followed by subgroup max.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_max_element
  // expected-warning @below {{unused hyperparameter: M}}
  func.func @lower_max_element() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, N = 32}>} {
    %cst_input = arith.constant 2.0 : f32
    %cst_init = arith.constant 0.0 : f32
    // CHECK: %[[INPUT:.*]] = arith.constant dense<2.000000e+00> : vector<16xf32>
    %input = wave.register %cst_input : vector<16xf32>
    // CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // CHECK-NOT: wave.max_element
    // CHECK: %[[INIT_ELEM:.*]] = vector.extract %[[INIT]][0] : f32 from vector<1xf32>
    // CHECK: %[[THREAD_REDUCE:.*]] = vector.reduction <maximumf>, %[[INPUT]], %[[INIT_ELEM]] : vector<16xf32> into f32
    // CHECK: gpu.subgroup_reduce maximumf %[[THREAD_REDUCE]] : (f32) -> f32
    %result = wave.max_element %input init(%init) along @N <warp> : (vector<16xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test reduction with integer types.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_sum_integer
  func.func @lower_sum_integer() attributes {wave.hyperparameters = #wave.hyperparameters<{K = 16}>} {
    %cst_input = arith.constant 1 : i32
    %cst_init = arith.constant 0 : i32
    // CHECK: %[[INPUT:.*]] = arith.constant dense<1> : vector<4xi32>
    %input = wave.register %cst_input : vector<4xi32>
    // CHECK: %[[INIT:.*]] = arith.constant dense<0> : vector<1xi32>
    %init = wave.register %cst_init : vector<1xi32>
    // CHECK-NOT: wave.sum
    // CHECK: %[[INIT_ELEM:.*]] = vector.extract %[[INIT]][0] : i32 from vector<1xi32>
    // CHECK: %[[THREAD_REDUCE:.*]] = vector.reduction <add>, %[[INPUT]], %[[INIT_ELEM]] : vector<4xi32> into i32
    // CHECK: gpu.subgroup_reduce add %[[THREAD_REDUCE]] : (i32) -> i32
    %result = wave.sum %input init(%init) along @K <warp> : (vector<4xi32>, vector<1xi32>) -> vector<1xi32>
    return
  }
}

// -----

// Test wave.sum lowering with <block> - uses gpu.all_reduce instead of gpu.subgroup_reduce.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_sum_block
  // expected-warning @below {{unused hyperparameter: N}}
  func.func @lower_sum_block() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, N = 32}>} {
    %cst_input = arith.constant 1.0 : f32
    %cst_init = arith.constant 0.0 : f32
    // CHECK: %[[INPUT:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
    %input = wave.register %cst_input : vector<8xf32>
    // CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // CHECK-NOT: wave.sum
    // CHECK: %[[INIT_ELEM:.*]] = vector.extract %[[INIT]][0] : f32 from vector<1xf32>
    // CHECK: %[[THREAD_REDUCE:.*]] = vector.reduction <add>, %[[INPUT]], %[[INIT_ELEM]] : vector<8xf32> into f32
    // CHECK: gpu.all_reduce add %[[THREAD_REDUCE]] {
    // CHECK: } : (f32) -> f32
    %result = wave.sum %input init(%init) along @M <block> : (vector<8xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test wave.max_element lowering with <block> - uses gpu.all_reduce instead of gpu.subgroup_reduce.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_max_element_block
  // expected-warning @below {{unused hyperparameter: M}}
  func.func @lower_max_element_block() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64, N = 32}>} {
    %cst_input = arith.constant 2.0 : f32
    %cst_init = arith.constant 0.0 : f32
    // CHECK: %[[INPUT:.*]] = arith.constant dense<2.000000e+00> : vector<16xf32>
    %input = wave.register %cst_input : vector<16xf32>
    // CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // CHECK-NOT: wave.max_element
    // CHECK: %[[INIT_ELEM:.*]] = vector.extract %[[INIT]][0] : f32 from vector<1xf32>
    // CHECK: %[[THREAD_REDUCE:.*]] = vector.reduction <maximumf>, %[[INPUT]], %[[INIT_ELEM]] : vector<16xf32> into f32
    // CHECK: gpu.all_reduce maximumf %[[THREAD_REDUCE]] {
    // CHECK: } : (f32) -> f32
    %result = wave.max_element %input init(%init) along @N <block> : (vector<16xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test warning when <block> but hardware constraint specifies only one wave per block.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @warn_block_reduction_single_wave
  func.func @warn_block_reduction_single_wave()
    attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64}>,
                wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>]} {
    %cst_input = arith.constant 1.0 : f32
    %cst_init = arith.constant 0.0 : f32
    %input = wave.register %cst_input : vector<8xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // expected-warning @below {{block reduction requested but hardware constraint specifies only one wave per block (waves_per_block = [1, 1, 1]); consider using wave-level reduction instead}}
    %result = wave.sum %input init(%init) along @M <block> : (vector<8xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test warning when <warp> but hardware constraint specifies multiple waves per block.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @warn_wave_reduction_multiple_waves
  func.func @warn_wave_reduction_multiple_waves()
    attributes {wave.hyperparameters = #wave.hyperparameters<{M = 64}>,
                wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1]>]} {
    %cst_input = arith.constant 1.0 : f32
    %cst_init = arith.constant 0.0 : f32
    %input = wave.register %cst_input : vector<8xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // expected-warning @below {{wave-level reduction requested but hardware constraint specifies multiple waves per block (waves_per_block = [2, 2, 1]); consider using block reduction to reduce across all waves}}
    %result = wave.sum %input init(%init) along @M <warp> : (vector<8xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test warning for max_element with <block> and single wave.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @warn_block_max_single_wave
  func.func @warn_block_max_single_wave()
    attributes {wave.hyperparameters = #wave.hyperparameters<{N = 32}>,
                wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1]>]} {
    %cst_input = arith.constant 2.0 : f32
    %cst_init = arith.constant 0.0 : f32
    %input = wave.register %cst_input : vector<16xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // expected-warning @below {{block reduction requested but hardware constraint specifies only one wave per block (waves_per_block = [1, 1, 1]); consider using wave-level reduction instead}}
    %result = wave.max_element %input init(%init) along @N <block> : (vector<16xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test warning for max_element with <warp> and multiple waves.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @warn_wave_max_multiple_waves
  func.func @warn_wave_max_multiple_waves()
    attributes {wave.hyperparameters = #wave.hyperparameters<{N = 32}>,
                wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [4, 1, 1]>]} {
    %cst_input = arith.constant 2.0 : f32
    %cst_init = arith.constant 0.0 : f32
    %input = wave.register %cst_input : vector<16xf32>
    %init = wave.register %cst_init : vector<1xf32>
    // expected-warning @below {{wave-level reduction requested but hardware constraint specifies multiple waves per block (waves_per_block = [4, 1, 1]); consider using block reduction to reduce across all waves}}
    %result = wave.max_element %input init(%init) along @N <warp> : (vector<16xf32>, vector<1xf32>) -> vector<1xf32>
    return
  }
}

// -----

// Test wave.self_index lowering with unit stride - produces start + iota.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_self_index_unit_stride
  func.func @lower_self_index_unit_stride() -> vector<4xi32> attributes {wave.hyperparameters = #wave.hyperparameters<{N = 64}>} {
    // CHECK-NOT: wave.self_index
    // CHECK: %[[TID:.*]] = gpu.thread_id  x
    // CHECK: %[[START:.*]] = affine.apply
    // CHECK: %[[IOTA:.*]] = vector.step : vector<4xindex>
    // CHECK: %[[START_VEC:.*]] = vector.broadcast %[[START]] : index to vector<4xindex>
    // CHECK: %[[ADD:.*]] = arith.addi %[[START_VEC]], %[[IOTA]] : vector<4xindex>
    // CHECK: %[[RESULT:.*]] = arith.index_cast %[[ADD]] : vector<4xindex> to vector<4xi32>
    // CHECK: return %[[RESULT]]
    %0 = wave.self_index @N index [{N : <[#wave.index_symbol<T0>] -> (T0, 4, 1)>}] : vector<4xi32>
    return %0 : vector<4xi32>
  }
}

// -----

// Test wave.self_index lowering with non-unit stride - produces start + iota * stride.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_self_index_with_stride
  func.func @lower_self_index_with_stride() -> vector<4xi64> attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
    // CHECK-NOT: wave.self_index
    // CHECK: %[[TID:.*]] = gpu.thread_id  x
    // CHECK: %[[START:.*]] = affine.apply
    // CHECK: %[[IOTA:.*]] = vector.step : vector<4xindex>
    // CHECK: %[[START_VEC:.*]] = vector.broadcast %[[START]] : index to vector<4xindex>
    // CHECK: %[[C16:.*]] = arith.constant 16 : index
    // CHECK: %[[STRIDE_VEC:.*]] = vector.broadcast %[[C16]] : index to vector<4xindex>
    // CHECK: %[[SCALED:.*]] = arith.muli %[[IOTA]], %[[STRIDE_VEC]] : vector<4xindex>
    // CHECK: %[[ADD:.*]] = arith.addi %[[START_VEC]], %[[SCALED]] : vector<4xindex>
    // CHECK: %[[RESULT:.*]] = arith.index_cast %[[ADD]] : vector<4xindex> to vector<4xi64>
    // CHECK: return %[[RESULT]]
    %0 = wave.self_index @M index [{M : <[#wave.index_symbol<T0>] -> (T0 * 4, 4, 16)>}] : vector<4xi64>
    return %0 : vector<4xi64>
  }
}

// -----

// Test wave.self_index lowering with complex start expression involving
// workgroup and thread IDs.
normalform.module [#wave.normal_form<full_types,index_exprs,memory_only_types,resolved_allocations,ordered_syms>] {
  // CHECK-LABEL: func.func @lower_self_index_complex_start
  func.func @lower_self_index_complex_start() -> vector<8xi32> attributes {wave.hyperparameters = #wave.hyperparameters<{M = 256, BLOCK_M = 64}>} {
    // CHECK-NOT: wave.self_index
    // CHECK-DAG: gpu.thread_id  x
    // CHECK-DAG: gpu.block_id  x
    // CHECK: affine.apply
    // CHECK: vector.step : vector<8xindex>
    // CHECK: arith.addi
    // CHECK: arith.index_cast {{.*}} : vector<8xindex> to vector<8xi32>
    %0 = wave.self_index @M index [{
      M : <[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.index_symbol<T0>] -> (WG0 * BLOCK_M + T0, 8, 1)>
    }] : vector<8xi32>
    return %0 : vector<8xi32>
  }
}
