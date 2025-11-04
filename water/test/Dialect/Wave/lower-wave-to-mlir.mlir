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
    { distributed_shape = #wave.expr_list<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
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
    { distributed_shape = #wave.expr_list<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>, offset = 128}
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
    { distributed_shape = #wave.expr_list<[BLOCK_M, BLOCK_K] -> (BLOCK_M, BLOCK_K + 4)>}
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
      M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 1, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 8) * _T1, BLOCK_N ceildiv 8, 1)}]
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
    M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 * 8 , 8, 64),
    // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
    // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
    // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
    N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 8) * _T1, 1, 1)}]
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
        M : [BLOCK_M, _WG0, _T0] -> (_WG0 * BLOCK_M + _T0, 1, 64),
        // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
        // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
        // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s0 * 64 + s1 * 32)>()[%[[BIDX_Y]], %[[TIDX_Y]]]
        N : [_WG1, _T1, BLOCK_N] -> (_WG1 * BLOCK_N + _T1 * 32, 4, 1)
      }] { bounds = #wave.read_write_bounds<{
        M = #wave.expr_list<[M] -> (M)>,
        N = #wave.expr_list<[N] -> (N)>}>}
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
        M : [BLOCK_M, _WG0, _T0] -> (_WG0 * BLOCK_M + _T0, 8, 64),
        N : [_WG1, _T1, BLOCK_N] -> (_WG1 * BLOCK_N + _T1 * 32, 1, 1)
      }] { bounds = #wave.read_write_bounds<{
        M = #wave.expr_list<[M] -> (M)>,
        N = #wave.expr_list<[N] -> (N)>}>}
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
      M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 1, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 8) * _T1, BLOCK_N ceildiv 8, 1)}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.store {{.*}}[%[[ROW]], %[[COL]]] : memref<{{.*}}xf16{{.*}}>, vector<8xf16>

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
      M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 8, 64),
      // CHECK: %[[TIDX_Y:.*]] = gpu.thread_id y
      // CHECK: %[[BIDX_Y:.*]] = gpu.block_id y
      // CHECK: %[[COL:.*]] = affine.apply affine_map<()[s0, s1] -> (s1 * 64 + s0 * 8)>()[%[[TIDX_Y]], %[[BIDX_Y]]]
      N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 8) * _T1, 1, 1)}]
     : vector<8xf16>, !wave.tensor<[@M, @N] of f16, <global>>
     // CHECK: vector.transfer_write {{.*}}, {{.*}}[{{.*}}, {{.*}}] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<8xf16>, memref<{{.*}}xf16{{.*}}>

  return
  }
}
