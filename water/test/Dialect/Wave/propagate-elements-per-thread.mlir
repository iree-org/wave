// RUN: water-opt %s --water-wave-propagate-elements-per-thread --split-input-file --verify-diagnostics --allow-unregistered-dialect | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @register_alone() attributes {wave.hyperparameters = #wave.hyperparameters<{Y = 10, Z = 1}>} {
    %cst = arith.constant 0.0 : f32
    // expected-error @below {{couldn't identify elements per thread for result #0}}
    wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @register_add() attributes {wave.hyperparameters = #wave.hyperparameters<{Y = 10, Z = 1}>} {
    %cst = arith.constant 0.0 : f32
    // expected-error @below {{couldn't identify elements per thread for result #0}}
    %reg = wave.register %cst { elements_per_thread = 4 } : !wave.tensor<[@Y, @Z] of f32, <register>>
    wave.add %reg, %reg : (!wave.tensor<[@Y, @Z] of f32, <register>>, !wave.tensor<[@Y, @Z] of f32, <register>>) -> !wave.tensor<[@Y, @Z] of f32, <register>>
    return
  }
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @propagate_register_write
func.func @propagate_register_write(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  %cst = arith.constant 0.0 : f16
  // CHECK: wave.register {{.*}} : vector<8xf16>
  %reg = wave.register %cst : !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.write {{.*}} : vector<8xf16>, !wave.tensor<[@M] of f16, <global>>
  wave.write %reg, %mem { elements_per_thread = 8 }
     : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <global>>
  return
}
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @propagate_backward_from_write
func.func @propagate_backward_from_write(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  %cst = arith.constant 0.0 : f16
  // CHECK: wave.register {{.*}} : vector<8xf16>
  %reg = wave.register %cst : !wave.tensor<[@M] of f16, <register>>
  %cst2 = arith.constant 42.0 : f16
  // CHECK: wave.register {{.*}} : vector<8xf16>
  %reg2 = wave.register %cst2 : !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.add {{.*}} : (vector<8xf16>, vector<8xf16>) -> vector<8xf16>
  %sum = wave.add %reg, %reg2 : (!wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.mul {{.*}} : (vector<8xf16>, vector<8xf16>) -> vector<8xf16>
  %mul = wave.mul %sum, %reg2 : (!wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.exp2 {{.*}} : (vector<8xf16>) -> vector<8xf16>
  %exp = wave.exp2 %mul : (!wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>

  // CHECK: wave.write {{.*}} : vector<8xf16>, !wave.tensor<[@M] of f16, <global>>
  wave.write %exp, %mem { elements_per_thread = 8 }
     : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <global>>
  return
}
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @propagate_forward_from_read
func.func @propagate_forward_from_read(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  // CHECK: wave.read {{.*}} : (!wave.tensor<[@M] of f16, <global>>) -> vector<4xf16>
  %reg = wave.read %mem { elements_per_thread = 4 } : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  %cst2 = arith.constant 42.0 : f16
  // CHECK: wave.register {{.*}} : vector<4xf16>
  %reg2 = wave.register %cst2 : !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.add {{.*}} : (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
  %sum = wave.add %reg, %reg2 : (!wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.mul {{.*}} : (vector<4xf16>, vector<4xf16>) -> vector<4xf16>
  %mul = wave.mul %sum, %reg2 : (!wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>
  // CHECK: wave.exp2 {{.*}} : (vector<4xf16>) -> vector<4xf16>
  %exp = wave.exp2 %mul : (!wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>

  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @missing_elements_per_thread(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  // expected-error @below {{couldn't identify elements per thread for result #0}}
  %reg = wave.read %mem : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @read_write_conflict(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  %reg = wave.read %mem {elements_per_thread = 4} : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  // expected-error @below {{failed to propagate elements per thread backward: mismatch between elements_per_thread attribute (8) and operand #0 (4)}}
  wave.write %reg, %mem {elements_per_thread = 8} : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <global>>
  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @read_write_conflict_indirect(%mem: !wave.tensor<[@M] of f16, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
  %reg = wave.read %mem {elements_per_thread = 4} : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
  %val = wave.exp2 %reg : (!wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>
  // expected-error @below {{failed to propagate elements per thread backward: mismatch between elements_per_thread attribute (8) and operand #0 (4)}}
  wave.write %reg, %mem {elements_per_thread = 8} : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <global>>
  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: func.func @alloc_is_harmless
func.func @alloc_is_harmless() attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 4, BLOCK_K = 28, M = 128, N=128, K= 128}>}  {
  // CHECK: wave.allocate
  %parent = wave.allocate { distributed_shape = #wave.expr_list<[] -> (256)> }
    : !wave.tensor<[@M,@N,@K] of i8, <shared>>

  // CHECK: wave.allocate
  %buf = wave.allocate in %parent : !wave.tensor<[@M,@N,@K] of i8, <shared>>
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> (BLOCK_M, BLOCK_K + 4)>, offset = 128}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @unsupported_op() attributes {wave.hyperparameters = #wave.hyperparameters<{Y = 100, Z = 200}>}  {
  %cst = arith.constant 42.0 : f32
  %reg = wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
  // expected-error @below {{cannot propagate elements per thread information across an operation not implementing the corresponding interface}}
  "foo.bar"(%reg) : (!wave.tensor<[@Y, @Z] of f32, <register>>) -> !wave.tensor<[@Y, @Z] of f32, <register>>
  return
}
}


// -----

// CHECK: #wave.normal_form<memory_only_types>
module {
  func.func @test_no_existing_normal_form_attr(%mem: !wave.tensor<[@M] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
    %0 = arith.constant 0.0 : f32
    %reg = wave.register %0 : !wave.tensor<[@M] of f32, <register>>
    wave.write %reg, %mem { elements_per_thread = 8 } : !wave.tensor<[@M] of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
    return
  }
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @memory_resharding_allowed
func.func @memory_resharding_allowed(%mem: !wave.tensor<[@M] of f16, <shared>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
  %cst = arith.constant 0.0 : f16
  // Register gets 8 elements per thread from write operation's backward propagation.
  // CHECK: wave.register {{.*}} : vector<8xf16>
  %reg8 = wave.register %cst : !wave.tensor<[@M] of f16, <register>>

  // Write 8 elements per thread to memory.
  // CHECK: wave.write {{.*}} : vector<8xf16>, !wave.tensor<[@M] of f16, <shared>>
  wave.write %reg8, %mem {elements_per_thread = 8} : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <shared>>

  // Read 4 elements per thread from same memory - this should be allowed (memory resharding).
  // CHECK: wave.read {{.*}} : (!wave.tensor<[@M] of f16, <shared>>) -> vector<4xf16>
  %reg4 = wave.read %mem {elements_per_thread = 4} : (!wave.tensor<[@M] of f16, <shared>>) -> !wave.tensor<[@M] of f16, <register>>

  return
}
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @write_backward_propagation
func.func @write_backward_propagation(%mem: !wave.tensor<[@M] of f16, <shared>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
  %cst = arith.constant 0.0 : f16
  // RegisterOp doesn't have explicit elements_per_thread - should get it from backward propagation.
  // CHECK: wave.register {{.*}} : vector<4xf16>
  %reg = wave.register %cst : !wave.tensor<[@M] of f16, <register>>

  // WriteOp should propagate elements_per_thread backward to register operand.
  // CHECK: wave.write {{.*}} : vector<4xf16>, !wave.tensor<[@M] of f16, <shared>>
  wave.write %reg, %mem {elements_per_thread = 4} : !wave.tensor<[@M] of f16, <register>>, !wave.tensor<[@M] of f16, <shared>>

  return
}
}

// -----

// CHECK: #wave.normal_form<full_types,memory_only_types>
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
// CHECK-LABEL: @read_register_propagation
func.func @read_register_propagation(%mem: !wave.tensor<[@M] of f16, <shared>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
  // ReadOp should only propagate to its register result, not validate memory.
  // CHECK: wave.read {{.*}} : (!wave.tensor<[@M] of f16, <shared>>) -> vector<6xf16>
  %reg = wave.read %mem {elements_per_thread = 6} : (!wave.tensor<[@M] of f16, <shared>>) -> !wave.tensor<[@M] of f16, <register>>

  // Downstream operation should get 6 elements per thread.
  // CHECK: wave.exp2 {{.*}} : (vector<6xf16>) -> vector<6xf16>
  %result = wave.exp2 %reg : (!wave.tensor<[@M] of f16, <register>>) -> !wave.tensor<[@M] of f16, <register>>

  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @mma_compute_lhs_from_rhs(%mem1: !wave.tensor<[@N, @K] of f16, <global>>, %mem2: !wave.tensor<[@M, @N] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 16, N = 16, K = 16}>, wave.constraints = [#wave.hardware_constraint<threads_per_wave = 32, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 1, N = 1, K = 16}, max_bits_per_load = 128>]} {
  // LHS without elements_per_thread - will be computed from RHS + MMA constraints.
  %lhs_init = arith.constant 0.0 : f16
  %lhs = wave.register %lhs_init : !wave.tensor<[@M, @K] of f16, <register>>

  // RHS properly initialized through read operation.
  %rhs = wave.read %mem1 {elements_per_thread = 8} : (!wave.tensor<[@N, @K] of f16, <global>>) -> !wave.tensor<[@N, @K] of f16, <register>>

  // ACC properly initialized through read operation.
  %acc = wave.read %mem2 {elements_per_thread = 8} : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>

  // LHS elements_per_thread computed via MMA backward propagation.
  %result = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  return
}
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
func.func @mma_compute_rhs_from_lhs(%mem1: !wave.tensor<[@M, @K] of f16, <global>>, %mem2: !wave.tensor<[@M, @N] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 16, N = 16, K = 16}>, wave.constraints = [#wave.hardware_constraint<threads_per_wave = 32, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 1, N = 1, K = 16}, max_bits_per_load = 128>]} {
  // LHS properly initialized through read operation.
  %lhs = wave.read %mem1 {elements_per_thread = 8} : (!wave.tensor<[@M, @K] of f16, <global>>) -> !wave.tensor<[@M, @K] of f16, <register>>

  // RHS without elements_per_thread - will be computed from LHS + MMA constraints.
  %rhs_init = arith.constant 0.0 : f16
  %rhs = wave.register %rhs_init : !wave.tensor<[@N, @K] of f16, <register>>

  // ACC properly initialized through read operation.
  %acc = wave.read %mem2 {elements_per_thread = 8} : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>

  // RHS elements_per_thread computed via MMA backward propagation.
  %result = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
  return
}
}

// -----

// Test MMA can compute both LHS and RHS when both are uninitialized
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @mma_compute_both_lhs_rhs(%mem1: !wave.tensor<[@M, @K] of f16, <global>>, %mem2: !wave.tensor<[@N, @K] of f16, <global>>, %mem3: !wave.tensor<[@M, @N] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 16, N = 16, K = 16}>, wave.constraints = [#wave.hardware_constraint<threads_per_wave = 32, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 1, N = 1, K = 16}, max_bits_per_load = 128>]} {
    // Both LHS and RHS without elements_per_thread - can compute from MMA formulas.
    %lhs_init = arith.constant 0.0 : f16
    %lhs = wave.register %lhs_init : !wave.tensor<[@M, @K] of f16, <register>>
    %rhs_init = arith.constant 0.0 : f16
    %rhs = wave.register %rhs_init : !wave.tensor<[@N, @K] of f16, <register>>

    // ACC properly initialized through read operation.
    %acc = wave.read %mem3 {elements_per_thread = 8} : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // With proper MMA formulas, we can now compute both LHS and RHS from constraints,
    // so this should succeed instead of failing.
    %result = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}

// -----

// Test MMA error when operand has wrong elements_per_thread
module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @mma_operand_mismatch(%mem1: !wave.tensor<[@M, @K] of f16, <global>>, %mem2: !wave.tensor<[@M, @N] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 16, N = 16, K = 16}>, wave.constraints = [#wave.hardware_constraint<threads_per_wave = 32, waves_per_block = [1, 1, 1], mma_type = #wave.mma_kind<f32_16x16x16_f16>, vector_shapes = {M = 1, N = 1, K = 16}, max_bits_per_load = 128>]} {
    // LHS with wrong elements_per_thread (should be 8, not 4).
    %lhs = wave.read %mem1 {elements_per_thread = 4} : (!wave.tensor<[@M, @K] of f16, <global>>) -> !wave.tensor<[@M, @K] of f16, <register>>

    // RHS without elements_per_thread - will be computed from MMA constraints.
    %rhs_init = arith.constant 0.0 : f16
    %rhs = wave.register %rhs_init : !wave.tensor<[@N, @K] of f16, <register>>

    // ACC properly initialized.
    %acc = wave.read %mem2 {elements_per_thread = 8} : (!wave.tensor<[@M, @N] of f32, <global>>) -> !wave.tensor<[@M, @N] of f32, <register>>

    // expected-error @below {{failed to propagate elements per thread backward: mismatch between computed from MMA kind (8) and LHS operand #0 (4)}}
    %result = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}
