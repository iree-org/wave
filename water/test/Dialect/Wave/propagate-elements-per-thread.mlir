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
