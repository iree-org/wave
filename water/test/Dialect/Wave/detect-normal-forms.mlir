// RUN: water-opt %s --water-wave-detect-normal-forms --split-input-file | FileCheck %s

// None of the ops inside are wave ops, so full types are specified.
// None of the ops need index expressions, meaning that form is also specified.
// CHECK-LABEL: @full_func_boundary_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types,index_exprs>
module @full_func_boundary_satisfied_module {
  func.func @full_func_boundary_satisfied(%arg0: !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32> {
    return %arg0 : !wave.tensor<[@M, @N] of f32>
  }
}

// -----

// None of the ops need index expressions, meaning that form is also specified.
// CHECK-LABEL: @full_func_boundary_not_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<index_exprs>
module @full_func_boundary_not_satisfied_module {
  func.func @full_func_boundary_not_satisfied(%arg0: !wave.tensor<any of f32>) -> !wave.tensor<any of f32> {
    return %arg0 : !wave.tensor<any of f32>
  }
}

// -----

// Explicit index expressions are provided.
// CHECK-LABEL: @index_exprs_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types,index_exprs>
module @index_exprs_satisfied_module {
  func.func @index_exprs_satisfied() {
    %c = arith.constant 0.0 : f32
    %0 = wave.register %c
      index [{
        M : [#wave.symbol<"THREAD_ID">, #wave.symbol<"BLOCK_SIZE">] -> (THREAD_ID floordiv BLOCK_SIZE, 1, 1),
        N : [#wave.symbol<"THREAD_ID">, #wave.symbol<"BLOCK_SIZE">] -> (THREAD_ID * BLOCK_SIZE + 42, 1, 1)
      }]
      : !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}

// -----

// CHECK-LABEL: @index_exprs_not_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types>
module @index_exprs_not_satisfied_module {
  func.func @index_exprs_not_satisfied() {
    %c = arith.constant 0.0 : f32
    %0 = wave.register %c : !wave.tensor<[@M, @N] of f32, <register>>
    return
  }
}

// -----

// CHECK-LABEL: @memory_only_types_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types,memory_only_types>
module @memory_only_types_satisfied_module {
  func.func @memory_only_types_satisfied(%global: !wave.tensor<[@X] of f32, <global>>) {
    %0 = wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_Y">] -> (BLOCK_Y)>} : !wave.tensor<[@Y] of bf16, <shared>>
    return
  }
}

// -----

// CHECK-LABEL: @memory_only_types_not_satisfied_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types>
module @memory_only_types_not_satisfied_module {
  func.func @memory_only_types_not_satisfied() {
    %c = arith.constant 0.0 : f32
    %0 = wave.register %c : !wave.tensor<[@X] of f32, <register>>
    return
  }
}

// -----

// CHECK-LABEL: @multiple_ops_with_index_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types,index_exprs>
module @multiple_ops_with_index_module {
  func.func @multiple_ops_with_index() {
    %c = arith.constant 0.0 : f32
    %0 = wave.register %c
      index [{M : [] -> (0, 1, 1)}]
      : !wave.tensor<[@M] of f32, <register>>
    %1 = wave.register %c
      index [{N : [] -> (0, 1, 1)}]
      : !wave.tensor<[@N] of f32, <register>>
    return
  }
}

// -----

// Empty module: all normal forms trivially satisfied, needs wave dialect to be registered.
// CHECK-LABEL: @empty_module
// CHECK-SAME: wave.normal_form = #wave.normal_form<full_types,index_exprs,memory_only_types>
module @empty_module {
}
