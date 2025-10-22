// RUN: water-opt %s --split-input-file --verify-diagnostics

func.func @mismatch_element_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@C, @B] of f32>, %acc: !wave.tensor<[@A, @C] of bf16>) {
  // expected-error @below {{unexpected accumulator/result elemental type 'bf16' for MMA kind f32_16x16x16_f16}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@C, @B] of f32>, !wave.tensor<[@A, @C] of bf16>) -> !wave.tensor<[@A, @C] of bf16>
}

// -----

func.func @mismatch_element_int(%lhs: !wave.tensor<[@A, @B] of i32>, %rhs: !wave.tensor<[@C, @B] of i32>, %acc: !wave.tensor<[@A, @C] of i32>) {
  // expected-error @below {{unexpected lhs/rhs elemental type 'i32' for MMA kind i32_32x32x16_i8}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<i32_32x32x16_i8>} : (!wave.tensor<[@A, @B] of i32>, !wave.tensor<[@C, @B] of i32>, !wave.tensor<[@A, @C] of i32>) -> !wave.tensor<[@A, @C] of i32>
}

// -----

func.func @mismatch_element_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@C, @B] of bf16>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS and RHS elemental types to match, got 'f32', 'bf16'}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@C, @B] of bf16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_rhs(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@B, @C] of f16>, %acc: !wave.tensor<[@A, @C] of f32>) {
  // expected-error @below {{expected LHS dimension #1 (#wave.symbol<"B">) to match RHS dimension #1 (#wave.symbol<"C">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@B, @C] of f16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
}

// -----

func.func @mismatch_dim_lhs_acc(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@C, @B] of f16>, %acc: !wave.tensor<[@E, @D] of f32>) {
  // expected-error @below {{expected LHS dimension #0 (#wave.symbol<"A">) to match accumulator dimension #0 (#wave.symbol<"E">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@C, @B] of f16>, !wave.tensor<[@E, @D] of f32>) -> !wave.tensor<[@E, @D] of f32>
}

// -----

func.func @mismatch_dim_rhs_acc(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@C, @B] of f16>, %acc: !wave.tensor<[@A, @D] of f32>) {
  // expected-error @below {{expected RHS dimension #0 (#wave.symbol<"C">) to match accumulator dimension #1 (#wave.symbol<"D">)}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@C, @B] of f16>, !wave.tensor<[@A, @D] of f32>) -> !wave.tensor<[@A, @D] of f32>
}

// -----

func.func @invalid_register_type(%arg0: f32) {
  // expected-error @below {{expected wave tensor or vector type, got 'f32'}}
  wave.register %arg0 : f32
}

// -----

func.func @register_type_mismatch(%arg0: f32) {
  // expected-error @below {{expected the type of the init value to match the elemental type of the result}}
  "wave.register"(%arg0) : (f32) -> !wave.tensor<[@A, @B] of bf16>
}

// -----

func.func @register_underspecified(%arg0: f32) {
  // expected-error @below {{expected fully-specified tensor type}}
  wave.register %arg0 : !wave.tensor<any of f32>
}

// -----

func.func @iterate_mismatching_results(%arg0: !wave.tensor<any of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{expects the same number of iter_args (2) and results (1)}}
  wave.iterate @I iter_args(%arg0, %arg1) {
  ^bb0(%arg2: !wave.tensor<any of f32>, %arg3: !wave.tensor<any of f32>):
    wave.yield %arg2 : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @iterate_mismatching_results(%arg0: !wave.tensor<[@A] of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{along control flow edge from parent operands to Region #0: source type #0 '!wave.tensor<[@A] of f32>' should match input type #0 '!wave.tensor<[@B] of f32>'}}
  wave.iterate @I iter_args(%arg0, %arg1) {
  ^bb0(%arg2: !wave.tensor<[@B] of f32>, %arg3: !wave.tensor<any of f32>):
    wave.yield %arg2, %arg3 : !wave.tensor<[@B] of f32>, !wave.tensor<any of f32>
  } : (!wave.tensor<[@A] of f32>, !wave.tensor<any of f32>) -> (!wave.tensor<any of f32>, !wave.tensor<any of f32>)
}

// -----

// must provide the full triple (start, step, stride)
func.func @index_attr_missing_step_stride(%arg0: f32) {
  // expected-error @+2 {{expected ','}}
  // expected-error @+1 {{custom op 'wave.register' expected three affine expressions for '(start, step, stride)'}}
  wave.register %arg0 index {X : [_WG0] -> (_WG0)} : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

// must provide the full triple (start, step, stride)
func.func @index_attr_missing_stride(%arg0: f32) {
  // expected-error @+2 {{expected ','}}
  // expected-error @+1 {{custom op 'wave.register' expected three affine expressions for '(start, step, stride)'}}
  wave.register %arg0 index {X : [_WG0] -> (_WG0, 1)} : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

func.func @index_attr_not_dict(%arg0: f32) {
  // expected-error @+1 {{'wave.register' op attribute 'index' failed to satisfy constraint: dictionary of named attribute values}}
  "wave.register"(%arg0) { index = 42 } : (f32) -> !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

func.func @mismatch_shape_binary(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>) {
  // expected-error @below {{expected operand #1 dimension #0 (#wave.symbol<"B">) to match operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.add %lhs, %rhs : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @mismatch_shape_unary(%lhs: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{expected result #0 dimension #0 (#wave.symbol<"B">) to match operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.exp2 %lhs : (!wave.tensor<[@A, @B] of f32>) -> !wave.tensor<[@B, @C] of f32>
  return
}

// -----

func.func @mismatch_shape_read(%lhs: !wave.tensor<[@A, @B] of f32, <global>>) {
  // expected-error @below {{expected result #0 dimension #0 (#wave.symbol<"B">) to match operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.read %lhs : (!wave.tensor<[@A, @B] of f32, <global>>) -> !wave.tensor<[@B, @C] of f32, <register>>
  return
}

// -----

func.func @mismatch_shape_write(%lhs: !wave.tensor<[@A, @B] of f32, <register>>, %rhs: !wave.tensor<[@B, @C] of f32, <global>>) {
  // expected-error @below {{expected operand #1 dimension #0 (#wave.symbol<"B">) to match operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.write %lhs, %rhs : !wave.tensor<[@A, @B] of f32, <register>>, !wave.tensor<[@B, @C] of f32, <global>>
}

// -----


func.func @empty_distributed_shape() {
  // expected-error @below {{wave expression attribute must have at least one dimension}}
  %buf = wave.allocate { distributed_shape = #wave.expr_list<[BLOCK_M, BLOCK_K] -> ()>}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
}

// -----

func.func @alloc_offset_no_parent() {
  // expected-error @below {{expects parent and offset to be present simultaneously}}
  %buf = wave.allocate { distributed_shape = #wave.expr_list<[] -> (42)>, offset = 42}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
}

// -----

func.func @alloc_parent_no_offset() {
  %alloc = wave.allocate { distributed_shape = #wave.expr_list<[] -> (100)> }
    : !wave.tensor<[@M, @K] of bf16, <shared>>
  // expected-error @below {{expects parent and offset to be present simultaneously}}
  %buf = wave.allocate in %alloc : !wave.tensor<[@M, @K] of bf16, <shared>> { distributed_shape = #wave.expr_list<[] -> (42)>}
    : !wave.tensor<[@M, @K] of bf16, <shared>>
}

// -----

module attributes { wave.hyperparameters = #wave.hyperparameters<{}> } {
  // expected-error @below {{defines hyperparameters when its ancestor already had}}
  // expected-note @above {{ancestor}}
  module attributes { wave.hyperparameters = #wave.hyperparameters<{}> } {}
}

// -----

module attributes { wave.hyperparameters = #wave.hyperparameters<{A = 42, C = 43}> } {
  // expected-error @below {{region #0 block #0 argument type #0 uses symbolic value #wave.symbol<"B"> not provided as a hyperparameter}}
  // expected-note @below {{available symbols: A, C}}
  // expected-note @below {{NYI support for symbol lowering}}
  func.func @missing_hyperparam_func(%arg0: !wave.tensor<[@B] of f32>) {
    return
  }
}


// -----

module attributes { wave.hyperparameters = #wave.hyperparameters<{A = 42, C = 43}> } {
  func.func @missing_hyperparam_result(%arg0: !wave.tensor<any of f32>) {
    // expected-error @below {{result type #0 uses symbolic value #wave.symbol<"B"> not provided as a hyperparameter}}
    // expected-note @below {{available symbols: A, C}}
    // expected-note @below {{NYI support for symbol lowering}}
    wave.add %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@B] of f32>
    return
  }
}

// -----

// expected-warning @below {{unused hyperparameter: C}}
module attributes { wave.hyperparameters = #wave.hyperparameters<{A = 42, C = 43}> } {
  func.func @missing_hyperparam_result(%arg0: !wave.tensor<any of f32>) {
    wave.add %arg0, %arg0 : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@A] of f32>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @index_key_unspecified(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128}>}  {
    // expected-error @below {{attribute "index" uses symbolic value "N" not provided as a hyperparameter}}
    // expected-note @below {{BLOCK_M, BLOCK_N, M}}
    %0 = wave.read %mem index {
        M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 1, 64),
        N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 2) * _T1, BLOCK_N ceildiv 2, 1)}
      : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @index_value_unspecified(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 256}>}  {
    // expected-error @below {{attribute "index" uses symbolic value #wave.symbol<"BLOCK_M"> not provided as a hyperparameter}}
    // expected-note @below {{available symbols: M, N}}
    %0 = wave.read %mem index {
        M : [BLOCK_M, _WG0, _T0] -> (BLOCK_M * _WG0 + (BLOCK_M floordiv 2) * (_T0 floordiv 64) + _T0 mod 64, 1, 64),
        N : [_T1, _WG1, BLOCK_N] -> (_WG1 * BLOCK_N + (BLOCK_N floordiv 2) * _T1, BLOCK_N ceildiv 2, 1)}
      : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types>} {
  func.func @elements_per_thread_mismatch(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
    // expected-error @below {{expected result vector type to have the number of elements per thread matching the attribute (4), got 42}}
    %0 = wave.read %mem {elements_per_thread=4}: (!wave.tensor<[@M] of f16, <global>>) -> vector<42xf16>
    return
  }
}

// -----

func.func @bounds_missing_dim(%mem: !wave.tensor<[@M, @N] of f32>, %val: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{bounds not provided for memory tensor symbol 'N'}}
  wave.write %val, %mem { bounds = #wave.read_write_bounds<{ M = #wave.expr_list<[BLOCK_M] -> (BLOCK_M * 64)>}> } : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32>
  return
}

// -----

func.func @bounds_extraneous_dim(%mem: !wave.tensor<[@N] of f32>, %val: !wave.tensor<[@N] of f32, <register>>) {
  // expected-error @below {{'bounds' specified for a symbol "M" not used in the indexed memory tensor}}
  wave.write %val, %mem { bounds = #wave.read_write_bounds<{ M = #wave.expr_list<[BLOCK_M] -> (BLOCK_M * 64)>}> } : !wave.tensor<[@N] of f32, <register>>, !wave.tensor<[@N] of f32>
  return
}

// -----

func.func @bounds_wrong_type(%mem: !wave.tensor<[@N] of f32>) {
  // expected-error @below {{'bounds' values must be WaveWaveExprListAttr, got 42 : i64}}
  wave.read %mem { bounds = #wave.read_write_bounds<{ N = 42 }> } : (!wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
  return
}

// -----

func.func @read_index_multi_step(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{'index' has more than one entry with non-unit step}}
  // expected-note @below {{second non-unit step dimension: 1}}
  wave.read %mem index {
    M : [_T0] -> (_T0, 2, 1),
    N : [_T1] -> (_T1, 2, 1)
  } : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<any of f32, <register>>
  return
}

// -----

func.func @read_index_elements_per_thread_mismatch(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{vectorized dimension step in the index expression with current hyperparameters (2) doesn't match the explicitly specified elements per thread value (4)}}
  wave.read %mem index {
    M : [_T0] -> (_T0, 2, 1),
    N : [_T1] -> (_T1, 1, 1)
  } {
    elements_per_thread = 4
  } : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<any of f32, <register>>
  return
}


// -----

func.func @read_index_type_mismatch(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{vectorized dimension step in the index expression with current hyperparameters (2) doesn't match the vector size (4)}}
  wave.read %mem index {
    M : [_T0] -> (_T0, 2, 1),
    N : [_T1] -> (_T1, 1, 1)
  } : (!wave.tensor<[@M, @N] of f32>) -> vector<4xf32>
  return
}

// -----

func.func @read_index_multi_step_eval(%mem: !wave.tensor<[@M, @N] of f32>) attributes {
  wave.hyperparameters = #wave.hyperparameters<{X = 1, Y = 1, M = 100, N = 200}>
} {
  // expected-error @below {{'index' has more than one entry with non-unit step}}
  // expected-note @below {{second non-unit step dimension: 1}}
  wave.read %mem index {
    M : [_T0, X] -> (_T0, 2 * X, 1),
    N : [_T1, X, Y] -> (_T1, X + Y, 1)
  } : (!wave.tensor<[@M, @N] of f32>) -> vector<4xf32>
  return
}
