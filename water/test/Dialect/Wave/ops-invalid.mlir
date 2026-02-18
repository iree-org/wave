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

func.func @mma_1d(%lhs: !wave.tensor<[@A] of f16>, %rhs: !wave.tensor<[@B] of f16>, %acc: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expects at least 2D operands for MMA}}
  wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A] of f16>, !wave.tensor<[@B] of f16>, !wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of f32>
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @mma_3d_mismatch(%a: !wave.tensor<[@M, @K, @B] of f16>,
                             %b: !wave.tensor<[@N, @K, @B] of f16>,
                             %c: !wave.tensor<[@M, @N, @B] of f32>) {
    // expected-error @below {{expected LHS dimension #0 (#wave.symbol<"M">) to match RHS dimension #0 (#wave.symbol<"N">)}}
    wave.mma %a, %b, %c {kind = #wave.mma_kind<f32_16x16x16_f16>}
      : (!wave.tensor<[@M, @K, @B] of f16>, !wave.tensor<[@N, @K, @B] of f16>, !wave.tensor<[@M, @N, @B] of f32>) -> !wave.tensor<[@M, @N, @B] of f32>
    return
  }
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

func.func @iterate_operands_block_args_mismatch(%arg0: !wave.tensor<any of f32>) {
  // expected-error @below {{expects the same number of operands (1) and block arguments (2)}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<any of f32>, %arg2: !wave.tensor<any of f32>):
    wave.yield %arg1 : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @iterate_iter_args_block_iter_args_mismatch(%arg0: !wave.tensor<any of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{expects the same number of operands (2) and block arguments (1)}}
  wave.iterate @I iter_args(%arg0, %arg1) {
  ^bb0(%arg2: !wave.tensor<any of f32>):
    wave.yield %arg2 : !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @iterate_iter_arg_block_arg_element_type_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected iter arg #0 and block iter arg #0 elemental types to match, got 'f32', 'bf16'}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of bf16>):
    wave.yield %arg1 : !wave.tensor<[@A] of bf16>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of bf16>
}

// -----

func.func @iterate_iter_arg_block_arg_rank_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{rank mismatch between iter arg #0 and block iter arg #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A, @B] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A, @B] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A, @B] of f32>
}

// -----

func.func @iterate_iter_arg_block_arg_shape_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected iter_args #0 dimension #0 (#wave.symbol<"A">) to match result #0 dimension #0 (#wave.symbol<"B">)}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@B] of f32>):
    wave.yield %arg1 : !wave.tensor<[@B] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@B] of f32>
}

// -----

func.func @iterate_iter_arg_block_arg_address_space_mismatch(%arg0: !wave.tensor<[@A] of f32, <register>>) {
  // expected-error @below {{address space mismatch between iter arg #0 and block iter arg #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32, <shared>>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32, <shared>>
  } : (!wave.tensor<[@A] of f32, <register>>) -> !wave.tensor<[@A] of f32, <shared>>
}

// -----

func.func @iterate_iter_arg_result_element_type_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected result #0 and terminator operand #0 elemental types to match, got 'bf16', 'f32'}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A] of bf16>
}

// -----

func.func @iterate_iter_arg_result_rank_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{rank mismatch between result #0 and terminator operand #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@A, @B] of f32>
}

// -----

func.func @iterate_iter_arg_result_shape_mismatch(%arg0: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected iter_args #0 dimension #0 (#wave.symbol<"A">) to match result #0 dimension #0 (#wave.symbol<"B">)}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>) -> !wave.tensor<[@B] of f32>
}

// -----

func.func @iterate_iter_arg_result_address_space_mismatch(%arg0: !wave.tensor<[@A] of f32, <register>>) {
  // expected-error @below {{address space mismatch between result #0 and terminator operand #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32, <register>>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32, <register>>
  } : (!wave.tensor<[@A] of f32, <register>>) -> !wave.tensor<[@A] of f32, <shared>>
}

// -----

func.func @iterate_capture_type_mismatch(%arg0: !wave.tensor<[@A] of f32>, %capture: !wave.tensor<[@B] of f32>) {
  // expected-error @below {{expects the same type for capture #0 and block argument #1}}
  wave.iterate @I iter_args(%arg0) captures(%capture) {
  ^bb0(%arg1: !wave.tensor<[@A] of f32>, %arg2: !wave.tensor<[@B] of bf16>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of f32>, !wave.tensor<[@B] of f32>) -> !wave.tensor<[@A] of f32>
}

// -----

func.func @iterate_results_terminator_operands_mismatch(%arg0: !wave.tensor<any of f32>, %arg1: !wave.tensor<any of f32>) {
  // expected-error @below {{expects the same number of results (1) and terminator operands (2)}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg2: !wave.tensor<any of f32>):
    wave.yield %arg2, %arg1 : !wave.tensor<any of f32>, !wave.tensor<any of f32>
  } : (!wave.tensor<any of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @iterate_result_terminator_element_type_mismatch(%arg0: !wave.tensor<[@A] of bf16>, %arg1: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected result #0 and terminator operand #0 elemental types to match, got 'bf16', 'f32'}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg2: !wave.tensor<[@A] of bf16>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A] of bf16>) -> !wave.tensor<[@A] of bf16>
}

// -----

func.func @iterate_result_terminator_rank_mismatch(%arg0: !wave.tensor<[@A, @B] of f32>, %arg1: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{rank mismatch between result #0 and terminator operand #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg2: !wave.tensor<[@A, @B] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@A, @B] of f32>) -> !wave.tensor<[@A, @B] of f32>
}

// -----

func.func @iterate_result_terminator_shape_mismatch(%arg0: !wave.tensor<[@B] of f32>, %arg1: !wave.tensor<[@A] of f32>) {
  // expected-error @below {{expected result #0 dimension #0 (#wave.symbol<"B">) to match terminator operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg2: !wave.tensor<[@B] of f32>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32>
  } : (!wave.tensor<[@B] of f32>) -> !wave.tensor<[@B] of f32>
}

// -----

func.func @iterate_result_terminator_address_space_mismatch(%arg0: !wave.tensor<[@A] of f32, <shared>>, %arg1: !wave.tensor<[@A] of f32, <register>>) {
  // expected-error @below {{address space mismatch between result #0 and terminator operand #0}}
  wave.iterate @I iter_args(%arg0) {
  ^bb0(%arg2: !wave.tensor<[@A] of f32, <shared>>):
    wave.yield %arg1 : !wave.tensor<[@A] of f32, <register>>
  } : (!wave.tensor<[@A] of f32, <shared>>) -> !wave.tensor<[@A] of f32, <shared>>
}

// -----

// must provide the full triple (start, step, stride)
func.func @index_attr_wrong_attr_type(%arg0: f32) {
  // expected-error @below {{expected symbol names to be one of WaveSymbolAttr, WaveIndexSymbolAttr, WaveIterSymbolAttr or WaveOperandAttr}}
  wave.register %arg0 index [{X : <[#wave.workgroup_dim<x>] -> (WG0)>}] : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

// must provide the full triple (start, step, stride)
func.func @index_attr_missing_step_stride(%arg0: f32) {
  // expected-error @+2 {{expected ','}}
  // expected-error @+1 {{custom op 'wave.register' expected three affine expressions for '(start, step, stride)'}}
  wave.register %arg0 index [{X : <[#wave.index_symbol<WG0>] -> (WG0)>}] : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

// must provide the full triple (start, step, stride)
func.func @index_attr_missing_stride(%arg0: f32) {
  // expected-error @+2 {{expected ','}}
  // expected-error @+1 {{custom op 'wave.register' expected three affine expressions for '(start, step, stride)'}}
  wave.register %arg0 index [{X : <[#wave.index_symbol<WG0>] -> (WG0, 1)>}] : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

func.func @index_attr_not_dict(%arg0: f32) {
  // expected-error @+1 {{'wave.register' op attribute 'index' failed to satisfy constraint: Array of dictionary attributes}}
  "wave.register"(%arg0) { index = 42 } : (f32) -> !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

// 'index' array elements must be dictionaries mapping to WaveIndexMappingAttr values.
func.func @index_attr_wrong_value_type(%arg0: f32) {
  // expected-error @below {{'index' attribute value for key "M" must be WaveIndexMappingAttr, got 42 : i64}}
  "wave.register"(%arg0) { index = [{ M = 42 }] } : (f32) -> vector<4xf32>
  return
}

// -----

func.func @index_attr_iter_not_allowed(%arg0: f32) {
  // expected-error @below {{index expression uses iterator symbol M which is not defined by any parent op}}
  wave.register %arg0 index [{M : <[#wave.iter<"M">] -> (_Iter_M, 1, 1)>}] : !wave.tensor<[@M] of f32, <register>>
  return
}

// -----

func.func @mismatch_shape_binary(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>) {
  // expected-error @below {{expected operand #1 dimension #0 (#wave.symbol<"B">) to match operand #0 dimension #0 (#wave.symbol<"A">)}}
  wave.add %lhs, %rhs : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>) -> !wave.tensor<any of f32>
}

// -----

func.func @mismatch_element_type_tensor_vector(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{expected result #0 and operand #0 elemental types to match, got 'f64', 'f32'}}
  wave.add %lhs, %rhs : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@A, @B] of f32>) -> vector<4xf64>
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
  // expected-error @below {{distributed shape must have at least one result}}
  %buf = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_K">] -> ()>}
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

func.func @child_alloc_with_tail_padding() {
  %parent = wave.allocate { distributed_shape = #wave.expr_list<[] -> (8192)> }
    : !wave.tensor<[@M] of i8, <shared>>
  // expected-error @below {{only top-level allocations can have tail_padding}}
  %buf = wave.allocate in %parent : !wave.tensor<[@M] of i8, <shared>>
    { distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, offset = 0, tail_padding = 64 : i64 }
    : !wave.tensor<[@M] of bf16, <shared>>
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

normalform.module [#wave.normal_form<full_types>] {
  func.func @index_key_unspecified(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 128}>}  {
    // expected-error @below {{attribute "index" uses symbolic value "N" not provided as a hyperparameter}}
    // expected-note @below {{BLOCK_M, BLOCK_N, M}}
    %0 = wave.read %mem index [{
        M : <[#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64)>,
        N : <[#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 2) * T1, BLOCK_N ceildiv 2, 1)>}]
      : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
    return
  }
}

// -----

func.func @read_element_type_mismatch(%mem: memref<64x64xf16, #gpu.address_space<workgroup>>) {
  // expected-error @below {{expected result #0 and operand #0 elemental types to match, got 'f32', 'f16'}}
  %0 = wave.read %mem : (memref<64x64xf16, #gpu.address_space<workgroup>>) -> vector<8xf32>
  return
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @index_value_unspecified(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, N = 256}>}  {
    // expected-error @below {{attribute "index" uses symbolic value #wave.symbol<"BLOCK_M"> not provided as a hyperparameter}}
    // expected-note @below {{available symbols: M, N}}
    %0 = wave.read %mem index [{
        M : <[#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 64, 1, 64)>,
        N : <[#wave.index_symbol<T1>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (BLOCK_N floordiv 2) * T1, BLOCK_N ceildiv 2, 1)>}]
      : (!wave.tensor<[@M] of f16, <global>>) -> !wave.tensor<[@M] of f16, <register>>
    return
  }
}

// -----

normalform.module [#wave.normal_form<full_types>] {
  func.func @elements_per_thread_mismatch(%mem: !wave.tensor<[@M] of f16, <global>>)
  attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>}  {
    // expected-error @below {{expected result vector type to have the number of elements per thread matching the attribute (4), got 42}}
    %0 = wave.read %mem {elements_per_thread=4}: (!wave.tensor<[@M] of f16, <global>>) -> vector<42xf16>
    return
  }
}

// -----

func.func @bounds_extraneous_dim(%mem: !wave.tensor<[@N] of f32>, %val: !wave.tensor<[@N] of f32, <register>>) {
  // expected-error @below {{'bounds' specified for a symbol M not used in the indexed memory tensor}}
  wave.write %val, %mem { bounds = #wave.symbol_mapping<@M = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M * 64)>> } : !wave.tensor<[@N] of f32, <register>>, !wave.tensor<[@N] of f32>
  return
}

// -----

func.func @bounds_wrong_rank(%mem: !wave.tensor<[@N] of f32>) {
  // expected-error @below {{op attribute 'bounds' failed to satisfy constraint: symbol mapping with 1-result expr list value}}
  wave.read %mem { bounds = #wave.symbol_mapping<@N = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M * 64, BLOCK_M * 64)>> } : (!wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32, <register>>
  return
}

// -----

func.func @read_index_multi_step(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{'index' has more than one entry with non-unit step}}
  // expected-note @below {{second non-unit step dimension: 1}}
  wave.read %mem index [{
    M : <[#wave.index_symbol<T0>] -> (T0, 2, 1)>,
    N : <[#wave.index_symbol<T1>] -> (T1, 2, 1)>
  }] : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<any of f32, <register>>
  return
}

// -----

func.func @read_index_elements_per_thread_mismatch(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{vectorized dimension step in the index expression with current hyperparameters (2) doesn't match the explicitly specified elements per thread value (4)}}
  wave.read %mem index [{
    M : <[#wave.index_symbol<T0>] -> (T0, 2, 1)>,
    N : <[#wave.index_symbol<T1>] -> (T1, 1, 1)>
  }]{
    elements_per_thread = 4
  } : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<any of f32, <register>>
  return
}


// -----

func.func @read_index_type_mismatch(%mem: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{vectorized dimension step in the index expression with current hyperparameters (2) doesn't match the vector size (4)}}
  wave.read %mem index [{
    M : <[#wave.index_symbol<T0>] -> (T0, 2, 1)>,
    N : <[#wave.index_symbol<T1>] -> (T1, 1, 1)>
  }] : (!wave.tensor<[@M, @N] of f32>) -> vector<4xf32>
  return
}

// -----

func.func @read_index_multi_step_eval(%mem: !wave.tensor<[@M, @N] of f32>) attributes {
  wave.hyperparameters = #wave.hyperparameters<{X = 1, Y = 1, M = 100, N = 200}>
} {
  // expected-error @below {{'index' has more than one entry with non-unit step}}
  // expected-note @below {{second non-unit step dimension: 1}}
  wave.read %mem index [{
    M : <[#wave.index_symbol<T0>, #wave.symbol<"X">] -> (T0, 2 * X, 1)>,
    N : <[#wave.index_symbol<T1>, #wave.symbol<"X">, #wave.symbol<"Y">] -> (T1, X + Y, 1)>
  }] : (!wave.tensor<[@M, @N] of f32>) -> vector<4xf32>
  return
}

// -----

func.func @extract_invalid_position_rank(%src: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{position must contain exactly one expression, but got 2}}
  wave.extract %src[#wave.expr_list<[] -> (0, 1)>] : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M] of f32>
  return
}

// -----

func.func @extract_vector_result_not_one_element(%src: vector<4xf32>) {
  // expected-error @below {{result must be a 1-element vector, got}}
  wave.extract %src[#wave.expr_list<[] -> (0)>] : (vector<4xf32>) -> vector<3xf32>
  return
}

// -----

func.func @extract_result_same_rank_as_source(%src: !wave.tensor<[@M, @N] of f32>) attributes {
  wave.hyperparameters = #wave.hyperparameters<{M = 16, N = 16}>
} {
  // expected-error @below {{result tensor must have one less dimension than source}}
  wave.extract %src[#wave.expr_list<[] -> (0)>] : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
  return
}

// -----

func.func @extract_source_not_fully_specified(%src: !wave.tensor<any of f32>) {
  // expected-error @below {{source tensor type must be fully specified}}
  %0 = wave.extract %src[#wave.expr_list<[] -> (0)>] : (!wave.tensor<any of f32>) -> !wave.tensor<[@X] of f32>
  return
}

// -----

func.func @extract_target_not_fully_specified(%src: !wave.tensor<[@X] of f32>) {
  // expected-error @below {{target tensor type must be fully specified}}
  %0 = wave.extract %src[#wave.expr_list<[] -> (0)>] : (!wave.tensor<[@X] of f32>) -> !wave.tensor<any of f32>
  return
}

// -----

func.func @extract_dimension_mismatch(%src: !wave.tensor<[@M, @N] of f32>) {
  // expected-error @below {{source tensor type dimensions must contain the result tensor type dimension}}
  %0 = wave.extract %src[#wave.expr_list<[] -> (0)>] : (!wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@X] of f32>
  return
}

// -----

func.func @extract_slice_mismatch_offset_size(%memory: !wave.tensor<[@A, @B] of f16>) {
  // expected-error @below {{offset, size, and stride must all have the same rank, but got offset rank 1, size rank 2, and stride rank 1}}
  wave.extract_slice %memory {offset = #wave.expr_list<[] -> (3)>, size = #wave.expr_list<[] -> (32, 16)>, stride = #wave.expr_list<[] -> (2)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return
}

// -----

func.func @extract_slice_mismatch_offset_stride(%memory: !wave.tensor<[@A, @B] of f16>) {
  // expected-error @below {{offset, size, and stride must all have the same rank, but got offset rank 2, size rank 2, and stride rank 1}}
  wave.extract_slice %memory {offset = #wave.expr_list<[] -> (3, 5)>, size = #wave.expr_list<[] -> (32, 16)>, stride = #wave.expr_list<[] -> (2)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return
}

// -----

func.func @extract_slice_mismatch_size_stride(%memory: !wave.tensor<[@A, @B] of f16>) {
  // expected-error @below {{offset, size, and stride must all have the same rank, but got offset rank 1, size rank 1, and stride rank 3}}
  wave.extract_slice %memory {offset = #wave.expr_list<[] -> (3)>, size = #wave.expr_list<[] -> (32)>, stride = #wave.expr_list<[] -> (2, 1, 4)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return
}

// -----

func.func @extract_slice_with_symbols(%memory: !wave.tensor<[@A, @B] of f16>) {
  // expected-error @below {{offset, size, and stride must be constant expressions with no symbols, but got offset with 1 symbols, size with 0 symbols, and stride with 0 symbols}}
  wave.extract_slice %memory {offset = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, size = #wave.expr_list<[] -> (32)>, stride = #wave.expr_list<[] -> (2)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return
}

// -----

func.func @extract_slice_all_with_symbols(%memory: !wave.tensor<[@A, @B] of f16>) {
  // expected-error @below {{offset, size, and stride must be constant expressions with no symbols, but got offset with 1 symbols, size with 1 symbols, and stride with 1 symbols}}
  wave.extract_slice %memory {offset = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M + 2)>, size = #wave.expr_list<[#wave.symbol<"BLOCK_K">] -> (BLOCK_K * 2)>, stride = #wave.expr_list<[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>} : (!wave.tensor<[@A, @B] of f16>) -> !wave.tensor<[@A, @B] of f16>
  return
}

// -----

func.func @cast_wave_tensor_shape_mismatch(%arg0: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{shape of input (#wave.symbol<"A">, #wave.symbol<"B">) must match shape of result (#wave.symbol<"C">, #wave.symbol<"D">)}}
  wave.cast %arg0 : !wave.tensor<[@A, @B] of f32> to !wave.tensor<[@C, @D] of bf16>
  return
}

// -----

func.func @cast_vector_shape_mismatch(%arg0: vector<4xf32>) {
  // expected-error @below {{shape of input (4) must match shape of result (8)}}
  wave.cast %arg0 : vector<4xf32> to vector<8xbf16>
  return
}

// -----

func.func @cast_invalid_result_type(%arg0: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{result #0 must be Wave tensor in register or a 1D vector, but got 'f32'}}
  wave.cast %arg0 : !wave.tensor<[@A, @B] of f32> to f32
  return
}

// -----

func.func @cast_wave_tensor_rank_mismatch(%arg0: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{shape of input (#wave.symbol<"A">, #wave.symbol<"B">) must match shape of result (#wave.symbol<"C">, #wave.symbol<"D">, #wave.symbol<"E">)}}
  wave.cast %arg0 : !wave.tensor<[@A, @B] of f32> to !wave.tensor<[@C, @D, @E] of bf16>
  return
}

// -----

func.func @cast_underspecified_to_different_shape(%arg0: !wave.tensor<[@A, @B] of f32>) {
  // expected-error @below {{shape of input (#wave.symbol<"A">, #wave.symbol<"B">) must match shape of result (#wave.symbol<"X">, #wave.symbol<"Y">)}}
  wave.cast %arg0 : !wave.tensor<[@A, @B] of f32> to !wave.tensor<[@X, @Y] of i32>
  return
}

// -----

// Test mixed wave tensor and vector types in iterate - should fail
func.func @iterate_mixed_tensor_vector_types() attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128, I = 4}>} {
  %cst = arith.constant 0.0 : f32
  %tensor_input = wave.register %cst : !wave.tensor<[@M] of f32, <register>>
  %vector_input = arith.constant dense<1.0> : vector<8xf32>

  // expected-error @below {{iter_args #0 and result #0 must be the same category of types (both wave tensors or both vectors)}}
  %iter_result:2 = wave.iterate @I iter_args(%tensor_input, %vector_input) {
  ^bb0(%in_arg0: !wave.tensor<[@M] of f32, <register>>, %in_arg1: vector<8xf32>):
    wave.yield %in_arg1, %in_arg0 : vector<8xf32>, !wave.tensor<[@M] of f32, <register>>
  } : (!wave.tensor<[@M] of f32, <register>>, vector<8xf32>) -> (vector<8xf32>, !wave.tensor<[@M] of f32, <register>>)
  return
}

// -----

// Test vector type mismatch in iterate
func.func @iterate_vector_type_mismatch() attributes {wave.hyperparameters = #wave.hyperparameters<{I = 4}>} {
  %input = arith.constant dense<1.0> : vector<8xf32>

  // expected-error @below {{iter_args #0 type ('vector<8xf32>') must match result #0 type ('vector<4xf32>')}}
  %result = wave.iterate @I iter_args(%input) {
  ^bb0(%in_arg: vector<8xf32>):
    %different = arith.constant dense<2.0> : vector<4xf32>
    wave.yield %different : vector<4xf32>
  } : (vector<8xf32>) -> (vector<4xf32>)
  return
}

// -----

// Test vector element type mismatch in iterate
func.func @iterate_vector_element_type_mismatch() attributes {wave.hyperparameters = #wave.hyperparameters<{I = 4}>} {
  %input = arith.constant dense<1.0> : vector<8xf32>

  // expected-error @below {{iter_args #0 type ('vector<8xf32>') must match result #0 type ('vector<8xf16>')}}
  %result = wave.iterate @I iter_args(%input) {
  ^bb0(%in_arg: vector<8xf32>):
    %different = arith.constant dense<2.0> : vector<8xf16>
    wave.yield %different : vector<8xf16>
  } : (vector<8xf32>) -> (vector<8xf16>)
  return
}

// -----

// Test that multidimensional vectors are rejected in iterate
func.func @iterate_multidim_vectors_rejected() attributes {wave.hyperparameters = #wave.hyperparameters<{I = 4}>} {
  %input = arith.constant dense<1.0> : vector<4x8xf32>

  // expected-error @below {{'wave.iterate' op operand #0 must be variadic of wave tensor or 1d vector type, but got 'vector<4x8xf32>'}}
  %result = wave.iterate @I iter_args(%input) {
  ^bb0(%in_arg: vector<4x8xf32>):
    wave.yield %in_arg : vector<4x8xf32>
  } : (vector<4x8xf32>) -> (vector<4x8xf32>)
  return
}

// -----

func.func @nonexistent_axis(%input: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // expected-error @below {{did not expect axis attribute when input type is fully specified}}
  %result = wave.sum %input init(%init) along @K <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}

// -----

func.func @rank_mismatch(%input: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32> {
  // expected-error @below {{init tensor rank (2) must be one less than input tensor rank (2)}}
  %result = wave.sum %input init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>
  return %result : !wave.tensor<[@M, @N] of f32>
}

// -----

func.func @rank_mismatch(%input: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N, @M] of f32> {
  %init_any = wave.reciprocal %init: (!wave.tensor<[@N] of f32>) -> !wave.tensor<any of f32>
  // expected-error @below {{result tensor rank (2) must be one less than input tensor rank (2)}}
  %result = wave.sum %input init(%init_any) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@N, @M] of f32>
  return %result : !wave.tensor<[@N, @M] of f32>
}

// -----

func.func @rank_mismatch(%input: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N, @M] of f32> {
  // expected-error @below {{rank mismatch between init and result}}
  %result = wave.sum %input init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N, @M] of f32>
  return %result : !wave.tensor<[@N, @M] of f32>
}

// -----

func.func @symbol_mismatch(%input: !wave.tensor<[@N, @M] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@M] of f32> {
  // expected-error @below {{expected init dimension #0 (#wave.symbol<"N">) to match result dimension #0 (#wave.symbol<"M">)}}
  %result = wave.sum %input init(%init) <warp> : (!wave.tensor<[@N, @M] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@M] of f32>
  return %result : !wave.tensor<[@M] of f32>
}

// -----

func.func @sum_along_first_dim(%input: !wave.tensor<[@M, @N] of f32>, %init: !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32> {
  // expected-error @below {{expected init dimension #0 (#wave.symbol<"N">) to match input dimension #0 (#wave.symbol<"M">)}}
  %result = wave.sum %input init(%init) <warp> : (!wave.tensor<[@M, @N] of f32>, !wave.tensor<[@N] of f32>) -> !wave.tensor<[@N] of f32>
  return %result : !wave.tensor<[@N] of f32>
}

// -----

func.func @underspecified_reduction(%input: !wave.tensor<any of f32>, %init: !wave.tensor<any of f32>) -> !wave.tensor<any of f32> {
  // expected-error @below {{expected axis attribute when input type is not fully specified}}
  %result = wave.sum %input init(%init) <warp> : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<any of f32>
  return %result : !wave.tensor<any of f32>
}

// -----

func.func @reduction_init_contains_axis_explicit(%input: !wave.tensor<any of f32>, %init: !wave.tensor<[@N, @M] of f32>) -> !wave.tensor<any of f32> {
  // Reducing along axis @M, but init tensor shape contains @M.
  // expected-error @below {{init tensor shape must not contain the reduced axis}}
  %result = wave.sum %input init(%init) along @M <warp> : (!wave.tensor<any of f32>, !wave.tensor<[@N, @M] of f32>) -> !wave.tensor<any of f32>
  return %result : !wave.tensor<any of f32>
}

// -----

func.func @reduction_result_contains_axis_explicit(%input: !wave.tensor<any of f32>, %init: !wave.tensor<any of f32>) -> !wave.tensor<[@N, @M] of f32> {
  // Reducing along axis @M, but result tensor shape contains @M.
  // expected-error @below {{result tensor shape must not contain the reduced axis}}
  %result = wave.sum %input init(%init) along @M <warp> : (!wave.tensor<any of f32>, !wave.tensor<any of f32>) -> !wave.tensor<[@N, @M] of f32>
  return %result : !wave.tensor<[@N, @M] of f32>
}

// -----

func.func @reduction_init_and_result_contain_axis(%input: !wave.tensor<any of f32>, %init: !wave.tensor<[@K] of f32>) -> !wave.tensor<[@K] of f32> {
  // Reducing along axis @K, but both init and result shapes contain @K.
  // expected-error @below {{init tensor shape must not contain the reduced axis}}
  %result = wave.max_element %input init(%init) along @K <warp> : (!wave.tensor<any of f32>, !wave.tensor<[@K] of f32>) -> !wave.tensor<[@K] of f32>
  return %result : !wave.tensor<[@K] of f32>
}

// -----

func.func @broadcast_source_dim_not_in_result(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // Source has [@M, @N], result has [@M, @P, @K] - N is missing (replaced by P).
  // expected-error @below {{source dimension 'N' not found in result shape}}
  wave.broadcast %arg0 : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @P, @K] of f32, <register>>
  return
}

// -----

func.func @broadcast_explicit_dims_with_fully_specified_types(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // When both source and result types are fully specified, explicit dims are not expected.
  // expected-error @below {{does not expect explicit dims when source and result types are fully specified}}
  wave.broadcast %arg0 dims [@K] : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N, @K] of f32, <register>>
  return
}

// -----

func.func @broadcast_element_type_mismatch(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // Source and result must have matching element types.
  // expected-error @below {{expected source and result elemental types to match, got 'f32', 'f16'}}
  wave.broadcast %arg0 : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N, @K] of f16, <register>>
  return
}

// -----

func.func @self_index_wrong_dim() -> !wave.tensor<[@M] of i32, <register>> {
  // expected-error @below {{result dimension 'M' must match the specified dimension 'N'}}
  %0 = wave.self_index @N : !wave.tensor<[@M] of i32, <register>>
  return %0 : !wave.tensor<[@M] of i32, <register>>
}

// -----

func.func @self_index_float_type() -> !wave.tensor<[@N] of f32, <register>> {
  // expected-error @below {{result element type must be an integer type, got 'f32'}}
  %0 = wave.self_index @N : !wave.tensor<[@N] of f32, <register>>
  return %0 : !wave.tensor<[@N] of f32, <register>>
}

// -----

func.func @self_index_multi_dim() -> !wave.tensor<[@M, @N] of i32, <register>> {
  // expected-error @below {{result must be a 1-dimensional tensor, got rank 2}}
  %0 = wave.self_index @N : !wave.tensor<[@M, @N] of i32, <register>>
  return %0 : !wave.tensor<[@M, @N] of i32, <register>>
}

// -----

func.func @self_index_vector_float_type() -> vector<4xf32> {
  // expected-error @below {{result element type must be an integer type, got 'f32'}}
  %0 = wave.self_index @N : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Test that permute result shape must not be empty
func.func @permute_empty_result_shape(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input shape rank (2) does not match target shape rank (0)}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[] of f32, <register>>
  return
}

// -----

// Test that permute input and result element types must match
func.func @permute_element_type_mismatch(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{expected input and result elemental types to match, got 'f32', 'f16'}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @M] of f16, <register>>
  return
}

// -----

// Test that permute result shape dimensions must exist in input shape
func.func @permute_unknown_dimension(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input dimension 'M' is not present in result shape}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@P, @Q] of f32, <register>>
  return
}

// -----

// Test that permute result shape with one unknown dimension fails
func.func @permute_partial_unknown_dimension(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input dimension 'M' is not present in result shape}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @B] of f32, <register>>
  return
}

// -----

// Test that permute result shape with extra dimension not in input fails
func.func @permute_extra_dimension(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input shape rank (2) does not match target shape rank (3)}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @M, @K] of f32, <register>>
  return
}

// -----

// Test that permute result shape rank must match input shape rank
func.func @permute_result_rank_mismatch(%arg0: !wave.tensor<[@M, @N, @K] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input shape rank (3) does not match target shape rank (2)}}
  wave.permute %arg0 : !wave.tensor<[@M, @N, @K] of f32, <register>> to !wave.tensor<[@N, @M] of f32, <register>>
  return
}

// -----

// Test that permute result shape must be a permutation of input shape
func.func @permute_result_not_permutation(%arg0: !wave.tensor<[@M, @N] of f32, <register>>) {
  // expected-error @below {{'wave.permute' op input dimension 'M' is not present in result shape}}
  wave.permute %arg0 : !wave.tensor<[@M, @N] of f32, <register>> to !wave.tensor<[@N, @K] of f32, <register>>
  return
}

// -----

// Test apply_expr with too many result expressions (no combinator).
func.func @apply_expr_multi_result(%arg0: !wave.tensor<[@M] of i32>) {
  // expected-error @below {{in absence of a combinator, expression must produce exactly one result, but got 2}}
  "wave.apply_expr"(%arg0) {expr = #wave.expr_list<[#wave.operand<0>] -> (_Operand_0, _Operand_0 + 1)>} : (!wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i32>
}

// -----

func.func @apply_expr_unused_operand(%arg0: !wave.tensor<[@M] of i32>, %arg1: !wave.tensor<[@M] of i32>) {
  // expected-warning @below {{operand #1 is not used in the expression}}
  "wave.apply_expr"(%arg0, %arg1) {expr = #wave.expr_list<[#wave.operand<0>] -> (_Operand_0 + 1)>} : (!wave.tensor<[@M] of i32>, !wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i32>
  return
}

// -----

func.func @apply_expr_operand_overflow(%arg0: !wave.tensor<[@M] of i32>, %arg1: !wave.tensor<[@M] of i32>) {
  // expected-error @below {{expression uses operand #2 but there are only 2 operands}}
  wave.apply_expr(%arg0, %arg1) <[#wave.operand<0>, #wave.operand<1>, #wave.operand<2>] -> (_Operand_0 + _Operand_1 + _Operand_2)> : (!wave.tensor<[@M] of i32>, !wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i32>
}

// -----

func.func @apply_expr_non_integer_result(%arg0: !wave.tensor<[@M] of f32>) {
  // expected-error @below {{operates on integers only}}
  wave.apply_expr(%arg0) <[#wave.operand<0>] -> (_Operand_0 + 1)> : (!wave.tensor<[@M] of f32>) -> !wave.tensor<[@M] of f32>
}

// -----

func.func @apply_expr_symbol_not_in_hyperparam(%arg0: !wave.tensor<[@M] of i32>) attributes { wave.hyperparameters = #wave.hyperparameters<{M = 42}>} {
  // expected-error @below {{op attribute "expr" uses symbolic value #wave.symbol<"Z"> not provided as a hyperparameter}}
  // expected-note @below {{available symbols: M}}
  wave.apply_expr(%arg0) <[#wave.symbol<"Z">, #wave.operand<0>] -> (Z + _Operand_0)> : (!wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i32>
  return
}

// -----

func.func @apply_expr_non_register_operand(%arg0: !wave.tensor<[@M] of i32, <global>>) {
  // expected-error @below {{tensor operands must be in register or unspecified address space}}
  wave.apply_expr(%arg0) <[#wave.operand<0>] -> (_Operand_0 + 1)> : (!wave.tensor<[@M] of i32, <global>>) -> !wave.tensor<[@M] of i32>
  return
}

// -----

// Test apply_expr with min/max combinator and zero results.
func.func @apply_expr_minmax_zero_results(%arg0: !wave.tensor<[@M] of i32>) {
  // expected-error @below {{for min/max combinators, expression must produce at least one result}}
  wave.apply_expr(%arg0) max<[#wave.operand<0>] -> ()> : (!wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i32>
}

// -----

// Test apply_expr with comparison combinator and one result (requires exactly two).
func.func @apply_expr_comparison_one_result(%arg0: !wave.tensor<[@M] of i32>, %arg1: !wave.tensor<[@M] of i32>) {
  // expected-error @below {{for comparison combinators, expression must produce exactly two results}}
  wave.apply_expr(%arg0, %arg1) gt<[#wave.operand<0>, #wave.operand<1>] -> (_Operand_0)> : (!wave.tensor<[@M] of i32>, !wave.tensor<[@M] of i32>) -> !wave.tensor<[@M] of i1>
}
