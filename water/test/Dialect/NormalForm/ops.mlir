// RUN: water-opt %s | water-opt | FileCheck %s
// RUN: water-opt %s --mlir-print-op-generic | water-opt | FileCheck %s

//-----------------------------------------------------------------------------
// Test dialect normal form attribute tests (single attribute).
//-----------------------------------------------------------------------------

// no_index_types passes when no index types are used.
// CHECK-LABEL: normalform.module @no_index_types_valid
// CHECK-SAME:  [#water_test.no_index_types]
normalform.module @no_index_types_valid [#water_test.no_index_types] {
  func.func @f(%arg: i32) -> f32 {
    %cst = arith.constant 0.0 : f32
    return %cst : f32
  }
}

// no_invalid_ops passes when no division operations are present.
// CHECK-LABEL: normalform.module @no_invalid_ops_valid
// CHECK-SAME:  [#water_test.no_invalid_ops]
normalform.module @no_invalid_ops_valid [#water_test.no_invalid_ops] {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.mulf %a, %b : f32
    return %0 : f32
  }
}

// no_invalid_attrs passes when no "invalid" string attributes are present.
// CHECK-LABEL: normalform.module @no_invalid_attrs_valid
// CHECK-SAME:  [#water_test.no_invalid_attrs]
normalform.module @no_invalid_attrs_valid [#water_test.no_invalid_attrs] {
  func.func @f() attributes {foo = "valid", bar = 42 : i32} {
    return
  }
}

// no_invalid_ops passes with scf.for block arguments (iter_args and induction
// variable). This tests that block arguments in nested regions are properly
// walked and verified.
// CHECK-LABEL: normalform.module @no_invalid_ops_nested_block_args
// CHECK-SAME:  [#water_test.no_invalid_ops]
normalform.module @no_invalid_ops_nested_block_args [#water_test.no_invalid_ops] {
  func.func @f() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = arith.constant 0.0 : f32
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f32) {
      %one = arith.constant 1.0 : f32
      %sum = arith.addf %acc, %one : f32
      scf.yield %sum : f32
    }
    return %result : f32
  }
}

//-----------------------------------------------------------------------------
// Test dialect normal form attribute tests (multiple attributes).
//-----------------------------------------------------------------------------

// Two attributes pass with valid IR.
// CHECK-LABEL: normalform.module @two_attrs_valid
// CHECK-SAME:  [#water_test.no_index_types, #water_test.no_invalid_ops]
normalform.module @two_attrs_valid [#water_test.no_index_types, #water_test.no_invalid_ops] {
  func.func @f(%arg: i32) {
    return
  }
}

// All three attributes pass with valid IR.
// CHECK-LABEL: normalform.module @all_attrs_valid
// CHECK-SAME:  [#water_test.no_index_types, #water_test.no_invalid_ops, #water_test.no_invalid_attrs]
normalform.module @all_attrs_valid [#water_test.no_index_types, #water_test.no_invalid_ops, #water_test.no_invalid_attrs] {
  func.func @f(%arg: i32) attributes {foo = "valid"} {
    return
  }
}

//-----------------------------------------------------------------------------
// Module without name.
//-----------------------------------------------------------------------------

// Anonymous module with single attribute.
// CHECK-LABEL: normalform.module [#water_test.no_invalid_ops]
normalform.module [#water_test.no_invalid_ops] {
  func.func @f() {
    return
  }
}
