// RUN: water-opt %s -lower-normalform-module --mlir-print-local-scope --split-input-file | FileCheck %s

//-----------------------------------------------------------------------------
// Test lowering of normalform.module to builtin.module.
//-----------------------------------------------------------------------------

// Test that a top-level normalform.module is inlined into the root module.
// CHECK: module {
// CHECK-NOT: normalform.module
// CHECK-NOT: module {
// CHECK:   func.func @inlined_into_root()
// CHECK: }
normalform.module [] {
  func.func @inlined_into_root() {
    return
  }
}

// -----

// Test that a named normalform.module is inlined into the root module.
// CHECK: module {
// CHECK-NOT: normalform.module
// CHECK-NOT: module {
// CHECK:   func.func @from_named_module()
// CHECK: }
normalform.module @named [] {
  func.func @from_named_module() {
    return
  }
}

// -----

// Test that multiple operations are preserved when inlining.
// CHECK: module {
// CHECK:   func.func @first()
// CHECK:   func.func @second()
// CHECK:   func.func @third()
// CHECK: }
normalform.module [] {
  func.func @first() {
    return
  }
  func.func @second() {
    return
  }
  func.func @third() {
    return
  }
}
