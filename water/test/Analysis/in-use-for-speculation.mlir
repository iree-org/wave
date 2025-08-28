
// RUN: water-opt %s --water-test-in-use-analysis | FileCheck %s

// CHECK-LABEL: @all_static
// CHECK-SAME: __in_use_blockarg_0_0 = ["used"]
// CHECK-SAME: __in_use_results = []
func.func @all_static(%memref: memref<2x2xf32>) -> f32 {
  // CHECK: arith.constant
  // CHECK-SAME: __in_use_results = ["used"]
  %0 = arith.constant 0 : index
  // CHECK: arith.constant
  // CHECK-SAME: __in_use_results = ["used"]
  %1 = arith.constant 1 : index
  // CHECK: memref.load
  // CHECK-SAME: __in_use_results = ["unused"]
  %2 = memref.load %memref[%0, %1] : memref<2x2xf32>
  // CHECK: return
  // CHECK-SAME: __in_use_results = []
  return %2 : f32
}

// -----

// From now on, we are not checking empty results.

// Values that have not been processed by the analysis because they are never
// used in memory or control flow operations as either affecting the control
// flow and index computation or not are marked as "bottom". For the purposes
// of speculation, these values can be seen as unused.

// CHECK-LABEL: @loop
// CHECK-COUNT-4: "used"
func.func @loop(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) -> f32 {
  // CHECK: arith.constant
  // CHECK-SAME: __in_use_results = ["bottom"]
  %0 = arith.constant 0.0 : f32
  // CHECK: scf.for
  %1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %0) -> (f32) {
    // CHECK-NEXT: memref.load
    // CHECK-SAME: __in_use_results = ["bottom"]
    %2 = memref.load %memref[%i] : memref<2xf32>
    // CHECK: arith.addf
    // CHECK: __in_use_results = ["bottom"]
    %3 = arith.addf %acc, %2 : f32
    scf.yield %3 : f32
  }
  // CHECK: __in_use_blockarg_0_0 = ["used", "bottom"]
  // CHECK-SAME: __in_use_results = ["unused"]
  // CHECK: return
  return %1 : f32
}

// -----

// Values that are used for indexing or control flow purposes on one dataflow
// path and not used on another dataflow path, e.g. the loop induction variable
// that is both the subscript and the value written, are marked as "top". For
// the purposes of the speculation, these values can be seen as used.

// CHECK-LABEL: @loop_store
// CHECK-COUNT-4: "used"
func.func @loop_store(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step {
    // CHECK: arith.index_cast
    // CHECK-SAME: __in_use_results = ["unused"]
    %2 = arith.index_cast %i : index to i64
    // CHECK: arith.sitofp
    // CHECK-SAME: __in_use_results = ["unused"]
    %3 = arith.sitofp %2 : i64 to f32
    memref.store %3, %memref[%i] : memref<2xf32>
    scf.yield
  }
  // CHECK: __in_use_blockarg_0_0 = ["top"]
  return
}
