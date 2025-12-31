// RUN: water-opt --water-reorder-affine-exprs %s | FileCheck %s

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (d1 + s0 + d0)>
// CHECK-DAG: #[[MAP_MUL:.*]] = affine_map<(d0)[s0, s1] -> (d0 * (s0 * s1))>

// CHECK-LABEL: func.func @test_nested_loops
func.func @test_nested_loops(%arg0: index, %arg1: index, %buf: memref<10x10xindex>) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      // Original: %j + %i + %arg0
      // %arg0 is invariant to both loops (hoistability = 2)
      // %i is invariant to inner loop only (hoistability = 1)
      // %j is not invariant to inner loop (hoistability = 0)
      // After reordering: most hoistable first, least hoistable last
      // CHECK: affine.apply #[[MAP]](%{{.*}}, %{{.*}})[%arg0]
      %0 = affine.apply affine_map<(d0, d1)[s0] -> (d0 + d1 + s0)>(%j, %i)[%arg0]
      memref.store %0, %buf[%i, %j] : memref<10x10xindex>
    }
  }
  return
}

// CHECK-LABEL: func.func @test_mul_reorder
func.func @test_mul_reorder(%arg0: index, %arg1: index, %buf: memref<10xindex>) {
  affine.for %i = 0 to 10 {
    // Multiplication should also be reordered.
    // %arg0 and %arg1 are more hoistable than %i.
    // CHECK: affine.apply #[[MAP_MUL]](%{{.*}})[%arg0, %arg1]
    %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 * s1)>(%i)[%arg0, %arg1]
    memref.store %0, %buf[%i] : memref<10xindex>
  }
  return
}

// CHECK-LABEL: func.func @test_no_reorder_single_loop
func.func @test_no_reorder_single_loop(%arg0: index, %buf: memref<10xindex>) {
  affine.for %i = 0 to 10 {
    // All operands have same hoistability in single loop, no reordering.
    // CHECK: affine.apply
    %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%i)[%arg0, %arg0]
    memref.store %0, %buf[%i] : memref<10xindex>
  }
  return
}
