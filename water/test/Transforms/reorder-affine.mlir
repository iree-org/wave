// RUN: water-opt --water-reorder-affine %s | FileCheck %s

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0 + s0 + d1)>

// CHECK-LABEL: func.func @test_reorder_in_loop
func.func @test_reorder_in_loop(%arg0: index, %arg1: index) {
  affine.for %i = 0 to 10 {
    // Loop-invariant %arg0 should be moved to symbol position.
    // CHECK: affine.apply #[[MAP]](%{{.*}})[%arg0]
    %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %arg0)
  }
  return
}

// CHECK-LABEL: func.func @test_nested_loops
func.func @test_nested_loops(%arg0: index, %arg1: index) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      // Loop-invariant %arg0 should be moved to symbol position.
      // Induction variables %i and %j remain as dimensions.
      // CHECK: affine.apply #[[MAP1]](%{{.*}}, %{{.*}})[%arg0]
      %0 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %i, %j)
    }
  }
  return
}
