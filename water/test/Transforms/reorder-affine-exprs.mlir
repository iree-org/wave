// RUN: water-opt --water-reorder-affine-exprs %s | FileCheck %s

// Test that commutative operations are canonicalized to enable CSE.
// The pass uses hash-based scoring to find optimal orderings.

// CHECK-DAG: #[[MAP_ADD:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-DAG: #[[MAP_MUL:.*]] = affine_map<(d0)[s0, s1] -> ((d0 * s0) * s1)>
// CHECK-DAG: #[[MAP_NESTED:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>
// CHECK-DAG: #[[MAP_MIXED:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + d1 * 3 + d2)>
// CHECK-DAG: #[[MAP_NONCOMM:.*]] = affine_map<(d0)[s0] -> ((d0 floordiv s0) mod 8)>
// CHECK-DAG: #[[MAP_SYMDIM:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 + s0 + d1 + s1)>
// CHECK-DAG: #[[MAP_CONST:.*]] = affine_map<(d0, d1) -> (d0 + d1 + 15)>
// CHECK-DAG: #[[MAP_DEEP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1 + d2 + d3 + d4)>
// CHECK-DAG: #[[MAP_SINGLE:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[MAP_MULCHAIN:.*]] = affine_map<(d0)[s0, s1, s2] -> (((d0 * s0) * s1) * s2)>

// CHECK-LABEL: func.func @test_canonicalize_add
func.func @test_canonicalize_add(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // Two expressions with same operands in different order should be canonicalized.
  // Both should use the same affine map after reordering.
  // CHECK: %[[V0:.*]] = affine.apply #[[MAP_ADD]]
  // CHECK: %[[V1:.*]] = affine.apply #[[MAP_ADD]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %arg1, %arg2)
  %1 = affine.apply affine_map<(d0, d1, d2) -> (d2 + d0 + d1)>(%arg0, %arg1, %arg2)
  return %0, %1 : index, index
}

// CHECK-LABEL: func.func @test_canonicalize_mul
func.func @test_canonicalize_mul(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // Multiplication with symbols should also be canonicalized.
  // Both should use the same affine map after reordering.
  // CHECK: %[[V0:.*]] = affine.apply #[[MAP_MUL]]
  // CHECK: %[[V1:.*]] = affine.apply #[[MAP_MUL]]
  %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 * s1)>(%arg0)[%arg1, %arg2]
  %1 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s1 * s0)>(%arg0)[%arg1, %arg2]
  return %0, %1 : index, index
}

// CHECK-LABEL: func.func @test_nested_commutative
func.func @test_nested_commutative(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  // Nested commutative operations are flattened and reordered.
  // CHECK: affine.apply #[[MAP_NESTED]]
  %0 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d0 + d1) + (d2 + d3))>(%arg0, %arg1, %arg2, %arg3)
  return %0 : index
}

// CHECK-LABEL: func.func @test_mixed_ops
func.func @test_mixed_ops(%arg0: index, %arg1: index, %arg2: index) -> index {
  // Mixed operations: (a * 2) + (b * 3) + c
  // Reordered to a canonical form.
  // CHECK: affine.apply #[[MAP_MIXED]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> ((d0 * 2) + (d1 * 3) + d2)>(%arg0, %arg1, %arg2)
  return %0 : index
}

// CHECK-LABEL: func.func @test_non_commutative_preserved
func.func @test_non_commutative_preserved(%arg0: index, %arg1: index) -> index {
  // Non-commutative operations (floordiv, mod) should NOT be reordered.
  // CHECK: affine.apply #[[MAP_NONCOMM]]
  %0 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv s0) mod 8)>(%arg0)[%arg1]
  return %0 : index
}

// CHECK-LABEL: func.func @test_common_subexpr_ordering
func.func @test_common_subexpr_ordering(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  // The key test: expressions that share sub-expressions should be ordered
  // to maximize hash hits. Both expressions should canonicalize to the same form.
  // CHECK: %[[V0:.*]] = affine.apply #[[MAP_ADD]]
  // CHECK: %[[V1:.*]] = affine.apply #[[MAP_ADD]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %arg1, %arg2)
  %1 = affine.apply affine_map<(d0, d1, d2) -> (d1 + d0 + d2)>(%arg0, %arg1, %arg2)
  return %0, %1 : index, index
}

// CHECK-LABEL: func.func @test_symbols_and_dims
func.func @test_symbols_and_dims(%arg0: index, %arg1: index) -> index {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {
      // Mix of dimensions and symbols in commutative expression.
      // CHECK: affine.apply #[[MAP_SYMDIM]]
      %0 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d0 + s0 + d1 + s1)>(%i, %j)[%arg0, %arg1]
    }
  }
  return %arg0 : index
}

// CHECK-LABEL: func.func @test_constant_in_expr
func.func @test_constant_in_expr(%arg0: index, %arg1: index) -> index {
  // Constants are folded: 5 + 10 = 15.
  // CHECK: affine.apply #[[MAP_CONST]]
  %0 = affine.apply affine_map<(d0, d1) -> (d0 + 5 + d1 + 10)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func.func @test_deep_nesting
func.func @test_deep_nesting(%a: index, %b: index, %c: index, %d: index, %e: index) -> index {
  // Deeply nested commutative operations are flattened.
  // CHECK: affine.apply #[[MAP_DEEP]]
  %0 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (((d0 + d1) + d2) + (d3 + d4))>(%a, %b, %c, %d, %e)
  return %0 : index
}

// CHECK-LABEL: func.func @test_single_term
func.func @test_single_term(%arg0: index) -> index {
  // Single term - nothing to reorder.
  // CHECK: affine.apply #[[MAP_SINGLE]]
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  return %0 : index
}

// CHECK-LABEL: func.func @test_same_value_multiple_times
func.func @test_same_value_multiple_times(%arg0: index) -> index {
  // Same value used multiple times in expression.
  // CHECK: affine.apply #[[MAP_ADD]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %arg0, %arg0)
  return %0 : index
}

// CHECK-LABEL: func.func @test_multiplication_chain
func.func @test_multiplication_chain(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index) {
  // Multiple multiplication chains that should be canonicalized to same form.
  // CHECK: %[[M0:.*]] = affine.apply #[[MAP_MULCHAIN]]
  // CHECK: %[[M1:.*]] = affine.apply #[[MAP_MULCHAIN]]
  %0 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 * s0 * s1 * s2)>(%arg0)[%arg1, %arg2, %arg3]
  %1 = affine.apply affine_map<(d0)[s0, s1, s2] -> (d0 * s2 * s1 * s0)>(%arg0)[%arg1, %arg2, %arg3]
  return %0, %1 : index, index
}
