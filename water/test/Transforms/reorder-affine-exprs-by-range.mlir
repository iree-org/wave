// RUN: water-opt --water-reorder-affine-exprs-by-range %s | FileCheck %s

// Test that commutative operations are reordered to minimize intermediate result widths.
// The pass uses integer range analysis to compute bit widths.

// CHECK-DAG: #[[MAP_CONST1:.*]] = affine_map<(d0, d1) -> (d0 + 105)>
// CHECK-DAG: #[[MAP_ADD3:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
// CHECK-DAG: #[[MAP_MUL:.*]] = affine_map<(d0)[s0, s1] -> (((d0 * s0) * s1) * 2)>
// CHECK-DAG: #[[MAP_MIXED:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + d1 * 3 + d2)>
// CHECK-DAG: #[[MAP_CONST2:.*]] = affine_map<(d0) -> (d0 + 1005)>
// CHECK-DAG: #[[MAP_ADD4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>
// CHECK-DAG: #[[MAP_LOOP:.*]] = affine_map<(d0, d1) -> (d0 + d1 + 1000)>
// CHECK-DAG: #[[MAP_SYM:.*]] = affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>
// CHECK-DAG: #[[MAP_NONCOMM:.*]] = affine_map<(d0)[s0] -> ((d0 floordiv s0) mod 8)>
// CHECK-DAG: #[[MAP_SINGLE:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[MAP_ALLCONST:.*]] = affine_map<() -> (60)>
// CHECK-DAG: #[[MAP_SMALL:.*]] = affine_map<(d0, d1) -> (d0 + d1 + 128)>
// CHECK-DAG: #[[MAP_POW2:.*]] = affine_map<(d0) -> (d0 + 274)>

// CHECK-LABEL: func.func @test_add_with_constants
func.func @test_add_with_constants(%arg0: index, %arg1: index) -> index {
  // Constants are folded: 100 + 5 = 105.
  // CHECK: affine.apply #[[MAP_CONST1]]
  %0 = affine.apply affine_map<(d0, d1) -> (d0 + 100 + 5)>(%arg0, %arg1)
  return %0 : index
}

// CHECK-LABEL: func.func @test_add_ordering
func.func @test_add_ordering(%arg0: index, %arg1: index, %arg2: index) -> index {
  // With different range operands, ordered to minimize intermediate widths.
  // CHECK: affine.apply #[[MAP_ADD3]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %arg1, %arg2)
  return %0 : index
}

// CHECK-LABEL: func.func @test_mul_with_constants
func.func @test_mul_with_constants(%arg0: index, %arg1: index) -> index {
  // Multiplication with constants: order optimized for intermediate width.
  // CHECK: affine.apply #[[MAP_MUL]]
  %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * 2 * s0 * s1)>(%arg0)[%arg1, %arg1]
  return %0 : index
}

// CHECK-LABEL: func.func @test_mixed_ops_range
func.func @test_mixed_ops_range(%arg0: index, %arg1: index, %arg2: index) -> index {
  // Mixed operations with different range characteristics.
  // CHECK: affine.apply #[[MAP_MIXED]]
  %0 = affine.apply affine_map<(d0, d1, d2) -> ((d0 * 2) + (d1 * 3) + d2)>(%arg0, %arg1, %arg2)
  return %0 : index
}

// CHECK-LABEL: func.func @test_constant_folding_context
func.func @test_constant_folding_context(%arg0: index) -> index {
  // Constants are folded: 1000 + 5 = 1005.
  // CHECK: affine.apply #[[MAP_CONST2]]
  %0 = affine.apply affine_map<(d0) -> (d0 + 1000 + 5)>(%arg0)
  return %0 : index
}

// CHECK-LABEL: func.func @test_nested_expressions
func.func @test_nested_expressions(%a: index, %b: index, %c: index, %d: index) -> index {
  // Nested additions are flattened and reordered.
  // CHECK: affine.apply #[[MAP_ADD4]]
  %0 = affine.apply affine_map<(d0, d1, d2, d3) -> ((d0 + d1) + (d2 + d3))>(%a, %b, %c, %d)
  return %0 : index
}

// CHECK-LABEL: func.func @test_in_loop_context
func.func @test_in_loop_context(%arg0: index, %arg1: index) -> index {
  // In loop context, induction variables have known ranges.
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 100 {
      // %i: [0,9], %j: [0,99], ordered to minimize intermediate results.
      // CHECK: affine.apply #[[MAP_LOOP]]
      %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1 + 1000)>(%i, %j)
    }
  }
  return %arg0 : index
}

// CHECK-LABEL: func.func @test_symbols_and_dims_range
func.func @test_symbols_and_dims_range(%arg0: index, %arg1: index) -> index {
  affine.for %i = 0 to 16 {
    // %i has narrow range [0, 15] = 4 bits.
    // CHECK: affine.apply #[[MAP_SYM]]
    %0 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%i)[%arg0, %arg1]
  }
  return %arg0 : index
}

// CHECK-LABEL: func.func @test_non_commutative_preserved
func.func @test_non_commutative_preserved(%arg0: index, %arg1: index) -> index {
  // Non-commutative operations should not be reordered.
  // CHECK: affine.apply #[[MAP_NONCOMM]]
  %0 = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv s0) mod 8)>(%arg0)[%arg1]
  return %0 : index
}

// CHECK-LABEL: func.func @test_single_operand
func.func @test_single_operand(%arg0: index) -> index {
  // Single operand - nothing to reorder.
  // CHECK: affine.apply #[[MAP_SINGLE]]
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  return %0 : index
}

// CHECK-LABEL: func.func @test_all_constants
func.func @test_all_constants() -> index {
  // All constants are folded: 10 + 20 + 30 = 60.
  // CHECK: affine.apply #[[MAP_ALLCONST]]
  %0 = affine.apply affine_map<() -> (10 + 20 + 30)>()
  return %0 : index
}

// CHECK-LABEL: func.func @test_small_range_loop
func.func @test_small_range_loop(%arg0: index) -> index {
  // Loop with small range should optimize ordering.
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      // %i: [0,3] = 2 bits, %j: [0,7] = 3 bits.
      // CHECK: affine.apply #[[MAP_SMALL]]
      %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1 + 128)>(%i, %j)
    }
  }
  return %arg0 : index
}

// CHECK-LABEL: func.func @test_power_of_two_constants
func.func @test_power_of_two_constants(%arg0: index) -> index {
  // Powers of two are folded: 256 + 16 + 2 = 274.
  // CHECK: affine.apply #[[MAP_POW2]]
  %0 = affine.apply affine_map<(d0) -> (d0 + 256 + 16 + 2)>(%arg0)
  return %0 : index
}
