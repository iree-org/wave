// RUN: water-opt --water-compose-affine-arith %s | FileCheck %s

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[MAP_MUL:.*]] = affine_map<(d0)[s0, s1] -> ((d0 * s0) * s1)>
// CHECK-DAG: #[[MAP_MUL1:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>

// CHECK-LABEL: func.func @test_addi_with_affine_apply
func.func @test_addi_with_affine_apply(%arg0: index, %arg1: index, %arg2: index) -> index {
  // CHECK: %[[RES:.*]] = affine.apply #[[MAP]](%arg0)[%arg1, %arg2]
  // CHECK-NOT: arith.addi
  // CHECK: return %[[RES]]
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%arg1]
  %1 = arith.addi %0, %arg2 overflow<nsw> : index
  return %1 : index
}

// CHECK-LABEL: func.func @test_addi_reversed
func.func @test_addi_reversed(%arg0: index, %arg1: index, %arg2: index) -> index {
  // CHECK: %[[RES:.*]] = affine.apply #[[MAP]](%arg0)[%arg1, %arg2]
  // CHECK-NOT: arith.addi
  // CHECK: return %[[RES]]
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%arg1]
  %1 = arith.addi %arg2, %0 overflow<nsw> : index
  return %1 : index
}

// CHECK-LABEL: func.func @test_addi_without_nsw
func.func @test_addi_without_nsw(%arg0: index, %arg1: index, %arg2: index) -> index {
  // CHECK: %[[V0:.*]] = affine.apply #[[MAP1]](%arg0)[%arg1]
  // CHECK: arith.addi %[[V0]], %arg2
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg0)[%arg1]
  %1 = arith.addi %0, %arg2 : index
  return %1 : index
}

// CHECK-LABEL: func.func @test_addi_no_affine
func.func @test_addi_no_affine(%arg0: index, %arg1: index) -> index {
  // CHECK: arith.addi %arg0, %arg1
  %0 = arith.addi %arg0, %arg1 overflow<nsw> : index
  return %0 : index
}

// CHECK-LABEL: func.func @test_muli_with_affine_apply
func.func @test_muli_with_affine_apply(%arg0: index, %arg1: index, %arg2: index) -> index {
  // CHECK: %[[RES:.*]] = affine.apply #[[MAP_MUL]](%arg0)[%arg1, %arg2]
  // CHECK-NOT: arith.muli
  // CHECK: return %[[RES]]
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%arg0)[%arg1]
  %1 = arith.muli %0, %arg2 overflow<nsw> : index
  return %1 : index
}

// CHECK-LABEL: func.func @test_muli_without_nsw
func.func @test_muli_without_nsw(%arg0: index, %arg1: index, %arg2: index) -> index {
  // CHECK: %[[V0:.*]] = affine.apply #[[MAP_MUL1]](%arg0)[%arg1]
  // CHECK: arith.muli %[[V0]], %arg2
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%arg0)[%arg1]
  %1 = arith.muli %0, %arg2 : index
  return %1 : index
}
