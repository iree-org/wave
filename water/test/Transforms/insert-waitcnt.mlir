// RUN: water-opt %s --water-insert-waitcnt | FileCheck %s

// Test waitcnt insertion for vector memory operations

// CHECK-LABEL: func.func @single_load_use
func.func @single_load_use(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: vector.load
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @two_loads_use_in_reverse_order
//  CHECK-SAME: (%[[ARG0:.*]]: memref<1024xf32>, %[[ARG1:.*]]: memref<1024xf32>, %{{.*}}: index)
func.func @two_loads_use_in_reverse_order(%memrefA: memref<1024xf32>, %memrefB: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: %[[LOAD_A:.*]] = vector.load %[[ARG0]]
  // CHECK: %[[LOAD_B:.*]] = vector.load %[[ARG1]]
  %loadA = vector.load %memrefA[%offset] : memref<1024xf32>, vector<4xf32>
  %loadB = vector.load %memrefB[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(1)
  // CHECK-NEXT: %[[ADD_A:.*]] = arith.addf %[[LOAD_A]], %[[LOAD_A]]
  %addA = arith.addf %loadA, %loadA : vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[ADD_B:.*]] = arith.addf %[[LOAD_B]], %[[ADD_A]]
  %addB = arith.addf %loadB, %addA : vector<4xf32>

  // CHECK-NOT: amdgpu.memory_counter_wait

  // CHECK: return %[[ADD_B]]
  return %addB : vector<4xf32>
}
