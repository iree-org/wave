// RUN: water-opt %s --water-insert-waitcnt | FileCheck %s

// Test waitcnt insertion for vector memory operations

// CHECK-LABEL: func.func @single_load_use
func.func @single_load_use(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: vector.load
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: amdgpu.memory_counter_wait load(0) store(0)
  // CHECK-NEXT: return
  return %result : vector<4xf32>
}
