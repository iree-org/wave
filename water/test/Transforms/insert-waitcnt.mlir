// RUN: water-opt %s --water-insert-waitcnt | FileCheck %s

// Smoke test to verify waitcnt insertion pass runs without crashing

// CHECK-LABEL: func.func @simple_function
func.func @simple_function(%arg0: f32) -> f32 {
  // CHECK: return %arg0
  return %arg0 : f32
}

// CHECK-LABEL: func.func @vector_load
func.func @vector_load(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: vector.load
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store
func.func @vector_store(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // CHECK: vector.store
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @load_store_sequence
func.func @load_store_sequence(%src: memref<1024xf32>, %dst: memref<1024xf32>, %offset: index) {
  // Test a simple load followed by a store
  // TODO: Eventually this should insert waitcnt between load and store

  // CHECK: vector.load
  %data = vector.load %src[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: vector.store
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return
  return
}
