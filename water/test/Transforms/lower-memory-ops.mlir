// RUN: water-opt %s --water-lower-memory-ops | FileCheck %s

// Test lowering of vector memory operations to AMDGPU global_load/store inline assembly

// CHECK-LABEL: func.func @simple_function
func.func @simple_function(%arg0: f32) -> f32 {
  // CHECK: return %arg0
  return %arg0 : f32
}

// CHECK-LABEL: func.func @vector_load
func.func @vector_load(%memref: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: arith.index_cast
  // CHECK: llvm.inttoptr
  // CHECK: llvm.inline_asm "global_load_b128 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store
func.func @vector_store(%memref: memref<1024xf32>, %offset: index, %data: vector<4xf32>) {
  // CHECK: memref.extract_aligned_pointer_as_index
  // CHECK: arith.index_cast
  // CHECK: llvm.inttoptr
  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @vector_load_2xf32
func.func @vector_load_2xf32(%memref: memref<1024xf32>, %offset: index) -> vector<2xf32> {
  // CHECK: llvm.inline_asm "global_load_b64 $0, $1, off", "=v,v"
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @load_store_sequence
func.func @load_store_sequence(%src: memref<1024xf32>, %dst: memref<1024xf32>, %offset: index) {
  // Test lowering of load/store sequence

  // CHECK: llvm.inline_asm "global_load_b128 $0, $1, off", "=v,v"
  %data = vector.load %src[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: llvm.inline_asm has_side_effects "global_store_b128 $0, $1, off", "v,v"
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return
  return
}
