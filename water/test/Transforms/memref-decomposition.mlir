// RUN: water-opt %s --water-memref-decomposition --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD:.*]] = ptr.load %[[PTR_ADD]] : !ptr.ptr<#llvm.address_space<0>> -> f32
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
func.func @store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: ptr.store %[[ARG3]], %[[PTR_ADD]] : f32, !ptr.ptr<#llvm.address_space<0>>
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}

// CHECK-LABEL: func.func @load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> f16 {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<4x8x16xf16> to memref<4x8x16xf16, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<4x8x16xf16, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD:.*]] = ptr.load %[[PTR_ADD]] : !ptr.ptr<#llvm.address_space<0>> -> f16
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<4x8x16xf16>
  return %0 : f16
}

// CHECK-LABEL: func.func @vector_load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @vector_load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> vector<4xf32> {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD:.*]] = ptr.load %[[PTR_ADD]] : !ptr.ptr<#llvm.address_space<0>> -> vector<4xf32>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<4xf32>)
func.func @vector_store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: vector<4xf32>) {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: ptr.store %[[ARG3]], %[[PTR_ADD]] : vector<4xf32>, !ptr.ptr<#llvm.address_space<0>>
  vector.store %val, %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @vector_load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @vector_load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> vector<8xf16> {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<4x8x16xf16> to memref<4x8x16xf16, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<4x8x16xf16, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[IDX]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD:.*]] = ptr.load %[[PTR_ADD]] : !ptr.ptr<#llvm.address_space<0>> -> vector<8xf16>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j, %k] : memref<4x8x16xf16>, vector<8xf16>
  return %0 : vector<8xf16>
}

// CHECK-LABEL: func.func @multiple_loads
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multiple_loads(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX0:.*]] = affine.apply
  // CHECK: %[[PTR_ADD0:.*]] = ptr.ptr_add %[[PTR]], %[[IDX0]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD0:.*]] = ptr.load %[[PTR_ADD0]] : !ptr.ptr<#llvm.address_space<0>> -> f32
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>

  // CHECK: %[[IDX1:.*]] = affine.apply
  // CHECK: %[[PTR_ADD1:.*]] = ptr.ptr_add %[[PTR]], %[[IDX1]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD1:.*]] = ptr.load %[[PTR_ADD1]] : !ptr.ptr<#llvm.address_space<0>> -> f32
  %1 = memref.load %arg0[%j, %i] : memref<10x20xf32>

  // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]]
  %2 = arith.addf %0, %1 : f32
  // CHECK: return %[[ADD]]
  return %2 : f32
}

// CHECK-LABEL: func.func @different_types
// CHECK-SAME: (%[[ARG0:.*]]: memref<8x16xi32>, %[[ARG1:.*]]: memref<4x8xf64>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @different_types(%arg0: memref<8x16xi32>, %arg1: memref<4x8xf64>, %i: index, %j: index) -> (i32, f64) {
  // Canonicalizer processes arg1 first
  // CHECK: %[[CAST1:.*]] = memref.memory_space_cast %[[ARG1]] : memref<4x8xf64> to memref<4x8xf64, #llvm.address_space<0>>
  // CHECK: %[[PTR1:.*]] = ptr.to_ptr %[[CAST1]] : memref<4x8xf64, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // Then arg0
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[ARG0]] : memref<8x16xi32> to memref<8x16xi32, #llvm.address_space<0>>
  // CHECK: %[[PTR0:.*]] = ptr.to_ptr %[[CAST0]] : memref<8x16xi32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[IDX0:.*]] = affine.apply
  // CHECK: %[[PTR_ADD0:.*]] = ptr.ptr_add %[[PTR0]], %[[IDX0]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD0:.*]] = ptr.load %[[PTR_ADD0]] : !ptr.ptr<#llvm.address_space<0>> -> i32
  %0 = memref.load %arg0[%i, %j] : memref<8x16xi32>

  // CHECK: %[[IDX1:.*]] = affine.apply
  // CHECK: %[[PTR_ADD1:.*]] = ptr.ptr_add %[[PTR1]], %[[IDX1]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[LOAD1:.*]] = ptr.load %[[PTR_ADD1]] : !ptr.ptr<#llvm.address_space<0>> -> f64
  %1 = memref.load %arg1[%i, %j] : memref<4x8xf64>
  // CHECK: return %[[LOAD0]], %[[LOAD1]]
  return %0, %1 : i32, f64
}

// CHECK-LABEL: func.func @reinterpret_cast_0d
// CHECK-SAME: (%[[BASE:.*]]: memref<f32>)
func.func @reinterpret_cast_0d(%base: memref<f32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[BASE]] : memref<f32> to memref<f32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST0]] : memref<f32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[FROM_PTR:.*]] = ptr.from_ptr %[[PTR]] : <#llvm.address_space<0>> -> memref<f32, #llvm.address_space<0>>
  // CHECK: %[[CAST1:.*]] = memref.memory_space_cast %[[FROM_PTR]] : memref<f32, #llvm.address_space<0>> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST1]] to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<1x2xf32, strided<[3, 4]>>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[REINTERPRET]] : memref<1x2xf32, strided<[3, 4]>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK: return %[[CAST2]]
  %0 = memref.reinterpret_cast %base to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}


// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: (%[[BASE:.*]]: memref<100xf32>, %[[OFFSET:.*]]: index, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index)
func.func @reinterpret_cast(%base: memref<100xf32>, %offset: index, %size0: index, %size1: index, %stride0: index, %stride1: index) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[BASE]] : memref<100xf32> to memref<100xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR:.*]] = ptr.to_ptr %[[CAST0]] : memref<100xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[OFF:.*]] = affine.apply
  // CHECK: %[[PTR_ADD:.*]] = ptr.ptr_add %[[PTR]], %[[OFF]] : !ptr.ptr<#llvm.address_space<0>>, index
  // CHECK: %[[FROM_PTR:.*]] = ptr.from_ptr %[[PTR_ADD]] : <#llvm.address_space<0>> -> memref<f32, #llvm.address_space<0>>
  // CHECK: %[[CAST1:.*]] = memref.memory_space_cast %[[FROM_PTR]] : memref<f32, #llvm.address_space<0>> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST1]] to offset: [0], sizes: [%[[SIZE0]], %[[SIZE1]]], strides: [%[[STRIDE0]], %[[STRIDE1]]] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK: return %[[REINTERPRET]]
  %0 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%size0, %size1], strides: [%stride0, %stride1] : memref<100xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// CHECK-LABEL: func.func @fat_raw_buffer_cast
// CHECK-SAME: (%[[ARG:.*]]: memref<10x20xf32>)
func.func @fat_raw_buffer_cast(%arg0: memref<10x20xf32>) -> memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>> {
  // CHECK: %[[CAST0:.*]] = memref.memory_space_cast %[[ARG]] : memref<10x20xf32> to memref<10x20xf32, #llvm.address_space<0>>
  // CHECK: %[[PTR0:.*]] = ptr.to_ptr %[[CAST0]] : memref<10x20xf32, #llvm.address_space<0>> -> <#llvm.address_space<0>>
  // CHECK: %[[FROM_PTR0:.*]] = ptr.from_ptr %[[PTR0]] : <#llvm.address_space<0>> -> memref<f32, #llvm.address_space<0>>
  // CHECK: %[[CAST1:.*]] = memref.memory_space_cast %[[FROM_PTR0]] : memref<f32, #llvm.address_space<0>> to memref<f32>
  // CHECK: %[[FAT:.*]] = amdgpu.fat_raw_buffer_cast %[[CAST1]] : memref<f32> to memref<f32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: %[[CAST2:.*]] = memref.memory_space_cast %[[FAT]] : memref<f32, #amdgpu.address_space<fat_raw_buffer>> to memref<f32, #llvm.address_space<7>>
  // CHECK: %[[PTR1:.*]] = ptr.to_ptr %[[CAST2]] : memref<f32, #llvm.address_space<7>> -> <#llvm.address_space<7>>
  // CHECK: %[[FROM_PTR1:.*]] = ptr.from_ptr %[[PTR1]] : <#llvm.address_space<7>> -> memref<f32, #llvm.address_space<7>>
  // CHECK: %[[CAST3:.*]] = memref.memory_space_cast %[[FROM_PTR1]] : memref<f32, #llvm.address_space<7>> to memref<f32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST3]] to offset: [0], sizes: [10, 20], strides: [20, 1] : memref<f32, #amdgpu.address_space<fat_raw_buffer>> to memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: return %[[REINTERPRET]]
  %0 = amdgpu.fat_raw_buffer_cast %arg0 : memref<10x20xf32> to memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %0 : memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
}
