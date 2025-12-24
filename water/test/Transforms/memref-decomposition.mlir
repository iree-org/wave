// RUN: water-opt %s --water-memref-decomposition --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> f32
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>
  return %0 : f32
}

// CHECK-LABEL: func.func @store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
func.func @store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: f32) {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.store %[[ARG3]], %[[GEP]] : f32, !llvm.ptr
  memref.store %val, %arg0[%i, %j] : memref<10x20xf32>
  return
}

// CHECK-LABEL: func.func @load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> f16 {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x8x16xf16> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map1()[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> f16
  // CHECK: return %[[LOAD]]
  %0 = memref.load %arg0[%i, %j, %k] : memref<4x8x16xf16>
  return %0 : f16
}

// CHECK-LABEL: func.func @vector_load_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @vector_load_2d(%arg0: memref<10x20xf32>, %i: index, %j: index) -> vector<4xf32> {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> vector<4xf32>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @vector_store_2d
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<4xf32>)
func.func @vector_store_2d(%arg0: memref<10x20xf32>, %i: index, %j: index, %val: vector<4xf32>) {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.store %[[ARG3]], %[[GEP]] : vector<4xf32>, !llvm.ptr
  vector.store %val, %arg0[%i, %j] : memref<10x20xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func.func @vector_load_3d
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8x16xf16>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @vector_load_3d(%arg0: memref<4x8x16xf16>, %i: index, %j: index, %k: index) -> vector<8xf16> {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x8x16xf16> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: %[[IDX:.*]] = affine.apply #map1()[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> vector<8xf16>
  // CHECK: return %[[LOAD]]
  %0 = vector.load %arg0[%i, %j, %k] : memref<4x8x16xf16>, vector<8xf16>
  return %0 : vector<8xf16>
}

// CHECK-LABEL: func.func @multiple_loads
// CHECK-SAME: (%[[ARG0:.*]]: memref<10x20xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @multiple_loads(%arg0: memref<10x20xf32>, %i: index, %j: index) -> f32 {
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX0:.*]] = affine.apply #map()[%[[ARG1]], %[[ARG2]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX0]] : index to i64
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD0:.*]] = llvm.load %[[GEP0]] : !llvm.ptr -> f32
  %0 = memref.load %arg0[%i, %j] : memref<10x20xf32>

  // CHECK: %[[IDX1:.*]] = affine.apply #map()[%[[ARG2]], %[[ARG1]]]
  // CHECK: %[[CAST2:.*]] = arith.index_cast %[[IDX1]] : index to i64
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD1:.*]] = llvm.load %[[GEP1]] : !llvm.ptr -> f32
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
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : memref<4x8xf64> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT0:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // Then arg0
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<8x16xi32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %[[CAST1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[IDX0:.*]] = affine.apply #map2()[%[[ARG2]], %[[ARG3]]]
  // CHECK: %[[CAST2:.*]] = arith.index_cast %[[IDX0]] : index to i64
  // CHECK: %[[GEP0:.*]] = llvm.getelementptr nusw %[[EXTRACT1]][%[[CAST2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD0:.*]] = llvm.load %[[GEP0]] : !llvm.ptr -> i32
  %0 = memref.load %arg0[%i, %j] : memref<8x16xi32>

  // CHECK: %[[IDX1:.*]] = affine.apply #map3()[%[[ARG2]], %[[ARG3]]]
  // CHECK: %[[CAST3:.*]] = arith.index_cast %[[IDX1]] : index to i64
  // CHECK: %[[GEP1:.*]] = llvm.getelementptr nusw %[[EXTRACT0]][%[[CAST3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[LOAD1:.*]] = llvm.load %[[GEP1]] : !llvm.ptr -> f64
  %1 = memref.load %arg1[%i, %j] : memref<4x8xf64>
  // CHECK: return %[[LOAD0]], %[[LOAD1]]
  return %0, %1 : i32, f64
}

// CHECK-LABEL: func.func @reinterpret_cast_0d
// CHECK-SAME: (%[[BASE:.*]]: memref<f32>)
func.func @reinterpret_cast_0d(%base: memref<f32>) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[C0_I64:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<f32> to !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[EXTRACT]], %[[POISON]][0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[EXTRACT]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT2:.*]] = llvm.insertvalue %[[C0_I64]], %[[INSERT1]][2] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[INSERT2]] : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST1]] to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<1x2xf32, strided<[3, 4]>>
  // CHECK: %[[CAST2:.*]] = memref.cast %[[REINTERPRET]] : memref<1x2xf32, strided<[3, 4]>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK: return %[[CAST2]]
  %0 = memref.reinterpret_cast %base to offset: [0], sizes: [1, 2], strides: [3, 4] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}


// CHECK-LABEL: func.func @reinterpret_cast
// CHECK-SAME: (%[[BASE:.*]]: memref<100xf32>, %[[OFFSET:.*]]: index, %[[SIZE0:.*]]: index, %[[SIZE1:.*]]: index, %[[STRIDE0:.*]]: index, %[[STRIDE1:.*]]: index)
func.func @reinterpret_cast(%base: memref<100xf32>, %offset: index, %size0: index, %size1: index, %stride0: index, %stride1: index) -> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK: %[[C0_I64:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[BASE]] : memref<100xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFF:.*]] = affine.apply #map4()[%[[OFFSET]]]
  // CHECK: %[[CAST1:.*]] = arith.index_cast %[[OFF]] : index to i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[GEP]], %[[POISON]][0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[GEP]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT2:.*]] = llvm.insertvalue %[[C0_I64]], %[[INSERT1]][2] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[INSERT2]] : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST2]] to offset: [0], sizes: [%[[SIZE0]], %[[SIZE1]]], strides: [%[[STRIDE0]], %[[STRIDE1]]] : memref<f32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  // CHECK: return %[[REINTERPRET]]
  %0 = memref.reinterpret_cast %base to offset: [%offset], sizes: [%size0, %size1], strides: [%stride0, %stride1] : memref<100xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %0 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// CHECK-LABEL: func.func @fat_raw_buffer_cast
// CHECK-SAME: (%[[ARG:.*]]: memref<10x20xf32>)
func.func @fat_raw_buffer_cast(%arg0: memref<10x20xf32>) -> memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>> {
  // CHECK: %[[POISON0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[C0_I64:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[POISON1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : memref<10x20xf32> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[INSERT0:.*]] = llvm.insertvalue %[[EXTRACT]], %[[POISON1]][0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT1:.*]] = llvm.insertvalue %[[EXTRACT]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[INSERT2:.*]] = llvm.insertvalue %[[C0_I64]], %[[INSERT1]][2] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %[[INSERT2]] : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
  // CHECK: %[[FAT:.*]] = amdgpu.fat_raw_buffer_cast %[[CAST1]] : memref<f32> to memref<f32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[FAT]] : memref<f32, #amdgpu.address_space<fat_raw_buffer>> to !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[EXTRACT1:.*]] = llvm.extractvalue %[[CAST2]][1] : !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[INSERT3:.*]] = llvm.insertvalue %[[EXTRACT1]], %[[POISON0]][0] : !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[INSERT4:.*]] = llvm.insertvalue %[[EXTRACT1]], %[[INSERT3]][1] : !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[INSERT5:.*]] = llvm.insertvalue %[[C0_I64]], %[[INSERT4]][2] : !llvm.struct<(ptr<7>, ptr<7>, i64)>
  // CHECK: %[[CAST3:.*]] = builtin.unrealized_conversion_cast %[[INSERT5]] : !llvm.struct<(ptr<7>, ptr<7>, i64)> to memref<f32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast %[[CAST3]] to offset: [0], sizes: [10, 20], strides: [20, 1] : memref<f32, #amdgpu.address_space<fat_raw_buffer>> to memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: return %[[REINTERPRET]]
  %0 = amdgpu.fat_raw_buffer_cast %arg0 : memref<10x20xf32> to memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %0 : memref<10x20xf32, #amdgpu.address_space<fat_raw_buffer>>
}

// CHECK-LABEL: func.func @gather_to_lds
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x256xf32>, %[[ARG1:.*]]: memref<64x64xf32, #gpu.address_space<workgroup>>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index)
func.func @gather_to_lds(%src: memref<128x256xf32>, %dst: memref<64x64xf32, #gpu.address_space<workgroup>>, %src_i: index, %src_j: index, %dst_i: index, %dst_j: index) {
  // CHECK-DAG: %[[POISON_DST:.*]] = llvm.mlir.poison : !llvm.struct<(ptr<3>, ptr<3>, i64)>
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-DAG: %[[POISON_SRC:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // CHECK-DAG: %[[CAST_DST:.*]] = builtin.unrealized_conversion_cast %[[ARG1]]
  // CHECK-DAG: %[[EXTRACT_DST:.*]] = llvm.extractvalue %[[CAST_DST]][1]
  // CHECK-DAG: %[[CAST_SRC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK-DAG: %[[EXTRACT_SRC:.*]] = llvm.extractvalue %[[CAST_SRC]][1]
  // CHECK: %[[SRC_OFF:.*]] = affine.apply #{{.*}}()[%[[ARG2]], %[[ARG3]]]
  // CHECK: %[[SRC_OFF_CAST:.*]] = arith.index_cast %[[SRC_OFF]]
  // CHECK: %[[SRC_GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT_SRC]][%[[SRC_OFF_CAST]]]
  // CHECK: %[[DST_OFF:.*]] = affine.apply #{{.*}}()[%[[ARG4]], %[[ARG5]]]
  // CHECK: %[[DST_OFF_CAST:.*]] = arith.index_cast %[[DST_OFF]]
  // CHECK: %[[DST_GEP:.*]] = llvm.getelementptr nusw %[[EXTRACT_DST]][%[[DST_OFF_CAST]]]
  // CHECK: %[[SRC_INS0:.*]] = llvm.insertvalue %[[SRC_GEP]], %[[POISON_SRC]][0]
  // CHECK: %[[SRC_INS1:.*]] = llvm.insertvalue %[[SRC_GEP]], %[[SRC_INS0]][1]
  // CHECK: %[[SRC_INS2:.*]] = llvm.insertvalue %[[C0]], %[[SRC_INS1]][2]
  // CHECK: %[[SRC_0D:.*]] = builtin.unrealized_conversion_cast %[[SRC_INS2]] : !llvm.struct<(ptr, ptr, i64)> to memref<f32>
  // CHECK: %[[DST_INS0:.*]] = llvm.insertvalue %[[DST_GEP]], %[[POISON_DST]][0]
  // CHECK: %[[DST_INS1:.*]] = llvm.insertvalue %[[DST_GEP]], %[[DST_INS0]][1]
  // CHECK: %[[DST_INS2:.*]] = llvm.insertvalue %[[C0]], %[[DST_INS1]][2]
  // CHECK: %[[DST_0D:.*]] = builtin.unrealized_conversion_cast %[[DST_INS2]] : !llvm.struct<(ptr<3>, ptr<3>, i64)> to memref<f32, #gpu.address_space<workgroup>>
  // CHECK: amdgpu.gather_to_lds %[[SRC_0D]][], %[[DST_0D]][] : vector<4xf32>, memref<f32>, memref<f32, #gpu.address_space<workgroup>>
  amdgpu.gather_to_lds %src[%src_i, %src_j], %dst[%dst_i, %dst_j] : vector<4xf32>, memref<128x256xf32>, memref<64x64xf32, #gpu.address_space<workgroup>>
  return
}

func.func private @get_memref() -> memref<10xf32>

// CHECK-LABEL: func.func @scf_for_loop_carried
// CHECK-SAME: (%[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @scf_for_loop_carried(%arg0: memref<10xf32>, %lb: index, %ub: index, %step: index) -> memref<10xf32> {
  // Memref is hoisted out of loop since it doesn't change
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // Loop no longer has iter_args since memref doesn't change
  // CHECK: scf.for %[[IV:.*]] = %[[ARG1]] to %[[ARG2]] step %[[ARG3]] iter_args(%[[NEW_ITER:.*]] = %[[EXTRACT]]) -> (!llvm.ptr) {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> (memref<10xf32>) {
    // CHECK: %[[IDX0:.*]] = affine.apply #[[MAP:.*]]()[%[[IV]]]
    // CHECK: %[[CAST1:.*]] = arith.index_cast %[[IDX0]] : index to i64
    // CHECK: %[[GEP0:.*]] = llvm.getelementptr nusw %[[NEW_ITER]][%[[CAST1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    // CHECK: %[[LOAD:.*]] = llvm.load %[[GEP0]] : !llvm.ptr -> f32
    %val = memref.load %iter[%iv] : memref<10xf32>
    // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[LOAD]]
    %new_val = arith.addf %val, %val : f32
    // CHECK: %[[IDX1:.*]] = affine.apply #[[MAP]]()[%[[IV]]]
    // CHECK: %[[CAST2:.*]] = arith.index_cast %[[IDX1]] : index to i64
    // CHECK: %[[GEP1:.*]] = llvm.getelementptr nusw %[[NEW_ITER]][%[[CAST2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    // CHECK: llvm.store %[[ADD]], %[[GEP1]] : f32, !llvm.ptr
    memref.store %new_val, %iter[%iv] : memref<10xf32>
    %new_iter = func.call @get_memref() : () -> memref<10xf32>
    scf.yield %new_iter : memref<10xf32>
  }

  return %result : memref<10xf32>
}
