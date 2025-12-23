// RUN: water-opt %s --water-alloc-to-alloca | FileCheck %s

// CHECK-LABEL: func.func @simple_alloc
func.func @simple_alloc() ->  memref<16xf32> {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<16xf32>
  %0 = memref.alloc() : memref<16xf32>
  // CHECK: return %[[ALLOCA]]
  return %0 : memref<16xf32>
}

// -----

// CHECK-LABEL: func.func @alloc_with_dynamic_sizes
func.func @alloc_with_dynamic_sizes(%size: index) {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[ALLOCA:.*]] = memref.alloca(%{{.*}}) : memref<?xf32>
  %0 = memref.alloc(%size) : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @alloc_with_alignment
func.func @alloc_with_alignment() {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() {alignment = 64 : i64} : memref<32xf32>
  %0 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
  return
}

// -----

// CHECK-LABEL: func.func @multiple_allocs
func.func @multiple_allocs() {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[ALLOCA1:.*]] = memref.alloca() : memref<8xi32>
  %0 = memref.alloc() : memref<8xi32>
  // CHECK: %[[ALLOCA2:.*]] = memref.alloca() : memref<16xf64>
  %1 = memref.alloc() : memref<16xf64>
  return
}
