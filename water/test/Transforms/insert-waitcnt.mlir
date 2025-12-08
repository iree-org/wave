// RUN: water-opt %s --water-insert-waitcnt | FileCheck %s

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

// CHECK-LABEL: func.func @raw_dependency
//  CHECK-SAME: (%[[MEM:.*]]: memref<1024xf32>, %[[DATA:.*]]: vector<4xf32>, %{{.*}}: index)
func.func @raw_dependency(%memref: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Store to memory
  // CHECK: vector.store %[[DATA]], %[[MEM]]
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Load from same memory - RAW dependency, must wait for store
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load %[[MEM]]
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @war_dependency
//  CHECK-SAME: (%[[MEM:.*]]: memref<1024xf32>, %[[DATA:.*]]: vector<4xf32>, %{{.*}}: index)
func.func @war_dependency(%memref: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Load from memory
  // CHECK: %[[LOAD:.*]] = vector.load %[[MEM]]
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to same memory - WAR dependency, must wait for load
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: vector.store %[[DATA]], %[[MEM]]
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK-NOT: amdgpu.memory_counter_wait
  // CHECK: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @waw_dependency
//  CHECK-SAME: (%[[MEM:.*]]: memref<1024xf32>, %[[DATA1:.*]]: vector<4xf32>, %[[DATA2:.*]]: vector<4xf32>, %{{.*}}: index)
func.func @waw_dependency(%memref: memref<1024xf32>, %data1: vector<4xf32>, %data2: vector<4xf32>, %offset: index) {
  // First store
  // CHECK: vector.store %[[DATA1]], %[[MEM]]
  vector.store %data1, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Second store to same memory - WAW dependency, must wait for first store
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: vector.store %[[DATA2]], %[[MEM]]
  vector.store %data2, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: return
  return
}

// CHECK-LABEL: func.func @raw_dependency_non_zero_waitcnt
func.func @raw_dependency_non_zero_waitcnt(%data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Allocate two distinct memrefs to guarantee no aliasing
  // CHECK: %[[MEM_A:.*]] = memref.alloc()
  %memrefA = memref.alloc() : memref<1024xf32>
  // CHECK: %[[MEM_B:.*]] = memref.alloc()
  %memrefB = memref.alloc() : memref<1024xf32>

  // Store to memory A
  // CHECK: vector.store %{{.*}}, %[[MEM_A]]
  vector.store %data, %memrefA[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to memory B (intervening operation, different memref)
  // CHECK: vector.store %{{.*}}, %[[MEM_B]]
  vector.store %data, %memrefB[%offset] : memref<1024xf32>, vector<4xf32>

  // Load from memory A - RAW dependency with store to A at distance 1
  // CHECK: amdgpu.memory_counter_wait load(1)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load %[[MEM_A]]
  %result = vector.load %memrefA[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @workgroup_memory_raw
func.func @workgroup_memory_raw(%data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Allocate workgroup (LDS) memory
  // CHECK: %[[LDS:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %lds = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

  // Store to LDS
  // CHECK: vector.store %{{.*}}, %[[LDS]]
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Load from LDS - RAW dependency, should use dsCnt not loadCnt
  // CHECK: amdgpu.memory_counter_wait ds(0)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load %[[LDS]]
  %result = vector.load %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait ds(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @mixed_global_and_workgroup
func.func @mixed_global_and_workgroup(%data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Allocate global and workgroup memory
  // CHECK: %[[GLOBAL:.*]] = memref.alloc() : memref<1024xf32>
  %global = memref.alloc() : memref<1024xf32>
  // CHECK: %[[LDS:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %lds = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

  // Store to global memory
  // CHECK: vector.store %{{.*}}, %[[GLOBAL]]
  vector.store %data, %global[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to LDS (different counter, no dependency)
  // CHECK-NOT: amdgpu.memory_counter_wait
  // CHECK: vector.store %{{.*}}, %[[LDS]]
  vector.store %data, %lds[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // Load from global - RAW dependency with global store at distance 0
  // (LDS store doesn't count because it's a different counter type)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load %[[GLOBAL]]
  %result = vector.load %global[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}
