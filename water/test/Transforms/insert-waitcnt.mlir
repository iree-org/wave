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

// CHECK-LABEL: func.func @lds_barriers
//  CHECK-SAME: (%[[ARG0:.*]]: memref<1024xf32>, %[[ARG1:.*]]: memref<1024xf32>, %{{.*}}: index)
func.func @lds_barriers(%memrefA: memref<1024xf32>, %memrefB: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: %[[LOAD_A:.*]] = vector.load %[[ARG0]]
  // CHECK: %[[LOAD_B:.*]] = vector.load %[[ARG1]]
  %loadA = vector.load %memrefA[%offset] : memref<1024xf32>, vector<4xf32>
  %loadB = vector.load %memrefB[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(1)
  // CHECK-NEXT: amdgpu.lds_barrier
  // CHECK-NEXT: %[[ADD_A:.*]] = arith.addf %[[LOAD_A]], %[[LOAD_A]]
  amdgpu.lds_barrier
  %addA = arith.addf %loadA, %loadA : vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: amdgpu.lds_barrier
  // CHECK-NEXT: %[[ADD_B:.*]] = arith.addf %[[LOAD_B]], %[[ADD_A]]
  amdgpu.lds_barrier
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

// CHECK-LABEL: func.func @raw_dependency_memref
//  CHECK-SAME: (%[[MEM:.*]]: memref<1024xf32>, %[[DATA:.*]]: f32, %{{.*}}: index)
func.func @raw_dependency_memref(%memref: memref<1024xf32>, %data: f32, %offset: index) -> f32 {
  // Store to memory
  // CHECK: memref.store %[[DATA]], %[[MEM]]
  memref.store %data, %memref[%offset] : memref<1024xf32>

  // Load from same memory - RAW dependency, must wait for store
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[LOAD:.*]] = memref.load %[[MEM]]
  %result = memref.load %memref[%offset] : memref<1024xf32>

  // CHECK: return %[[LOAD]]
  return %result : f32
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
//  CHECK-SAME: (%[[GLOBAL:.*]]: memref<1024xf32>, %[[LDS:.*]]: memref<1024xf32, #gpu.address_space<workgroup>>, %{{.*}}: vector<4xf32>, %{{.*}}: index)
func.func @mixed_global_and_workgroup(%global: memref<1024xf32>, %lds: memref<1024xf32, #gpu.address_space<workgroup>>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
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

// CHECK-LABEL: func.func @existing_waitcnt
func.func @existing_waitcnt(%memref: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Store to memory
  // CHECK: vector.store
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Existing wait operation - should clear pending operations
  // CHECK: amdgpu.memory_counter_wait load(0)
  amdgpu.memory_counter_wait load(0)

  // Another store after the wait
  // CHECK: vector.store
  vector.store %data, %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // Load requires wait for the second store only (first was already waited on)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load
  %result = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @existing_waitcnt_more_strict
func.func @existing_waitcnt_more_strict(%data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  %memref1 = memref.alloc() : memref<1024xf32>
  %memref2 = memref.alloc() : memref<1024xf32>

  // Store to memory
  // CHECK: vector.store
  // CHECK: vector.store
  vector.store %data, %memref1[%offset] : memref<1024xf32>, vector<4xf32>
  vector.store %data, %memref2[%offset] : memref<1024xf32>, vector<4xf32>

  // Existing wait operation - should clear pending operations
  // Normally, the distance will be 1, but explicit amdgpu.memory_counter_wait
  // overrides it.
  // CHECK-NOT: amdgpu.memory_counter_wait load(1)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NOT: amdgpu.memory_counter_wait load(1)
  amdgpu.memory_counter_wait load(0)

  // CHECK: %[[LOAD:.*]] = vector.load
  %result = vector.load %memref1[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}


// CHECK-LABEL: func.func @control_flow_merge
func.func @control_flow_merge(%cond: i1, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  %memref1 = memref.alloc() : memref<1024xf32>
  %memref2 = memref.alloc() : memref<1024xf32>

  // Common operation before branching
  // CHECK: vector.store
  vector.store %data, %memref1[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: cf.cond_br
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // Extra operation in this path
  // CHECK: vector.store
  vector.store %data, %memref2[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: cf.br
  cf.br ^bb3

^bb2:
  // No extra operations, just branch to merge point
  // CHECK: cf.br
  cf.br ^bb3

^bb3:
  // bb1 branch has distance 1 but bb2 has distance 0, so we need to conservatively
  // take 0
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load
  %result = vector.load %memref1[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @control_flow_merge_same_lists
func.func @control_flow_merge_same_lists(%cond: i1, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  %memref1 = memref.alloc() : memref<1024xf32>
  %memref2 = memref.alloc() : memref<1024xf32>

  // Common operation before branching
  // CHECK: vector.store
  vector.store %data, %memref1[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: cf.cond_br
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: vector.store
  vector.store %data, %memref2[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: cf.br
  cf.br ^bb3

^bb2:
  vector.store %data, %memref2[%offset] : memref<1024xf32>, vector<4xf32>
  // CHECK: cf.br
  cf.br ^bb3

^bb3:
  // both branches has the same distance 1
  // CHECK: amdgpu.memory_counter_wait load(1)
  // CHECK-NEXT: %[[LOAD:.*]] = vector.load
  %result = vector.load %memref1[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[LOAD]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @loop_carried_dependency
func.func @loop_carried_dependency(%lb: index, %ub: index, %step: index, %memref: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: scf.for
  %result = scf.for %i = %lb to %ub step %step iter_args(%arg = %data) -> (vector<4xf32>) {
    // Store in each iteration
    // CHECK-NOT: amdgpu.memory_counter_wait
    // CHECK: vector.store
    vector.store %arg, %memref[%offset] : memref<1024xf32>, vector<4xf32>

    // Load in the same iteration - RAW dependency with store from this iteration
    // In steady state, the backedge brings pending operations from previous iteration
    // CHECK: amdgpu.memory_counter_wait load(0)
    // CHECK-NEXT: %[[LOADED:.*]] = vector.load
    %loaded = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

    // Yield uses the load result, which is async, so need to wait for it
    // CHECK: amdgpu.memory_counter_wait load(0)
    // CHECK-NEXT: scf.yield %[[LOADED]]
    scf.yield %loaded : vector<4xf32>
  }

  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @loop_load_before_store
func.func @loop_load_before_store(%lb: index, %ub: index, %step: index, %memref: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // CHECK: scf.for
  %result = scf.for %i = %lb to %ub step %step iter_args(%arg = %data) -> (vector<4xf32>) {
    // Load first - in steady state, has RAW dependency with store from previous iteration
    // CHECK: amdgpu.memory_counter_wait load(0)
    // CHECK-NEXT: %[[LOADED:.*]] = vector.load
    %loaded = vector.load %memref[%offset] : memref<1024xf32>, vector<4xf32>

    // Store after load - WAR dependency with load in same iteration
    // The wait for the load clears it from pending, so this wait is for the load
    // CHECK: amdgpu.memory_counter_wait load(0)
    // CHECK-NEXT: vector.store
    vector.store %arg, %memref[%offset] : memref<1024xf32>, vector<4xf32>

    // Yield uses load result - load was already waited on by the store, no additional wait needed
    // CHECK-NOT: amdgpu.memory_counter_wait
    // CHECK: scf.yield %[[LOADED]]
    scf.yield %loaded : vector<4xf32>
  }

  // CHECK: return
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @memref_copy_raw_source
func.func @memref_copy_raw_source(%src: memref<1024xf32>, %dst: memref<1024xf32>, %data: vector<4xf32>, %offset: index) {
  // Store to source
  // CHECK: vector.store
  vector.store %data, %src[%offset] : memref<1024xf32>, vector<4xf32>

  // Copy from source - RAW dependency (reads from source that was just written)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: memref.copy
  memref.copy %src, %dst : memref<1024xf32> to memref<1024xf32>

  // CHECK: return
  return
}

// CHECK-LABEL: func.func @memref_copy_waw_target
func.func @memref_copy_waw_target(%src: memref<1024xf32>, %dst: memref<1024xf32>, %data: vector<4xf32>, %offset: index) {
  // Store to destination
  // CHECK: vector.store
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // Copy to destination - WAW dependency (writes to target that was just written)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: memref.copy
  memref.copy %src, %dst : memref<1024xf32> to memref<1024xf32>

  // CHECK: return
  return
}

// CHECK-LABEL: func.func @memref_copy_war_target
func.func @memref_copy_war_target(%src: memref<1024xf32>, %dst: memref<1024xf32>, %offset: index) -> vector<4xf32> {
  // Load from destination
  // CHECK: %[[RESULT:.*]] = vector.load
  %result = vector.load %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // Copy to destination - WAR dependency (writes to target that was just read)
  // The copy's wait also synchronizes the load, so return doesn't need another wait
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: memref.copy
  memref.copy %src, %dst : memref<1024xf32> to memref<1024xf32>

  // CHECK: return %[[RESULT]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @memref_copy_both_dependencies
func.func @memref_copy_both_dependencies(%src: memref<1024xf32>, %dst: memref<1024xf32>, %data: vector<4xf32>, %offset: index) -> vector<4xf32> {
  // Store to source
  // CHECK: vector.store
  vector.store %data, %src[%offset] : memref<1024xf32>, vector<4xf32>

  // Store to destination
  // CHECK: vector.store
  vector.store %data, %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // Copy needs to wait for both stores:
  // - RAW on source (copy reads from source)
  // - WAW on target (copy writes to destination)
  // Both stores alias with their respective memrefs, so we need wait(0)
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: memref.copy
  memref.copy %src, %dst : memref<1024xf32> to memref<1024xf32>

  // Load from destination after copy - RAW dependency with copy
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: %[[RESULT:.*]] = vector.load
  %result = vector.load %dst[%offset] : memref<1024xf32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: return %[[RESULT]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @gather_to_lds
func.func @gather_to_lds(%global: memref<1024xf32>, %lds: memref<1024xf32, #gpu.address_space<workgroup>>, %data: vector<4xf32>, %src_offset: index, %dst_offset: index) -> vector<4xf32> {
  // Store to global memory
  // CHECK: vector.store
  vector.store %data, %global[%src_offset] : memref<1024xf32>, vector<4xf32>

  // Gather from global to LDS - has both RAW (reads from global) and acts as store to LDS
  // Should wait for global store using load counter
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: amdgpu.gather_to_lds
  amdgpu.gather_to_lds %global[%src_offset], %lds[%dst_offset] : f32, memref<1024xf32>, memref<1024xf32, #gpu.address_space<workgroup>>

  // Load from LDS - RAW dependency with gather writing to LDS
  // Should wait for LDS operation using ds counter
  // CHECK: amdgpu.memory_counter_wait ds(0)
  // CHECK-NEXT: %[[RESULT:.*]] = vector.load
  %result = vector.load %lds[%dst_offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait ds(0)
  // CHECK-NEXT: return %[[RESULT]]
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @double_buffering
func.func @double_buffering(%src: memref<1024xf32>, %lb: index, %ub: index, %step: index, %offset: index) {
  %buff0 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %buff1 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

  %out = memref.alloc() : memref<1024xf32>

  // CHECK-NOT: amdgpu.memory_counter_wait
  //     CHECK: memref.copy
  memref.copy %src, %buff0 : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%current = %buff0, %next = %buff1) -> (memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>) {
    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: memref.copy
    memref.copy %src, %next : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

    // Skip the second buffer copy
    //     CHECK: amdgpu.memory_counter_wait ds(1)
    //     CHECK: vector.load
    %data = vector.load %current[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // Cannot skip unfortunately
    //     CHECK: amdgpu.memory_counter_wait load(0) ds(0)
    //     CHECK: vector.store
    vector.store %data, %out[%offset] : memref<1024xf32>, vector<4xf32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: scf.yield
    scf.yield %next, %current : memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>
  }

  // CHECK: return
  return
}

// CHECK-LABEL: func.func @triple_buffering
func.func @triple_buffering(%src: memref<1024xf32>, %lb: index, %ub: index, %step: index, %offset: index) {
  %buff0 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %buff1 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %buff2 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

  %out = memref.alloc() : memref<1024xf32>

  // CHECK-NOT: amdgpu.memory_counter_wait
  //     CHECK: memref.copy
  memref.copy %src, %buff0 : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

  // CHECK-NOT: amdgpu.memory_counter_wait
  //     CHECK: memref.copy
  memref.copy %src, %buff1 : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%current = %buff0, %next = %buff1, %next_next = %buff2) -> (memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>) {
    // Skip the second buffer copy
    //     CHECK: amdgpu.memory_counter_wait ds(1)
    //     CHECK: vector.load
    %data = vector.load %current[%offset] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: memref.copy
    memref.copy %src, %next_next : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

    // Skip the prev copy
    //     CHECK: amdgpu.memory_counter_wait load(0) ds(1)
    //     CHECK: vector.store
    vector.store %data, %out[%offset] : memref<1024xf32>, vector<4xf32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: scf.yield
    scf.yield %next, %next_next, %current : memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>
  }

  // CHECK: return
  return
}


// CHECK-LABEL: func.func @triple_buffering_reg_space
func.func @triple_buffering_reg_space(%src: memref<1024xf32>, %lb: index, %ub: index, %step: index, %offset: index) {
  %c0 = arith.constant 0 : index
  %buff0 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %buff1 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %buff2 = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
  %reg = memref.alloca() : memref<4xf32, 128 : i32>

  %out = memref.alloc() : memref<1024xf32>

  // CHECK-NOT: amdgpu.memory_counter_wait
  //     CHECK: memref.copy
  memref.copy %src, %buff0 : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

  // CHECK-NOT: amdgpu.memory_counter_wait
  //     CHECK: memref.copy
  memref.copy %src, %buff1 : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%current = %buff0, %next = %buff1, %next_next = %buff2) -> (memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>) {
    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: memref.copy
    memref.copy %src, %next_next : memref<1024xf32> to memref<1024xf32, #gpu.address_space<workgroup>>

    // Skip the the prev copy
    //     CHECK: amdgpu.memory_counter_wait ds(1)
    //     CHECK: vector.load
    %data = vector.load %reg[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: vector.store
    vector.store %data, %out[%offset] : memref<1024xf32>, vector<4xf32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: memref.subview
    %subview = memref.subview %current[%offset] [4] [1] : memref<1024xf32, #gpu.address_space<workgroup>> to memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>

    // This copy only depends on buffer 2 iterations ago
    //     CHECK: amdgpu.memory_counter_wait ds(2)
    //     CHECK: memref.copy
    memref.copy %subview, %reg : memref<4xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>> to memref<4xf32, 128 : i32>

    // CHECK-NOT: amdgpu.memory_counter_wait
    //     CHECK: scf.yield
    scf.yield %next, %next_next, %current : memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>, memref<1024xf32, #gpu.address_space<workgroup>>
  }

  // CHECK: return
  return
}

// CHECK-LABEL: func.func @load_store_repeated
func.func @load_store_repeated(%src0: memref<4xf32>, %src1: memref<4xf32>, %offset: index) {
  %c0 = arith.constant 0 : index
  %buff0 = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
  %buff1 = memref.alloc() : memref<4xf32, #gpu.address_space<workgroup>>
  %reg0 = memref.alloca() : memref<4xf32, 128 : i32>
  %reg1 = memref.alloca() : memref<4xf32, 128 : i32>
  %reg2 = memref.alloca() : memref<4xf32, 128 : i32>
  %reg3 = memref.alloca() : memref<4xf32, 128 : i32>

  // CHECK-COUNT-4: memref.copy
  memref.copy %src0, %reg0 : memref<4xf32> to memref<4xf32, 128 : i32>
  memref.copy %src1, %reg1 : memref<4xf32> to memref<4xf32, 128 : i32>

  memref.copy %buff0, %reg2 : memref<4xf32, #gpu.address_space<workgroup>> to memref<4xf32, 128 : i32>
  memref.copy %buff1, %reg3 : memref<4xf32, #gpu.address_space<workgroup>> to memref<4xf32, 128 : i32>

  // CHECK: amdgpu.memory_counter_wait load(1)
  // CHECK-NEXT: vector.load
  %data0 = vector.load %reg0[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK: amdgpu.memory_counter_wait load(0)
  // CHECK-NEXT: vector.load
  %data1 = vector.load %reg1[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>

  // CHECK: amdgpu.memory_counter_wait ds(1)
  // CHECK-NEXT: vector.load
  %data2 = vector.load %reg2[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>
  // CHECK: amdgpu.memory_counter_wait ds(0)
  // CHECK-NEXT: vector.load
  %data3 = vector.load %reg3[%c0] : memref<4xf32, 128 : i32>, vector<4xf32>

  return
}
