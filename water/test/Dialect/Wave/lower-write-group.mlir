// RUN: water-opt %s --split-input-file --lower-wave-to-mlir | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_basic
  func.func @test_basic(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %waveValue1 = wave.register %cst : vector<4xf32>
    %waveValue2 = wave.register %cst : vector<4xf32>

    // Phase 1: Single LDS allocation for the shared group
    // CHECK: %[[SHARED_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

    // Phase 2: Register→LDS stores with correct index calculations
    // Verify affine maps for index calculations: original global index (WG0*1024+T0*4) and LDS base (WG0*1024)
    // CHECK: %[[BLOCK_ID:.*]] = gpu.block_id x
    // CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x
    // CHECK: %[[GLOBAL_IDX:.*]] = affine.apply #{{.*}}()[%[[BLOCK_ID]], %[[THREAD_ID]]]
    // CHECK: %[[LDS_BASE:.*]] = affine.apply #{{.*}}()[%{{.*}}]
    // CHECK: %[[LDS_STORE_IDX:.*]] = arith.subi %[[GLOBAL_IDX]], %[[LDS_BASE]] : index
    // CHECK: vector.store %{{.*}}, %[[SHARED_ALLOC]][%[[LDS_STORE_IDX]]] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    // Second operation with offset (+256)
    // CHECK: vector.store %{{.*}}, %[[SHARED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // Phase 3: Single shared barrier
    // CHECK: amdgpu.lds_barrier
    // Ensure only one barrier for the group
    // CHECK-NOT: amdgpu.lds_barrier

    // Phase 4: LDS→Register loads using lds_load_indices (T0*64)
    // CHECK: %[[LDS_LOAD_IDX:.*]] = affine.apply #{{.*}}()[%{{.*}}]
    // CHECK: %[[LOADED_VEC1:.*]] = vector.load %[[SHARED_ALLOC]][%[[LDS_LOAD_IDX]]] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // Phase 5: Register→Global stores using global_store_indices (WG0*1024+T0*64)
    // CHECK: %[[GLOBAL_STORE_IDX1:.*]] = affine.apply #{{.*}}()[%{{.*}}, %{{.*}}]
    // CHECK: vector.store %[[LOADED_VEC1]], %{{.*}}[%[[GLOBAL_STORE_IDX1]]] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

    // Second operation: LDS load and global store with offset (+256)
    // CHECK: %[[LOADED_VEC2:.*]] = vector.load %[[SHARED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: %[[GLOBAL_STORE_IDX2:.*]] = affine.apply #{{.*}}()[%{{.*}}, %{{.*}}]
    // CHECK: vector.store %[[LOADED_VEC2]], %{{.*}}[%[[GLOBAL_STORE_IDX2]]] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

    // Verify no wave.write operations remain after lowering
    // CHECK-NOT: wave.write

    wave.write %waveValue1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "shared_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    wave.write %waveValue2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4 + 256, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "shared_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 256)>
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_different_groups
  func.func @test_different_groups(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %waveValue1 = wave.register %cst : vector<4xf32>
    %waveValue2 = wave.register %cst : vector<4xf32>

    // First group (group_a): allocation, register→LDS, barrier, LDS→register, register→global
    // CHECK: %[[ALLOC_A:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: vector.store %{{.*}}, %[[ALLOC_A]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    // CHECK: amdgpu.lds_barrier
    // CHECK: %{{.*}} = vector.load %[[ALLOC_A]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

    // Second group (group_b): separate allocation, register→LDS, barrier, LDS→register, register→global
    // CHECK: %[[ALLOC_B:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: vector.store %{{.*}}, %[[ALLOC_B]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    // CHECK: amdgpu.lds_barrier
    // CHECK: %{{.*}} = vector.load %[[ALLOC_B]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write

    wave.write %waveValue1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "group_a",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    wave.write %waveValue2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "group_b",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_mixed_lds_and_regular_writes
  func.func @test_mixed_lds_and_regular_writes(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %waveValue1 = wave.register %cst : vector<4xf32>
    %waveValue2 = wave.register %cst : vector<4xf32>

    // First operation uses LDS promotion: allocation, register→LDS, barrier, LDS→register, register→global
    // CHECK: %[[MIXED_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: vector.store %{{.*}}, %[[MIXED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    // CHECK: amdgpu.lds_barrier
    // CHECK: %{{.*}} = vector.load %[[MIXED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

    // Second operation uses regular write - direct register→global store (no LDS)
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<4xf32>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write

    wave.write %waveValue1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "lds_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    wave.write %waveValue2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = false,
        group_id = "regular_group"
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_nested_lds_promotion_group
  func.func @test_nested_lds_promotion_group(%cond: i1, %mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %waveValue1 = wave.register %cst : vector<4xf32>
    %waveValue2 = wave.register %cst : vector<4xf32>

    scf.if %cond {
      // Nested LDS promotion group should generate: allocation, register→LDS stores, barrier, LDS→register loads, register→global stores
      // CHECK: %[[NESTED_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
      // CHECK: vector.store %{{.*}}, %[[NESTED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      // CHECK: vector.store %{{.*}}, %[[NESTED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      // CHECK: amdgpu.lds_barrier
      // CHECK: %{{.*}} = vector.load %[[NESTED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
      // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>
      // CHECK: %{{.*}} = vector.load %[[NESTED_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
      // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

      wave.write %waveValue1, %mem1 index [{
          M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
        }] {
        memory_access_pattern = #wave.memory_access_pattern<
          use_lds_promotion = true,
          group_id = "nested_group",
          lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
          lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
          lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
          lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
          global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
          >
      } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

      wave.write %waveValue2, %mem2 index [{
          M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4 + 256, 4, 1)
        }] {
        memory_access_pattern = #wave.memory_access_pattern<
          use_lds_promotion = true,
          group_id = "nested_group",
          lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
          lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
          lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
          lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
          global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 256)>
          >
      } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>
    }

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_lds_promotion_group_same_scope
  func.func @test_lds_promotion_group_same_scope(%cond: i1, %mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %waveValue1 = wave.register %cst : vector<4xf32>
    %waveValue2 = wave.register %cst : vector<4xf32>

    scf.if %cond {
      // Both operations are in the same nested scope (scf.if block) - this should work correctly
      // Single LDS allocation for the shared group
      // CHECK: %[[SAME_SCOPE_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

      // Register→LDS stores
      // CHECK: vector.store %{{.*}}, %[[SAME_SCOPE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
      // CHECK: vector.store %{{.*}}, %[[SAME_SCOPE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>

      // Single barrier for the group
      // CHECK: amdgpu.lds_barrier

      // LDS→Register loads and Register→Global stores
      // CHECK: %{{.*}} = vector.load %[[SAME_SCOPE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
      // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>
      // CHECK: %{{.*}} = vector.load %[[SAME_SCOPE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<64xf32>
      // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<64xf32>

      wave.write %waveValue1, %mem1 index [{
          M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
        }] {
        memory_access_pattern = #wave.memory_access_pattern<
          use_lds_promotion = true,
          group_id = "same_scope_group",
          lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
          lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
          lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
          lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
          global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
          >
      } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

      wave.write %waveValue2, %mem2 index [{
          M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4 + 256, 4, 1)
        }] {
        memory_access_pattern = #wave.memory_access_pattern<
          use_lds_promotion = true,
          group_id = "same_scope_group",
          lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
          lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
          lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
          lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
          global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 256)>
          >
      } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>
    }

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_multidim_lds_promotion
  func.func @test_multidim_lds_promotion(%mem: !wave.tensor<[@M, @N] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 64, BLOCK_N = 64, M = 1024, N = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %value = wave.register %cst : vector<8xf32>

    // Tests 2D LDS allocation and lowering
    // Phase 1: 2D LDS allocation (BLOCK_M × BLOCK_N = 64×64 = 4096 elements)
    // CHECK: %[[ALLOC_2D:.*]] = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>

    // Phase 2: Register→LDS store with 2D indexing
    // Original 2D access: (WG0*BLOCK_M + T0*8, WG1*BLOCK_N + T1*1)
    // LDS store: subtract 2D base (WG0*BLOCK_M, WG1*BLOCK_N) to get local LDS coordinates
    // CHECK-DAG: vector.transfer_write %{{.*}}, %[[ALLOC_2D]][%{{.*}}, %{{.*}}] {{.*}} : vector<8xf32>, memref<64x64xf32, #gpu.address_space<workgroup>>

    // Phase 3: Barrier synchronization
    // CHECK: amdgpu.lds_barrier

    // Phase 4: LDS→Register load with vectorized 2D pattern (T0*8, T1*4)
    // CHECK-DAG: %[[REG:.*]] = vector.transfer_read %[[ALLOC_2D]][%{{.*}}, %{{.*}}], %{{.*}} {{.*}} : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<8xf32>

    // Phase 5: Register→Global store with 2D coordinates (WG0*BLOCK_M + T0*8, WG1*BLOCK_N + T1*4)
    // CHECK-DAG: vector.transfer_write %[[REG]], %{{.*}}[%{{.*}}, %{{.*}}] {{.*}} : vector<8xf32>, memref<1024x1024xf32, #gpu.address_space<global>>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write

    wave.write %value, %mem index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8, 8, 1),
        N : [#wave.symbol<"BLOCK_N">, #wave.index_symbol<WG1>, #wave.index_symbol<T1>] -> (BLOCK_N * WG1 + T1, 1, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "multidim_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (WG0 * BLOCK_M, WG1 * BLOCK_N)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (BLOCK_M, BLOCK_N)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>, #wave.index_symbol<T1>] -> (T0 * 8, T1 * 4)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (8, 4)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_M">, #wave.symbol<"BLOCK_N">] -> (WG0 * BLOCK_M + T0 * 8, WG1 * BLOCK_N + T1 * 4)>
      >
    } : vector<8xf32>, !wave.tensor<[@M, @N] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_variable_vector_sizes
  func.func @test_variable_vector_sizes(%mem: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, VEC_SIZE = 32, M = 4096}>
  } {
    %cst = arith.constant 0.0 : f32
    %value = wave.register %cst : vector<32xf32>

    // Variable vector size test: LDS allocation with symbolic VEC_SIZE (32)
    // CHECK: %[[VAR_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>
    // CHECK: vector.store %{{.*}}, %[[VAR_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<32xf32>
    // CHECK: amdgpu.lds_barrier
    // CHECK: %{{.*}} = vector.load %[[VAR_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<32xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<4096xf32, #gpu.address_space<global>>, vector<32xf32>
    // CHECK-NOT: wave.write

    wave.write %value, %mem index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"VEC_SIZE">] -> (BLOCK_M * WG0 + T0 * VEC_SIZE, VEC_SIZE, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "variable_vec_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>, #wave.symbol<"VEC_SIZE">] -> (T0 * VEC_SIZE)>,
        lds_load_vector_sizes = #wave.expr_list<[#wave.symbol<"VEC_SIZE">] -> (VEC_SIZE)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">, #wave.symbol<"VEC_SIZE">] -> (WG0 * BLOCK_M + T0 * VEC_SIZE)>
      >
    } : vector<32xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_complex_expressions
  func.func @test_complex_expressions(%mem: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, OFFSET = 128, STRIDE = 16, M = 8192}>
  } {
    %cst = arith.constant 0.0 : f32
    %value = wave.register %cst : vector<16xf32>

    // Complex expression test: LDS allocation with OFFSET and STRIDE symbolics
    // CHECK: %[[COMPLEX_ALLOC:.*]] = memref.alloc() : memref<1152xf32, #gpu.address_space<workgroup>>
    // CHECK: vector.store %{{.*}}, %[[COMPLEX_ALLOC]][%{{.*}}] : memref<1152xf32, #gpu.address_space<workgroup>>, vector<16xf32>
    // CHECK: amdgpu.lds_barrier
    // CHECK: %{{.*}} = vector.load %[[COMPLEX_ALLOC]][%{{.*}}] : memref<1152xf32, #gpu.address_space<workgroup>>, vector<16xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8192xf32, #gpu.address_space<global>>, vector<16xf32>
    // CHECK-NOT: wave.write

    wave.write %value, %mem index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"OFFSET">, #wave.symbol<"STRIDE">] -> (BLOCK_M * WG0 + OFFSET + T0 * STRIDE + 8, STRIDE, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "complex_expr_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.symbol<"OFFSET">] -> (WG0 * BLOCK_M + OFFSET)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">, #wave.symbol<"OFFSET">] -> (BLOCK_M + OFFSET)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>, #wave.symbol<"STRIDE">] -> (T0 * STRIDE + 8)>,
        lds_load_vector_sizes = #wave.expr_list<[#wave.symbol<"STRIDE">] -> (STRIDE)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">, #wave.symbol<"OFFSET">, #wave.symbol<"STRIDE">] -> (WG0 * BLOCK_M + OFFSET + T0 * STRIDE + 8)>
      >
    } : vector<16xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_no_lds_promo
  func.func @test_no_lds_promo(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 512, M = 2048}>
  } {
    %cst = arith.constant 0.0 : f32
    %value1 = wave.register %cst : vector<4xf32>
    %value2 = wave.register %cst : vector<4xf32>

    // First operation uses regular write (no LDS promotion)
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<4xf32>

    // Second operation has no memory access pattern - should use default vector write
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<4xf32>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write
    wave.write %value1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = false,
        group_id = "regular_group"
      >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    // Second operation has no memory access pattern at all - should use default vector write
    // CHECK-NOT: memory_access_pattern
    wave.write %value2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
  }

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // Verify the correct affine maps are generated with concrete stride values
  // CHECK: #[[MAP_GLOBAL_256:.*]] = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 8)>
  // CHECK: #[[MAP_LDS_BASE_256:.*]] = affine_map<()[s0] -> (s0 * 256)>
  // CHECK: #[[MAP_LDS_LOAD_16:.*]] = affine_map<()[s0] -> (s0 * 16)>
  // CHECK: #[[MAP_GLOBAL_STORE_16:.*]] = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 16)>

  func.func @verify_index_math(%mem: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 256, M = 2048}>
  } {
    %cst = arith.constant 0.0 : f32
    %value = wave.register %cst : vector<8xf32>

    // LDS block size should be BLOCK_M = 256
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>

    // Original index: BLOCK_M * WG0 + T0 * 8 = 256 * WG0 + T0 * 8
    // CHECK: %[[GLOBAL_IDX:.*]] = affine.apply #[[MAP_GLOBAL_256]]()[%{{.*}}, %{{.*}}]

    // LDS base: WG0 * BLOCK_M = WG0 * 256
    // CHECK: %[[LDS_BASE:.*]] = affine.apply #[[MAP_LDS_BASE_256]]()[%{{.*}}]

    // LDS store index: (256*WG0 + T0*8) - (256*WG0) = T0*8 (local offset)
    // CHECK: %[[LDS_STORE_IDX:.*]] = arith.subi %[[GLOBAL_IDX]], %[[LDS_BASE]] : index
    // CHECK: vector.store %{{.*}}, %[[ALLOC]][%[[LDS_STORE_IDX]]] : memref<256xf32, #gpu.address_space<workgroup>>, vector<8xf32>

    // CHECK: amdgpu.lds_barrier

    // LDS load index: T0 * 16 (different stride for vectorized access)
    // CHECK: %[[LDS_LOAD_IDX:.*]] = affine.apply #[[MAP_LDS_LOAD_16]]()[%{{.*}}]
    // CHECK: %[[LOADED:.*]] = vector.load %[[ALLOC]][%[[LDS_LOAD_IDX]]] : memref<256xf32, #gpu.address_space<workgroup>>, vector<16xf32>

    // Verify constants are properly materialized during materialization
    // CHECK: %{{.*}} = arith.constant 256 : index

    // Global store index: WG0 * 256 + T0 * 16
    // CHECK: %[[GLOBAL_STORE_IDX:.*]] = affine.apply #[[MAP_GLOBAL_STORE_16]]()[%{{.*}}, %{{.*}}]
    // CHECK: vector.store %[[LOADED]], %{{.*}}[%[[GLOBAL_STORE_IDX]]] : memref<2048xf32, #gpu.address_space<global>>, vector<16xf32>

    wave.write %value, %mem index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "math_verification_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 16)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (16)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 16)>
      >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    return
  }
  }

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_mixed_vector_sizes_same_group
  func.func @test_mixed_vector_sizes_same_group(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 1024, M = 1024}>
  } {
    %cst = arith.constant 0.0 : f32
    %value1 = wave.register %cst : vector<4xf32>   // 4-element vector
    %value2 = wave.register %cst : vector<8xf32>   // 8-element vector (different size)

    // Single LDS allocation for the mixed-size group
    // CHECK: %[[MIXED_SIZE_ALLOC:.*]] = memref.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

    // First operation with 4-element vector
    // CHECK: vector.store %{{.*}}, %[[MIXED_SIZE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    wave.write %value1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 4, 4, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "mixed_vector_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 32)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (32)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 32)>
        >
    } : vector<4xf32>, !wave.tensor<[@M] of f32, <global>>

    // Second operation with 8-element vector (different input vector size, but same LDS vectorization)
    // CHECK: vector.store %{{.*}}, %[[MIXED_SIZE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    wave.write %value2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 256, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "mixed_vector_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 32)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (32)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 32 + 256)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Single barrier for the entire mixed-size group
    // CHECK: amdgpu.lds_barrier

    // LDS→register loads and register→global stores (same vectorization for both operations)
    // CHECK: %{{.*}} = vector.load %[[MIXED_SIZE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<32xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<32xf32>
    // CHECK: %{{.*}} = vector.load %[[MIXED_SIZE_ALLOC]][%{{.*}}] : memref<1024xf32, #gpu.address_space<workgroup>>, vector<32xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xf32, #gpu.address_space<global>>, vector<32xf32>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write

    return
  }
  }

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  // CHECK-LABEL: @test_large_group
  func.func @test_large_group(%mem1: !wave.tensor<[@M] of f32, <global>>, %mem2: !wave.tensor<[@M] of f32, <global>>, %mem3: !wave.tensor<[@M] of f32, <global>>, %mem4: !wave.tensor<[@M] of f32, <global>>, %mem5: !wave.tensor<[@M] of f32, <global>>, %mem6: !wave.tensor<[@M] of f32, <global>>) attributes {
    wave.hyperparameters = #wave.hyperparameters<{BLOCK_M = 2048, M = 2048}>
  } {
    %cst = arith.constant 0.0 : f32
    %value1 = wave.register %cst : vector<8xf32>
    %value2 = wave.register %cst : vector<8xf32>
    %value3 = wave.register %cst : vector<8xf32>
    %value4 = wave.register %cst : vector<8xf32>
    %value5 = wave.register %cst : vector<8xf32>
    %value6 = wave.register %cst : vector<8xf32>

    // Single LDS allocation for the large stress test group (6 operations)
    // CHECK: %[[STRESS_ALLOC:.*]] = memref.alloc() : memref<2048xf32, #gpu.address_space<workgroup>>

    // All 6 register→LDS stores should happen first
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>
    // CHECK: vector.store %{{.*}}, %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<8xf32>

    // Single barrier for the entire large group
    // CHECK: amdgpu.lds_barrier
    // Ensure only one barrier despite large number of operations
    // CHECK-NOT: amdgpu.lds_barrier

    // All 6 LDS→register loads and register→global stores
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>
    // CHECK: %{{.*}} = vector.load %[[STRESS_ALLOC]][%{{.*}}] : memref<2048xf32, #gpu.address_space<workgroup>>, vector<64xf32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<2048xf32, #gpu.address_space<global>>, vector<64xf32>

    // Operation 1
    wave.write %value1, %mem1 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Operation 2
    wave.write %value2, %mem2 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 256, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 256)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Operation 3
    wave.write %value3, %mem3 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 512, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 512)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Operation 4
    wave.write %value4, %mem4 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 768, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 768)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Operation 5
    wave.write %value5, %mem5 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 1024, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 1024)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Operation 6
    wave.write %value6, %mem6 index [{
        M : [#wave.symbol<"BLOCK_M">, #wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (BLOCK_M * WG0 + T0 * 8 + 1280, 8, 1)
      }] {
      memory_access_pattern = #wave.memory_access_pattern<
        use_lds_promotion = true,
        group_id = "stress_test_group",
        lds_block_global_base = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M)>,
        lds_block_shape = #wave.expr_list<[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
        lds_load_indices = #wave.expr_list<[#wave.index_symbol<T0>] -> (T0 * 64)>,
        lds_load_vector_sizes = #wave.expr_list<[] -> (64)>,
        global_store_indices = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + T0 * 64 + 1280)>
        >
    } : vector<8xf32>, !wave.tensor<[@M] of f32, <global>>

    // Verify no wave.write operations remain
    // CHECK-NOT: wave.write

    return
  }
}
