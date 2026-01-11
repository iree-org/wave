// RUN: water-opt %s --water-insert-waitcnt | FileCheck %s

// CHECK-LABEL: func.func @no_dependency
func.func @no_dependency(%global1: memref<64x64xf32>, %global2: memref<64x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index

  // First load to LDS1.
  %base1 = amdgpu.make_dma_base %global1[%c0, %c0], %lds1[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc1 = amdgpu.make_dma_descriptor %base1 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc1 : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Second load to different LDS2 (no dependency).
  %base2 = amdgpu.make_dma_base %global2[%c0, %c0], %lds2[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc2 = amdgpu.make_dma_descriptor %base2 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  // CHECK-NOT: amdgpu.memory_counter_wait
  amdgpu.tensor_load_to_lds %desc2 : !amdgpu.tdm_descriptor

  return
}

// CHECK-LABEL: func.func @raw_dependency_vector_load
func.func @raw_dependency_vector_load(%global: memref<64x64xf32>, %lds: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index

  // Tensor load to LDS (write to LDS).
  %base = amdgpu.make_dma_base %global[%c0, %c0], %lds[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc = amdgpu.make_dma_descriptor %base globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Vector load from LDS (read from LDS) - creates RAW dependency.
  // CHECK: amdgpu.memory_counter_wait tensor(0)
  // CHECK: amdgpu.lds_barrier
  %vec = vector.load %lds[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  return
}

// CHECK-LABEL: func.func @multiple_pending_ops
func.func @multiple_pending_ops(%global: memref<64x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index

  // First tensor load to LDS1.
  %base1 = amdgpu.make_dma_base %global[%c0, %c0], %lds1[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc1 = amdgpu.make_dma_descriptor %base1 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc1 : !amdgpu.tdm_descriptor

  // Second tensor load to LDS2.
  %base2 = amdgpu.make_dma_base %global[%c0, %c0], %lds2[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc2 = amdgpu.make_dma_descriptor %base2 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc2 : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Vector load from LDS1 - has RAW dependency with first load, should wait for 1 (second op is still pending).
  // CHECK: amdgpu.memory_counter_wait tensor(1)
  // CHECK: amdgpu.lds_barrier
  %vec = vector.load %lds1[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  return
}

// CHECK-LABEL: func.func @rocdl_barrier_signal
func.func @rocdl_barrier_signal(%global: memref<64x64xf32>, %lds: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index

  // Tensor load to LDS.
  %base = amdgpu.make_dma_base %global[%c0, %c0], %lds[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc = amdgpu.make_dma_descriptor %base globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor

  // ROCDL barrier signal/wait.
  rocdl.s.barrier.signal id = 0
  rocdl.s.barrier.wait id = 0

  // Vector load from LDS - creates RAW dependency.
  // CHECK: amdgpu.memory_counter_wait tensor(0)
  // CHECK-NEXT: rocdl.s.barrier.signal id = 0
  // CHECK-NEXT: rocdl.s.barrier.wait id = 0
  %vec = vector.load %lds[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>


  return
}


// CHECK-LABEL: func.func @scf_for_loop
func.func @scf_for_loop(%global: memref<64x64xf32>, %lds: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // Tensor load to LDS before loop.
  %base = amdgpu.make_dma_base %global[%c0, %c0], %lds[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc = amdgpu.make_dma_descriptor %base globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Loop that reads from LDS - should insert wait before loop.
  // CHECK: amdgpu.memory_counter_wait tensor(0)
  // CHECK: amdgpu.lds_barrier
  // CHECK: scf.for
  scf.for %i = %c0 to %c4 step %c1 {
    %vec = vector.load %lds[%i, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  }

  return
}

// CHECK-LABEL: func.func @double_buffer_loop
func.func @double_buffer_loop(%global: memref<512x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index

  // Initial load to buffer 1.
  %base_init = amdgpu.make_dma_base %global[%c0, %c0], %lds1[%c0, %c0] : memref<512x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc_init = amdgpu.make_dma_descriptor %base_init globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc_init : !amdgpu.tdm_descriptor

  // Double buffer loop: load next chunk while processing current chunk.
  // CHECK: scf.for
  scf.for %i = %c0 to %c8 step %c1 iter_args(%current_buf = %lds1, %next_buf = %lds2) -> (memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>) {
    // Load next data to next_buf.
    %next_idx = arith.addi %i, %c1 : index
    %global_offset = arith.muli %next_idx, %c64 : index
    %base_next = amdgpu.make_dma_base %global[%global_offset, %c0], %next_buf[%c0, %c0] : memref<512x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
    %desc_next = amdgpu.make_dma_descriptor %base_next globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
    amdgpu.tensor_load_to_lds %desc_next : !amdgpu.tdm_descriptor

    // Barrier to ensure previous load completed.
    // CHECK: amdgpu.memory_counter_wait tensor(1)
    // CHECK-NEXT: amdgpu.lds_barrier
    amdgpu.lds_barrier

    // Process current buffer (read from the buffer loaded in previous iteration).
    %vec = vector.load %current_buf[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // Swap buffers for next iteration.
    scf.yield %next_buf, %current_buf : memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  }

  return
}

// CHECK-LABEL: func.func @triple_buffer_loop
func.func @triple_buffer_loop(%global: memref<512x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds3: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index

  // Initial load to buffer 1.
  %base_init1 = amdgpu.make_dma_base %global[%c0, %c0], %lds1[%c0, %c0] : memref<512x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc_init1 = amdgpu.make_dma_descriptor %base_init1 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc_init1 : !amdgpu.tdm_descriptor

  // Initial load to buffer 2.
  %base_init2 = amdgpu.make_dma_base %global[%c64, %c0], %lds2[%c0, %c0] : memref<512x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc_init2 = amdgpu.make_dma_descriptor %base_init2 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc_init2 : !amdgpu.tdm_descriptor

  // Triple buffer loop: load next chunk while processing current chunk, with two extra buffers in flight.
  // CHECK: scf.for
  scf.for %i = %c0 to %c8 step %c1 iter_args(%current_buf = %lds1, %next_buf = %lds2, %next_next_buf = %lds3) -> (memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>) {
    // Load next data to next_next_buf (2 iterations ahead).
    %next_idx = arith.addi %i, %c1 : index
    %global_offset = arith.muli %next_idx, %c64 : index
    %base_next = amdgpu.make_dma_base %global[%global_offset, %c0], %next_next_buf[%c0, %c0] : memref<512x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
    %desc_next = amdgpu.make_dma_descriptor %base_next globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
    amdgpu.tensor_load_to_lds %desc_next : !amdgpu.tdm_descriptor

    // Barrier to ensure load from 2 iterations ago completed.
    // With triple buffering, we can have 2 loads in flight (current + previous iteration).
    // Wait until at most 2 remain, ensuring the oldest (from 2 iters ago) is done.
    // CHECK: amdgpu.memory_counter_wait tensor(2)
    // CHECK-NEXT: amdgpu.lds_barrier
    amdgpu.lds_barrier

    // Process current buffer (read from the buffer loaded 2 iterations ago).
    %vec = vector.load %current_buf[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // Rotate buffers for next iteration.
    scf.yield %next_buf, %next_next_buf, %current_buf : memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  }

  return
}

// CHECK-LABEL: func.func @vector_ops_only
func.func @vector_ops_only(%lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>

  // Loop with vector load/store but no tensor operations.
  // CHECK: scf.for
  // CHECK-NOT: amdgpu.memory_counter_wait
  scf.for %i = %c0 to %c4 step %c1 {
    // Vector load from LDS1.
    %vec = vector.load %lds1[%i, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

    // Barrier.
    amdgpu.lds_barrier

    // Vector store to LDS2.
    vector.store %vec, %lds2[%i, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  }

  return
}

// CHECK-LABEL: func.func @select_memref
func.func @select_memref(%global: memref<64x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>, %cond: i1) {
  %c0 = arith.constant 0 : index

  // Tensor load to LDS1.
  %base1 = amdgpu.make_dma_base %global[%c0, %c0], %lds1[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc1 = amdgpu.make_dma_descriptor %base1 globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc1 : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Select between LDS1 and LDS2 based on condition.
  %selected = arith.select %cond, %lds1, %lds2 : memref<64x64xf32, #gpu.address_space<workgroup>>

  // Vector load from selected buffer - should detect dependency with LDS1.
  // CHECK: amdgpu.memory_counter_wait tensor(0)
  // CHECK-NEXT: amdgpu.lds_barrier
  %vec = vector.load %selected[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  return
}

// CHECK-LABEL: func.func @select_tensor_base
func.func @select_tensor_base(%global: memref<64x64xf32>, %lds1: memref<64x64xf32, #gpu.address_space<workgroup>>, %lds2: memref<64x64xf32, #gpu.address_space<workgroup>>, %cond: i1) {
  %c0 = arith.constant 0 : index

  // Select which LDS buffer to use for tensor load.
  %selected_lds = arith.select %cond, %lds1, %lds2 : memref<64x64xf32, #gpu.address_space<workgroup>>

  // Tensor load using selected LDS buffer - writes to either LDS1 or LDS2.
  %base = amdgpu.make_dma_base %global[%c0, %c0], %selected_lds[%c0, %c0] : memref<64x64xf32>, memref<64x64xf32, #gpu.address_space<workgroup>> -> !amdgpu.tdm_base<f32>
  %desc = amdgpu.make_dma_descriptor %base globalSize [64, 64] globalStride [64, 1] sharedSize [64, 64] : !amdgpu.tdm_base<f32> -> !amdgpu.tdm_descriptor
  amdgpu.tensor_load_to_lds %desc : !amdgpu.tdm_descriptor

  // Barrier.
  amdgpu.lds_barrier

  // Vector load from LDS1 - should detect dependency since tensor load might have written here.
  // CHECK: amdgpu.memory_counter_wait tensor(0)
  // CHECK-NEXT: amdgpu.lds_barrier
  %vec = vector.load %lds1[%c0, %c0] : memref<64x64xf32, #gpu.address_space<workgroup>>, vector<4xf32>

  return
}
