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
