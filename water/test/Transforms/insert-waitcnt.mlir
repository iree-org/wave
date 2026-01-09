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
