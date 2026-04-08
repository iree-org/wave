// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: dynamic shapes support -- index-typed kernel arguments loaded from
// kernarg buffer, dynamic SRD buffer sizes, vector.store translation with
// dynamic stride address computation from memref.reinterpret_cast.

// CHECK-LABEL: waveasm.program @dynamic_shapes_kernel

// Test 1: index args loaded from kernarg buffer via s_load_dwordx2
// CHECK: waveasm.raw "s_load_dwordx2
// CHECK: waveasm.raw "s_load_dwordx2

// Test 2: SRD buffer size is 0x7FFFFFFE (2147483646, sentinel-safe max) for dynamic memrefs.
// CHECK: 2147483646

// Test 3: scalar args moved to VGPRs after SRD setup
// CHECK: waveasm.raw "v_mov_b32
// CHECK: waveasm.raw "v_mov_b32

// Test 4: dynamic stride address computation (runtime v_mul_lo_u32)
// CHECK: waveasm.v_mul_lo_u32
// CHECK: waveasm.v_mul_lo_u32

// Test 5: vector.store emits buffer_store
// CHECK: waveasm.buffer_store

module {
func.func @dynamic_shapes_kernel(
    %binding_a: !stream.binding,
    %binding_c: !stream.binding,
    %M: index,
    %N: index) {

  %c0 = arith.constant 0 : index

  %flat_a = "stream.binding.subspan"(%binding_a, %c0) : (!stream.binding, index) -> memref<f16>
  %flat_c = "stream.binding.subspan"(%binding_c, %c0) : (!stream.binding, index) -> memref<f32>

  // Reinterpret with dynamic sizes and static strides (input)
  %a = memref.reinterpret_cast %flat_a to
      offset: [0], sizes: [%M, 128], strides: [128, 1]
      : memref<f16> to memref<?x128xf16, strided<[128, 1]>>

  // Reinterpret with dynamic sizes AND dynamic stride (output)
  %c = memref.reinterpret_cast %flat_c to
      offset: [0], sizes: [%M, %N], strides: [%N, 1]
      : memref<f32> to memref<?x?xf32, strided<[?, 1]>>

  %thread_id = gpu.thread_id x

  // Load from input
  %loaded = vector.load %a[%thread_id, %c0]
      : memref<?x128xf16, strided<[128, 1]>>, vector<1xf16>
  %ext = arith.extf %loaded : vector<1xf16> to vector<1xf32>

  // Store with dynamic stride to output
  vector.store %ext, %c[%thread_id, %c0]
      : memref<?x?xf32, strided<[?, 1]>>, vector<1xf32>

  return
}
}
