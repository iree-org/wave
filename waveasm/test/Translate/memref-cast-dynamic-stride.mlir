// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: memref.cast propagates dynamic strides from memref.reinterpret_cast
// through the cast chain so that vector.store correctly uses runtime
// v_mul_lo_u32 for address computation instead of static stride constants.

module {
func.func @cast_stride_propagation(
    %binding_in: !stream.binding,
    %binding_out: !stream.binding,
    %M: index,
    %N: index) {

  %c0 = arith.constant 0 : index

  %flat_in = "stream.binding.subspan"(%binding_in, %c0) : (!stream.binding, index) -> memref<f32>
  %flat_out = "stream.binding.subspan"(%binding_out, %c0) : (!stream.binding, index) -> memref<f32>

  // Reinterpret input with static stride (for loading a VGPR value)
  %in = memref.reinterpret_cast %flat_in to
      offset: [0], sizes: [%M], strides: [1]
      : memref<f32> to memref<?xf32, strided<[1]>>

  // Reinterpret output with dynamic stride
  %rc = memref.reinterpret_cast %flat_out to
      offset: [0], sizes: [%M, %N], strides: [%N, 1]
      : memref<f32> to memref<?x?xf32, strided<[?, 1]>>

  // Cast erases static info but must preserve dynamic stride metadata
  %cast = memref.cast %rc
      : memref<?x?xf32, strided<[?, 1]>>
        to memref<?x?xf32, strided<[?, 1]>>

  %tid = gpu.thread_id x

  // Load a value to get a VGPR for the store
  %v = vector.load %in[%tid] : memref<?xf32, strided<[1]>>, vector<1xf32>

  // CHECK-LABEL: waveasm.program @cast_stride_propagation

  // The store through the cast chain must use dynamic stride (v_mul_lo_u32),
  // not a static stride constant. This confirms memref.cast propagated
  // the dynamic stride from reinterpret_cast.
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.buffer_store

  vector.store %v, %cast[%tid, %c0]
      : memref<?x?xf32, strided<[?, 1]>>, vector<1xf32>

  return
}
}
