// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: vector.maskedload and vector.maskedstore with proper exec masking.
// The mask disables lanes via s_and_saveexec_b64 before the memory access,
// then exec is restored via s_mov_b64 exec.

module {
func.func @masked_load_store(
    %binding_in: !stream.binding,
    %binding_out: !stream.binding,
    %M: index,
    %N: index) {

  %c0 = arith.constant 0 : index
  %passthru = arith.constant dense<0.0> : vector<1xf32>

  %flat_in = "stream.binding.subspan"(%binding_in, %c0) : (!stream.binding, index) -> memref<f32>
  %flat_out = "stream.binding.subspan"(%binding_out, %c0) : (!stream.binding, index) -> memref<f32>

  %in = memref.reinterpret_cast %flat_in to
      offset: [0], sizes: [%M, %N], strides: [%N, 1]
      : memref<f32> to memref<?x?xf32, strided<[?, 1]>>

  %out = memref.reinterpret_cast %flat_out to
      offset: [0], sizes: [%M, %N], strides: [%N, 1]
      : memref<f32> to memref<?x?xf32, strided<[?, 1]>>

  %tid = gpu.thread_id x

  // Bounds check: tid < M
  %cmp = arith.cmpi slt, %tid, %M : index
  %mask = vector.broadcast %cmp : i1 to vector<1xi1>

  // CHECK-LABEL: waveasm.program @masked_load_store

  // --- maskedload: exec mask around buffer_load ---
  // CHECK: waveasm.v_cmp_ne_u32
  // CHECK: waveasm.s_and_saveexec_b64
  // CHECK: waveasm.buffer_load
  // CHECK: waveasm.s_mov_b64_exec
  %loaded = vector.maskedload %in[%tid, %c0], %mask, %passthru
      : memref<?x?xf32, strided<[?, 1]>>, vector<1xi1>, vector<1xf32> into vector<1xf32>

  // --- maskedstore: exec mask around buffer_store ---
  // CHECK: waveasm.v_cmp_ne_u32
  // CHECK: waveasm.s_and_saveexec_b64
  // CHECK: waveasm.buffer_store
  // CHECK: waveasm.s_mov_b64_exec
  vector.maskedstore %out[%tid, %c0], %mask, %loaded
      : memref<?x?xf32, strided<[?, 1]>>, vector<1xi1>, vector<1xf32>

  return
}
}
