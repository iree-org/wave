// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: arith.cmpi with VGPR operands materializes VCC into a VGPR via
// V_CNDMASK_B32, arith.select restores VCC before conditional move,
// and arith.cmpf also materializes VCC into a VGPR.

module {
func.func @cmpi_select_cmpf(
    %binding_a: !stream.binding,
    %binding_c: !stream.binding,
    %M: index) {

  %c0 = arith.constant 0 : index

  %flat_a = "stream.binding.subspan"(%binding_a, %c0) : (!stream.binding, index) -> memref<f32>
  %flat_c = "stream.binding.subspan"(%binding_c, %c0) : (!stream.binding, index) -> memref<f32>

  %a = memref.reinterpret_cast %flat_a to
      offset: [0], sizes: [%M], strides: [1]
      : memref<f32> to memref<?xf32, strided<[1]>>

  %c_buf = memref.reinterpret_cast %flat_c to
      offset: [0], sizes: [%M], strides: [1]
      : memref<f32> to memref<?xf32, strided<[1]>>

  %tid = gpu.thread_id x

  // Load a value from memory so we have a VGPR operand for cmpf
  %v = vector.load %a[%tid] : memref<?xf32, strided<[1]>>, vector<1xf32>
  %vs = vector.extract %v[0] : f32 from vector<1xf32>

  // --- Test 1: arith.cmpi slt with VGPR index operands ---
  // tid and M are both VGPRs -> V_CMP + VCC materialization
  // CHECK-LABEL: waveasm.program @cmpi_select_cmpf
  // CHECK: waveasm.v_cmp_lt_i32
  // CHECK: waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<1> -> !waveasm.vreg
  // CHECK: waveasm.v_cndmask_b32
  %cmp = arith.cmpi slt, %tid, %M : index

  // --- Test 2: arith.select restores VCC from materialized bool ---
  // CHECK: waveasm.v_cmp_ne_u32
  // CHECK: waveasm.v_cndmask_b32
  %cf0 = arith.constant 0.0 : f32
  %sel = arith.select %cmp, %vs, %cf0 : f32

  // --- Test 3: arith.cmpf with VGPR operands ---
  // CHECK: waveasm.v_cmp_lt_f32
  // CHECK: waveasm.v_mov_b32 %{{.*}} : !waveasm.imm<1> -> !waveasm.vreg
  // CHECK: waveasm.v_cndmask_b32
  %cmpf = arith.cmpf olt, %vs, %cf0 : f32

  return
}
}
