// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: Barrett reduction for symbolic (runtime) divisors.
// Algorithm: float-rcp seed + Newton-Raphson refinement + v_mul_hi_u32
//            quotient + two remainder-based correction steps.
// Exact for all 32-bit unsigned values.

module {
func.func @barrett_floordiv(
    %binding: !stream.binding,
    %D: index) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x

  // CHECK-LABEL: waveasm.program @barrett_floordiv

  // --- Full Barrett reduction sequence for floordiv(tid, D) ---
  //
  // Step 1: Float reciprocal seed
  // CHECK: waveasm.v_cvt_f32_u32
  // CHECK: waveasm.v_rcp_f32
  // CHECK: waveasm.v_mul_f32
  // CHECK: waveasm.v_cvt_u32_f32
  //
  // Step 2: Newton-Raphson refinement
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_add_u32
  //
  // Step 3: Quotient via mulhi
  // CHECK: waveasm.v_mul_hi_u32
  //
  // Step 4: Remainder + first correction
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_subrev_u32
  // CHECK: waveasm.v_cmp_ge_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.v_cndmask_b32
  //
  // Step 5: Second correction
  // CHECK: waveasm.v_cmp_ge_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  %fdiv = affine.apply affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%tid, %D]

  return
}
}
