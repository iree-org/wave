// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: affine.apply with symbolic (runtime) divisors for floordiv, ceildiv,
// and mod. These use Barrett reduction (float-rcp seed + Newton-Raphson
// refinement + v_mul_hi_u32 quotient + two correction steps).

module {
func.func @symbolic_div(
    %binding: !stream.binding,
    %D: index) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x

  // CHECK-LABEL: waveasm.program @symbolic_div

  // --- Test 1: floordiv with symbolic divisor (Barrett reduction) ---
  // Seed: cvt_f32 -> rcp -> mul(0x4f7ffffe) -> cvt_u32
  // Newton-Raphson: sub(0,d) -> mul_lo -> mul_hi -> add
  // Quotient: mul_hi(x, recip)
  // Correction: mul_lo, sub, subrev, cmp_ge, cndmask, add, cndmask, cmp_ge, cndmask, add
  // CHECK: waveasm.v_cvt_f32_u32
  // CHECK: waveasm.v_rcp_f32
  // CHECK: waveasm.v_mul_f32
  // CHECK: waveasm.v_cvt_u32_f32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_subrev_u32
  // CHECK: waveasm.v_cmp_ge_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_cmp_ge_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  %fdiv = affine.apply affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%tid, %D]

  // --- Test 2: ceildiv with symbolic divisor ---
  // Overflow-safe: Barrett floordiv(tid, D), then q + (rem != 0 ? 1 : 0)
  // CHECK: waveasm.v_cvt_f32_u32
  // CHECK: waveasm.v_rcp_f32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_cmp_ne_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  %cdiv = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%tid, %D]

  // --- Test 3: mod with symbolic divisor ---
  // Expect: Barrett floordiv(tid, D), then tid - q*D
  // CHECK: waveasm.v_cvt_f32_u32
  // CHECK: waveasm.v_rcp_f32
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  %mod = affine.apply affine_map<()[s0, s1] -> (s0 mod s1)>()[%tid, %D]

  return
}
}
