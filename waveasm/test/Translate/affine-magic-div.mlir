// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: affine.apply with constant non-power-of-2 divisors.
// These use magic number multiplication (Hacker's Delight):
//   Simple form: mulhi(x, magic) >> shift
//   Add form:    (mulhi(x, m) + ((x - mulhi(x, m)) >> 1)) >> shift

module {
func.func @magic_div(%binding: !stream.binding) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x

  // CHECK-LABEL: waveasm.program @magic_div

  // --- Test 1: floordiv by constant 3 (simple form) ---
  // magic = 0xAAAAAAAB, shift = 1, needsAdd = false
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_lshrrev_b32
  %div3 = affine.apply affine_map<()[s0] -> (s0 floordiv 3)>()[%tid]

  // --- Test 2: floordiv by constant 7 (add form) ---
  // magic = 0x24924925, shift = 2, needsAdd = true
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_lshrrev_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.v_lshrrev_b32
  %div7 = affine.apply affine_map<()[s0] -> (s0 floordiv 7)>()[%tid]

  // --- Test 3: mod by constant 5 ---
  // rem = x - floordiv(x, 5) * 5
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  %mod5 = affine.apply affine_map<()[s0] -> (s0 mod 5)>()[%tid]

  // --- Test 4: ceildiv by constant 6 ---
  // ceildiv(x, 6) = floordiv(x, 6) + (x mod 6 != 0 ? 1 : 0)
  // Overflow-safe: avoids (x + 5) wrapping modulo 2^32.
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_lshrrev_b32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_cmp_ne_u32
  // CHECK: waveasm.v_cndmask_b32
  // CHECK: waveasm.v_add_u32
  %cdiv6 = affine.apply affine_map<()[s0] -> (s0 ceildiv 6)>()[%tid]

  // --- Test 5: floordiv by constant 2 (power of 2, should use shift) ---
  // CHECK: waveasm.v_lshrrev_b32
  // CHECK-NOT: waveasm.v_mul_hi_u32
  %div2 = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%tid]

  return
}
}
