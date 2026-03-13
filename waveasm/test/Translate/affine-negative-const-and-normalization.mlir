// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: correct handling of negative constants in affine expressions and
// normalization of >32-bit constants produced by MLIR's affine canonicalizer.
//
// Bug 1 (negative constants): BitRange::fromConstant(-16) previously returned
// an empty range because the old while(v > 0) loop never executed for negative
// values.  This caused the OR-for-add optimisation to fire incorrectly: the
// compiler emitted v_or_b32 instead of v_add_u32 for (tid floordiv 8) - 16,
// producing wrong results when tid >= 128 (bit 4 of the quotient overlaps
// with bit 4 of 0xFFFFFFF0).
//
// Bug 2 (large constants): MLIR's affine canonicalizer can compose nested
// affine.apply ops, multiplying coefficients by a large LCM factor that
// exceeds 32 bits.  The normalization pass divides all variable coefficients
// and the divisor by their GCD to bring them back into 32-bit range.

module {

// --- Negative constant: must use v_add_u32, not v_or_b32 ---

func.func @negative_const_add(%binding: !stream.binding) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x upper_bound 256

  // CHECK-LABEL: waveasm.program @negative_const_add

  // tid floordiv 8 = shift right by 3
  // CHECK: waveasm.v_lshrrev_b32
  // With upper_bound 256, floordiv 8 gives [0,31] (bits 0..4).
  // -16 = 0xFFFFFFF0 occupies bits 4..31 -- bit 4 overlaps.
  // CHECK: waveasm.constant -16
  // CHECK: waveasm.v_add_u32
  // CHECK-NOT: waveasm.v_or_b32
  %r = affine.apply affine_map<()[s0] -> (s0 floordiv 8 - 16)>()[%tid]

  return
}

// --- Large constant normalization (power-of-2 reduced divisor) ---

func.func @large_const_norm_po2(%binding: !stream.binding) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x
  %tid_y = gpu.thread_id y

  // CHECK-LABEL: waveasm.program @large_const_norm_po2

  // Input:  (s0 * 6000000000 + s1 * 9000000000) floordiv 12000000000
  // GCD of {6e9, 9e9, 12e9} = 3e9
  // Normalized: (s0 * 2 + s1 * 3) floordiv 4
  // s0*2 becomes lshlrev by 1; floordiv 4 becomes lshrrev by 2
  // All constants fit in 32 bits after normalization.
  // CHECK: waveasm.constant 1
  // CHECK: waveasm.v_lshlrev_b32
  // CHECK: waveasm.constant 3
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.constant 2
  // CHECK: waveasm.v_lshrrev_b32
  %r = affine.apply affine_map<()[s0, s1] -> ((s0 * 6000000000 + s1 * 9000000000) floordiv 12000000000)>()[%tid, %tid_y]

  return
}

// --- Large constant normalization (non-power-of-2 reduced divisor) ---

func.func @large_const_norm_magic(%binding: !stream.binding) {

  %c0 = arith.constant 0 : index
  %flat = "stream.binding.subspan"(%binding, %c0) : (!stream.binding, index) -> memref<f32>
  %tid = gpu.thread_id x
  %tid_y = gpu.thread_id y

  // CHECK-LABEL: waveasm.program @large_const_norm_magic

  // Input:  (s0 * 21000000000 + s1 * 14000000000) floordiv 49000000000
  // GCD of {21e9, 14e9, 49e9} = 7e9
  // Normalized: (s0 * 3 + s1 * 2) floordiv 7
  // floordiv 7 uses magic number multiplication (add form)
  // All constants fit in 32 bits after normalization.
  // CHECK: waveasm.constant 3
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_add_u32
  // floordiv 7 via magic number multiplication
  // CHECK: waveasm.v_mul_hi_u32
  %r = affine.apply affine_map<()[s0, s1] -> ((s0 * 21000000000 + s1 * 14000000000) floordiv 49000000000)>()[%tid, %tid_y]

  return
}

}
