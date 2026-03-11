// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: integer division and modulo lowering
//   - Power-of-2: shift / AND
//   - Non-power-of-2 constant: magic number multiplication (Hacker's Delight)

// Test 1: Division by power of 2 should use right shift
// CHECK-LABEL: waveasm.program @div_power_of_2
func.func @div_power_of_2(%arg0: i32) -> i32 {
  // Division by 16 -> shift right by 4
  // CHECK: waveasm.constant 4
  // CHECK: waveasm.v_lshrrev_b32
  %c16 = arith.constant 16 : i32
  %div = arith.divui %arg0, %c16 : i32
  return %div : i32
}

// Test 2: Division by non-power-of-2 constant uses magic number
// CHECK-LABEL: waveasm.program @div_by_7
func.func @div_by_7(%arg0: i32) -> i32 {
  // div by 7: magic = 0x24924925, shift = 2, needsAdd = true
  // Expect: mul_hi -> sub -> lshrrev(1) -> add -> lshrrev(shift-1)
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_sub_u32
  // CHECK: waveasm.v_lshrrev_b32
  // CHECK: waveasm.v_add_u32
  // CHECK: waveasm.v_lshrrev_b32
  %c7 = arith.constant 7 : i32
  %div = arith.divui %arg0, %c7 : i32
  return %div : i32
}

// Test 3: Division by non-power-of-2 constant (simple form, no add)
// CHECK-LABEL: waveasm.program @div_by_3
func.func @div_by_3(%arg0: i32) -> i32 {
  // div by 3: magic = 0xAAAAAAAB, shift = 1, needsAdd = false
  // Expect: mul_hi -> lshrrev(1)
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_lshrrev_b32
  %c3 = arith.constant 3 : i32
  %div = arith.divui %arg0, %c3 : i32
  return %div : i32
}

// Test 4: Modulo by power of 2 should use AND
// CHECK-LABEL: waveasm.program @mod_power_of_2
func.func @mod_power_of_2(%arg0: i32) -> i32 {
  // Modulo by 32 -> AND with 31
  // CHECK: waveasm.constant 31
  // CHECK: waveasm.v_and_b32
  %c32 = arith.constant 32 : i32
  %rem = arith.remui %arg0, %c32 : i32
  return %rem : i32
}

// Test 5: Modulo by non-power-of-2 constant uses magic + subtract
// CHECK-LABEL: waveasm.program @mod_by_5
func.func @mod_by_5(%arg0: i32) -> i32 {
  // rem by 5 = x - floordiv(x, 5) * 5
  // Expect: mul_hi (magic), lshrrev (shift), mul_lo (q*5), sub (x - q*5)
  // CHECK: waveasm.v_mul_hi_u32
  // CHECK: waveasm.v_lshrrev_b32
  // CHECK: waveasm.v_mul_lo_u32
  // CHECK: waveasm.v_sub_u32
  %c5 = arith.constant 5 : i32
  %rem = arith.remui %arg0, %c5 : i32
  return %rem : i32
}

// Test 6: Common floor pattern: floor(x / 64) - power of 2
// CHECK-LABEL: waveasm.program @floor_div_64
func.func @floor_div_64(%arg0: i32) -> i32 {
  // CHECK: waveasm.constant 6
  // CHECK: waveasm.v_lshrrev_b32
  %c64 = arith.constant 64 : i32
  %div = arith.divui %arg0, %c64 : i32
  return %div : i32
}

// Test 7: Combined div + mod (same constant)
// CHECK-LABEL: waveasm.program @div_and_mod_by_8
func.func @div_and_mod_by_8(%arg0: i32) -> i32 {
  // div by 8 -> shift right by 3
  // CHECK: waveasm.v_lshrrev_b32
  %c8 = arith.constant 8 : i32
  %div = arith.divui %arg0, %c8 : i32
  // mod by 8 -> AND with 7
  // CHECK: waveasm.v_and_b32
  %rem = arith.remui %arg0, %c8 : i32
  %sum = arith.addi %div, %rem : i32
  return %sum : i32
}
