// RUN: waveasm-translate %s | FileCheck %s
//
// Test: MLIR to WAVEASM translation

// CHECK-LABEL: waveasm.program @translate_arith
func.func @translate_arith(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: waveasm.v_add_u32
  %add = arith.addi %arg0, %arg1 : i32

  // CHECK: waveasm.v_sub_u32
  %sub = arith.subi %add, %arg1 : i32

  // CHECK: waveasm.v_mul_lo_u32
  %mul = arith.muli %sub, %arg0 : i32

  return %mul : i32
}

// CHECK-LABEL: waveasm.program @translate_bitwise
func.func @translate_bitwise(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: waveasm.v_and_b32
  %and = arith.andi %arg0, %arg1 : i32

  // CHECK: waveasm.v_or_b32
  %or = arith.ori %and, %arg1 : i32

  // CHECK: waveasm.v_xor_b32
  %xor = arith.xori %or, %arg0 : i32

  return %xor : i32
}

// CHECK-LABEL: waveasm.program @translate_shifts
func.func @translate_shifts(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: waveasm.v_lshlrev_b32
  %shl = arith.shli %arg0, %arg1 : i32

  // CHECK: waveasm.v_lshrrev_b32
  %shr = arith.shrui %shl, %arg1 : i32

  // CHECK: waveasm.v_ashrrev_i32
  %ashr = arith.shrsi %shr, %arg1 : i32

  return %ashr : i32
}

// CHECK-LABEL: waveasm.program @translate_constant
func.func @translate_constant() -> i32 {
  // CHECK: waveasm.constant 42
  %c42 = arith.constant 42 : i32
  return %c42 : i32
}

// CHECK-LABEL: waveasm.program @translate_div_mod_pow2
func.func @translate_div_mod_pow2(%arg0: i32) -> i32 {
  // Division by power of 2 should use shift
  // CHECK: waveasm.constant 4
  // CHECK: waveasm.v_lshrrev_b32
  %c16 = arith.constant 16 : i32
  %div = arith.divui %arg0, %c16 : i32

  // Modulo by power of 2 should use AND
  // CHECK: waveasm.constant 7
  // CHECK: waveasm.v_and_b32
  %c8 = arith.constant 8 : i32
  %rem = arith.remui %arg0, %c8 : i32

  %result = arith.addi %div, %rem : i32
  return %result : i32
}
