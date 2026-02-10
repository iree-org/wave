// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test floor operation synthesis (power-of-2 division/modulo)

// Test 1: Division by power of 2 should use right shift
// CHECK-LABEL: waveasm.program @div_power_of_2
func.func @div_power_of_2(%arg0: i32) -> i32 {
  // Division by 16 -> shift right by 4
  %c16 = arith.constant 16 : i32
  %div = arith.divui %arg0, %c16 : i32
  return %div : i32
}

// Test 2: Division by non-power of 2 (placeholder for now)
// CHECK-LABEL: waveasm.program @div_general
func.func @div_general(%arg0: i32) -> i32 {
  %c7 = arith.constant 7 : i32
  %div = arith.divui %arg0, %c7 : i32
  return %div : i32
}

// Test 3: Modulo by power of 2 should use AND
// CHECK-LABEL: waveasm.program @mod_power_of_2
func.func @mod_power_of_2(%arg0: i32) -> i32 {
  // Modulo by 32 -> AND with 31
  %c32 = arith.constant 32 : i32
  %rem = arith.remui %arg0, %c32 : i32
  return %rem : i32
}

// Test 4: Common floor pattern: floor(x / 64)
// CHECK-LABEL: waveasm.program @floor_div_64
func.func @floor_div_64(%arg0: i32) -> i32 {
  %c64 = arith.constant 64 : i32
  %div = arith.divui %arg0, %c64 : i32
  return %div : i32
}

// Test 5: Combined pattern: (x / 8) + (x % 8)
// CHECK-LABEL: waveasm.program @div_and_mod
func.func @div_and_mod(%arg0: i32) -> i32 {
  %c8 = arith.constant 8 : i32
  %div = arith.divui %arg0, %c8 : i32
  %rem = arith.remui %arg0, %c8 : i32
  %sum = arith.addi %div, %rem : i32
  return %sum : i32
}
