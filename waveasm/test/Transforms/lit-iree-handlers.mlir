// RUN: waveasm-translate %s | FileCheck %s
//
// Test: Additional handler coverage

// CHECK-LABEL: waveasm.program @test_complex_arith
func.func @test_complex_arith(%arg0: i32, %arg1: i32) -> i32 {
  // Test shift operations
  // CHECK: waveasm.s_lshl_b32
  %c3 = arith.constant 3 : i32
  %shl = arith.shli %arg0, %c3 : i32

  // Test comparison
  // CHECK: waveasm.s_cmp_lt_i32
  %cmp = arith.cmpi slt, %arg0, %arg1 : i32

  // Test select
  // CHECK: waveasm.s_cselect_b32
  %sel = arith.select %cmp, %shl, %arg1 : i32

  return %sel : i32
}

// CHECK-LABEL: waveasm.program @test_index_computation
func.func @test_index_computation(%arg0: index, %arg1: index) -> index {
  // Test index arithmetic
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index

  // Division by power of 2 -> shift
  // CHECK: waveasm.s_lshr_b32
  %div = arith.divui %arg0, %c64 : index

  // Modulo by power of 2 -> and
  // CHECK: waveasm.s_and_b32
  %rem = arith.remui %arg1, %c8 : index

  %result = arith.addi %div, %rem : index
  return %result : index
}

// CHECK-LABEL: waveasm.program @test_type_conversions
func.func @test_type_conversions(%arg0: i32, %arg1: i32) -> i32 {
  // Test basic i32 operations with bitwise ops
  // CHECK: waveasm.s_and_b32
  %mask = arith.constant 65535 : i32
  %masked = arith.andi %arg0, %mask : i32

  // CHECK: waveasm.s_add_u32
  %result = arith.addi %masked, %arg1 : i32
  return %result : i32
}
