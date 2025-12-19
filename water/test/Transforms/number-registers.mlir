// RUN: water-opt %s --pass-pipeline='builtin.module(func.func(water-number-registers))' | FileCheck %s

// CHECK-LABEL: func @test_simple_numbering
// CHECK-SAME: attributes {water.total_vgprs = 8 : i32}
func.func @test_simple_numbering(%arg0: memref<100xf32>) -> f32 {
  %c0 = arith.constant 0 : index

  // 1xf32 = 4 bytes = 1 register, starts at reg 0
  // CHECK: memref.alloca() {water.vgpr_count = 1 : i32, water.vgpr_number = 0 : i32}
  %reg0 = memref.alloca() : memref<1xf32, 128 : i32>

  // 4xf32 = 16 bytes = 4 registers, starts at reg 4
  // CHECK: memref.alloca() {water.vgpr_count = 4 : i32, water.vgpr_number = 4 : i32}
  %reg1 = memref.alloca() : memref<4xf32, 128 : i32>

  // 1xf32 = 4 bytes = 1 register, starts at reg 1 (after reg0)
  // CHECK: memref.alloca() {water.vgpr_count = 1 : i32, water.vgpr_number = 1 : i32}
  %reg2 = memref.alloca() : memref<1xf32, 128 : i32>

  %subview0 = memref.subview %arg0[%c0] [1] [1] : memref<100xf32> to memref<1xf32, strided<[1], offset: ?>>
  memref.copy %subview0, %reg0 : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, 128 : i32>

  %val0 = memref.load %reg0[%c0] : memref<1xf32, 128 : i32>

  return %val0 : f32
}

// CHECK-LABEL: func @test_loop_with_registers
// CHECK-SAME: attributes {water.total_vgprs = 1 : i32}
func.func @test_loop_with_registers(%arg0: memref<100xf32>, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index

  // Register allocated outside loop
  // CHECK: memref.alloca() {water.vgpr_count = 1 : i32, water.vgpr_number = 0 : i32}
  %reg = memref.alloca() : memref<1xf32, 128 : i32>

  scf.for %iv = %lb to %ub step %step {
    %subview = memref.subview %arg0[%iv] [1] [1] : memref<100xf32> to memref<1xf32, strided<[1], offset: ?>>
    memref.copy %subview, %reg : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, 128 : i32>
    %val = memref.load %reg[%c0] : memref<1xf32, 128 : i32>
    memref.store %val, %arg0[%iv] : memref<100xf32>
  }

  return
}

// CHECK-LABEL: func @test_triple_buffering_numbering
// CHECK-SAME: attributes {water.total_vgprs = 12 : i32}
func.func @test_triple_buffering_numbering(%src: memref<1024xf32>, %lb: index, %ub: index, %step: index, %offset: index) {
  %c0 = arith.constant 0 : index

  // Three registers for triple buffering, each 4xf32 = 4 registers
  // CHECK: memref.alloca() {water.vgpr_count = 4 : i32, water.vgpr_number = 0 : i32}
  %reg0 = memref.alloca() : memref<4xf32, 128 : i32>

  // CHECK: memref.alloca() {water.vgpr_count = 4 : i32, water.vgpr_number = 4 : i32}
  %reg1 = memref.alloca() : memref<4xf32, 128 : i32>

  // CHECK: memref.alloca() {water.vgpr_count = 4 : i32, water.vgpr_number = 8 : i32}
  %reg2 = memref.alloca() : memref<4xf32, 128 : i32>

  return
}

// CHECK-LABEL: func @test_mixed_memspaces
// CHECK-SAME: attributes {water.total_vgprs = 1 : i32}
func.func @test_mixed_memspaces(%arg0: memref<100xf32>) {
  %c0 = arith.constant 0 : index

  // Non-register space alloca - should not be numbered
  // CHECK: memref.alloca() : memref<10xf32>
  // CHECK-NOT: water.vgpr_number
  %local = memref.alloca() : memref<10xf32>

  // Register space alloca - should be numbered
  // CHECK: memref.alloca() {water.vgpr_count = 1 : i32, water.vgpr_number = 0 : i32}
  %reg = memref.alloca() : memref<1xf32, 128 : i32>

  return
}
