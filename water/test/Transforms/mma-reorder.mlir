// RUN: water-opt %s --water-mma-reorder | FileCheck %s

// Test: Consecutive ops with same B get reuseB, then same A gets reuseA.
// CHECK-LABEL: func.func @reorder_for_reuse
// CHECK-SAME:    (%[[A0:.*]]: vector<16xf16>, %[[A1:.*]]: vector<16xf16>, %[[B0:.*]]: vector<16xf16>, %[[B1:.*]]: vector<16xf16>, %[[C:.*]]: vector<32xf32>)
func.func @reorder_for_reuse(
    %a0: vector<16xf16>, %a1: vector<16xf16>,
    %b0: vector<16xf16>, %b1: vector<16xf16>,
    %c: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>) {
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A0]], %[[B0]], %[[C]] {reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A1]], %[[B0]], %[[C]] {reuseA = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A1]], %[[B1]], %[[C]] {reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A0]], %[[B1]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.wmma.f32.16x16x32.f16 %a0, %b0, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %1 = rocdl.wmma.f32.16x16x32.f16 %a1, %b0, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %2 = rocdl.wmma.f32.16x16x32.f16 %a0, %b1, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %3 = rocdl.wmma.f32.16x16x32.f16 %a1, %b1, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  return %0, %1, %2, %3 : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
}

// Test: Ops sharing B should get reuseB flag.
// CHECK-LABEL: func.func @reuse_b_flag
// CHECK-SAME:    (%[[A0:.*]]: vector<16xf16>, %[[A1:.*]]: vector<16xf16>, %[[B:.*]]: vector<16xf16>, %[[C:.*]]: vector<32xf32>)
func.func @reuse_b_flag(
    %a0: vector<16xf16>, %a1: vector<16xf16>,
    %b: vector<16xf16>, %c: vector<32xf32>) -> (vector<32xf32>, vector<32xf32>) {
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A0]], %[[B]], %[[C]] {reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A1]], %[[B]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.wmma.f32.16x16x32.f16 %a0, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %1 = rocdl.wmma.f32.16x16x32.f16 %a1, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  return %0, %1 : vector<32xf32>, vector<32xf32>
}

// Test: Chained accumulator ops preserve dependency order and get both reuse flags.
// CHECK-LABEL: func.func @chained_accumulator
// CHECK-SAME:    (%[[A:.*]]: vector<16xf16>, %[[B:.*]]: vector<16xf16>, %[[C:.*]]: vector<32xf32>)
func.func @chained_accumulator(
    %a: vector<16xf16>, %b: vector<16xf16>, %c: vector<32xf32>) -> vector<32xf32> {
  // CHECK: rocdl.sched.barrier 0
  // CHECK: %[[R0:.*]] = rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]] {reuseA = true, reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[R0]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %1 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %0 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  return %1 : vector<32xf32>
}

// Test: Single WMMA op - no reordering, no barriers (sequence too small).
// CHECK-LABEL: func.func @single_wmma
func.func @single_wmma(
    %a: vector<16xf16>, %b: vector<16xf16>, %c: vector<32xf32>) -> vector<32xf32> {
  // CHECK-NOT: rocdl.sched.barrier
  // CHECK: rocdl.wmma.f32.16x16x32.f16
  // CHECK-NOT: rocdl.sched.barrier
  %0 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  return %0 : vector<32xf32>
}

// Test: Non-consecutive WMMA ops are processed as separate sequences.
// CHECK-LABEL: func.func @non_consecutive_wmma
// CHECK-SAME:    (%[[A:.*]]: vector<16xf16>, %[[B:.*]]: vector<16xf16>, %[[C:.*]]: vector<32xf32>, %[[X:.*]]: f32)
func.func @non_consecutive_wmma(
    %a: vector<16xf16>, %b: vector<16xf16>, %c: vector<32xf32>,
    %x: f32) -> (vector<32xf32>, vector<32xf32>, f32) {
  // Two separate sequences separated by arith.addf.
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]] {reuseA = true, reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: arith.addf
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]] {reuseA = true, reuseB = true}
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %1 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %y = arith.addf %x, %x : f32
  %2 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %3 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  return %0, %2, %y : vector<32xf32>, vector<32xf32>, f32
}

// Test: MFMA ops get reordered for operand locality (no reuse flags).
// CHECK-LABEL: func.func @mfma_reorder
// CHECK-SAME:    (%[[A0:.*]]: vector<4xf16>, %[[A1:.*]]: vector<4xf16>, %[[B0:.*]]: vector<4xf16>, %[[B1:.*]]: vector<4xf16>, %[[C:.*]]: vector<4xf32>)
func.func @mfma_reorder(
    %a0: vector<4xf16>, %a1: vector<4xf16>,
    %b0: vector<4xf16>, %b1: vector<4xf16>,
    %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  // Reordering groups matching operands together.
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[A0]], %[[B0]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[A1]], %[[B0]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[A1]], %[[B1]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[A0]], %[[B1]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.mfma.f32.16x16x16f16 %a0, %b0, %c, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  %1 = rocdl.mfma.f32.16x16x16f16 %a1, %b0, %c, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  %2 = rocdl.mfma.f32.16x16x16f16 %a0, %b1, %c, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  %3 = rocdl.mfma.f32.16x16x16f16 %a1, %b1, %c, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  return %0, %1, %2, %3 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}

// Test: Mixed MFMA and WMMA ops in the same sequence.
// CHECK-LABEL: func.func @mixed_mfma_wmma
// CHECK-SAME:    (%[[A:.*]]: vector<16xf16>, %[[B:.*]]: vector<16xf16>, %[[C32:.*]]: vector<32xf32>, %[[MA:.*]]: vector<4xf16>, %[[MB:.*]]: vector<4xf16>, %[[MC:.*]]: vector<4xf32>)
func.func @mixed_mfma_wmma(
    %a: vector<16xf16>, %b: vector<16xf16>, %c32: vector<32xf32>,
    %ma: vector<4xf16>, %mb: vector<4xf16>, %mc: vector<4xf32>)
    -> (vector<32xf32>, vector<4xf32>, vector<32xf32>, vector<4xf32>) {
  // All four ops form one sequence since they are all matrix multiply ops.
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C32]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %[[A]], %[[B]], %[[C32]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[MA]], %[[MB]], %[[MC]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.f32.16x16x16f16 %[[MA]], %[[MB]], %[[MC]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c32 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %1 = rocdl.mfma.f32.16x16x16f16 %ma, %mb, %mc, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  %2 = rocdl.wmma.f32.16x16x32.f16 %a, %b, %c32 {signA = false, signB = false, modC = 0 : i16} : (vector<16xf16>, vector<16xf16>, vector<32xf32>) -> vector<32xf32>
  %3 = rocdl.mfma.f32.16x16x16f16 %ma, %mb, %mc, 0, 0, 0 : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  return %0, %1, %2, %3 : vector<32xf32>, vector<4xf32>, vector<32xf32>, vector<4xf32>
}

// Test: Scaled MFMA ops get reordered for operand locality.
// CHECK-LABEL: func.func @scaled_mfma_reorder
// CHECK-SAME:    (%[[A0:.*]]: vector<8xi32>, %[[A1:.*]]: vector<8xi32>, %[[B0:.*]]: vector<8xi32>, %[[B1:.*]]: vector<8xi32>, %[[C:.*]]: vector<16xf32>, %[[S:.*]]: i32)
func.func @scaled_mfma_reorder(
    %a0: vector<8xi32>, %a1: vector<8xi32>,
    %b0: vector<8xi32>, %b1: vector<8xi32>,
    %c: vector<16xf32>, %s: i32)
    -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 %[[A0]], %[[B0]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 %[[A1]], %[[B0]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 %[[A1]], %[[B1]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4 %[[A0]], %[[B1]], %[[C]]
  // CHECK: rocdl.sched.barrier 0
  %0 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %a0, %b0, %c, 0, 0, 0, %s, 0, %s : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
  %1 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %a1, %b0, %c, 0, 0, 0, %s, 0, %s : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
  %2 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %a0, %b1, %c, 0, 0, 0, %s, 0, %s : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
  %3 = rocdl.mfma.scale.f32.32x32x64.f8f6f4 %a1, %b1, %c, 0, 0, 0, %s, 0, %s : (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
  return %0, %1, %2, %3 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
}
