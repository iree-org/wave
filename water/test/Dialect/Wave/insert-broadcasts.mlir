// RUN: water-opt %s --water-wave-insert-broadcasts | FileCheck %s

// CHECK-LABEL: @insert_broadcast_after_div
func.func @insert_broadcast_after_div() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: %[[TID:.*]] = gpu.thread_id  x
  %tid = gpu.thread_id x
  // CHECK: %[[C64:.*]] = arith.constant 64
  %c64 = arith.constant 64 : index
  // CHECK: %[[DIV:.*]] = arith.divui %[[TID]], %[[C64]]
  // CHECK: %[[BCAST:.*]] = gpu.subgroup_broadcast %[[DIV]],  first_active_lane
  // CHECK: return %[[BCAST]]
  %warp_id = arith.divui %tid, %c64 : index
  return %warp_id : index
}

// -----

// CHECK-LABEL: @no_broadcast_after_broadcast
func.func @no_broadcast_after_broadcast(%arg0: i32) -> i32 {
  // CHECK: gpu.thread_id
  %tid = gpu.thread_id x
  // CHECK: arith.index_cast
  %tid_i32 = arith.index_cast %tid : index to i32
  // CHECK: arith.addi
  %divergent = arith.addi %tid_i32, %arg0 : i32
  // CHECK: %[[BCAST:.*]] = gpu.subgroup_broadcast
  // CHECK-NOT: gpu.subgroup_broadcast %[[BCAST]]
  // CHECK: return %[[BCAST]]
  %broadcast = gpu.subgroup_broadcast %divergent, first_active_lane : i32
  return %broadcast : i32
}

// -----

// CHECK-LABEL: @insert_broadcast_after_and
func.func @insert_broadcast_after_and() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: %[[TID:.*]] = gpu.thread_id  x
  %tid = gpu.thread_id x
  // CHECK: %[[MASK:.*]] = arith.constant -64
  %mask = arith.constant -64 : index
  // CHECK: %[[AND:.*]] = arith.andi %[[TID]], %[[MASK]]
  // CHECK: %[[BCAST:.*]] = gpu.subgroup_broadcast %[[AND]],  first_active_lane
  // CHECK: return %[[BCAST]]
  %result = arith.andi %tid, %mask : index
  return %result : index
}

// -----

// CHECK-LABEL: @no_broadcast_for_uniform_inputs
func.func @no_broadcast_for_uniform_inputs() -> index {
  // CHECK: %[[BID:.*]] = gpu.block_id  x
  %bid = gpu.block_id x
  // CHECK: %[[C2:.*]] = arith.constant 2
  %c2 = arith.constant 2 : index
  // CHECK: %[[ADD:.*]] = arith.addi %[[BID]], %[[C2]]
  // CHECK-NOT: gpu.subgroup_broadcast
  // CHECK: return %[[ADD]]
  %result = arith.addi %bid, %c2 : index
  return %result : index
}
