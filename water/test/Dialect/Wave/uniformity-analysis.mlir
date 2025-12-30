// RUN: water-opt %s --water-wave-uniformity-analysis | FileCheck %s

// CHECK-LABEL: @constant_uniform
func.func @constant_uniform() -> i32 {
  // CHECK: arith.constant {wave.uniform}
  %c = arith.constant 42 : i32
  return %c : i32
}

// -----

// CHECK-LABEL: @thread_id_x_divergent
func.func @thread_id_x_divergent() -> index {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  return %tid : index
}

// -----

// CHECK-LABEL: @thread_id_y_uniform
func.func @thread_id_y_uniform() -> index {
  // CHECK: gpu.thread_id y {wave.uniform}
  %tid = gpu.thread_id y
  return %tid : index
}

// -----

// CHECK-LABEL: @thread_id_z_uniform
func.func @thread_id_z_uniform() -> index {
  // CHECK: gpu.thread_id z {wave.uniform}
  %tid = gpu.thread_id z
  return %tid : index
}

// -----

// CHECK-LABEL: @lane_id_divergent
func.func @lane_id_divergent() -> index {
  // CHECK: gpu.lane_id
  // CHECK-NOT: wave.uniform
  %lid = gpu.lane_id
  return %lid : index
}

// -----

// CHECK-LABEL: @all_thread_dims
func.func @all_thread_dims() -> (index, index, index) {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %x = gpu.thread_id x
  // CHECK: gpu.thread_id y {wave.uniform}
  %y = gpu.thread_id y
  // CHECK: gpu.thread_id z {wave.uniform}
  %z = gpu.thread_id z
  return %x, %y, %z : index, index, index
}

// -----

// CHECK-LABEL: @block_ids_uniform
func.func @block_ids_uniform() -> (index, index, index) {
  // CHECK: gpu.block_id x {wave.uniform}
  %x = gpu.block_id x
  // CHECK: gpu.block_id y {wave.uniform}
  %y = gpu.block_id y
  // CHECK: gpu.block_id z {wave.uniform}
  %z = gpu.block_id z
  return %x, %y, %z : index, index, index
}

// -----

// CHECK-LABEL: @divergent_propagation
func.func @divergent_propagation() -> index {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.addi
  // CHECK-NOT: wave.uniform
  %result = arith.addi %tid, %tid : index
  return %result : index
}

// -----

// CHECK-LABEL: @uniform_propagation
func.func @uniform_propagation() -> index {
  // CHECK: gpu.thread_id y {wave.uniform}
  %tid = gpu.thread_id y
  // CHECK: arith.addi {{.*}} {wave.uniform}
  %result = arith.addi %tid, %tid : index
  return %result : index
}

// -----

// CHECK-LABEL: @mixed_uniform_divergent
func.func @mixed_uniform_divergent() -> index {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid_x = gpu.thread_id x
  // CHECK: gpu.thread_id y {wave.uniform}
  %tid_y = gpu.thread_id y
  // CHECK: arith.addi
  // CHECK-NOT: wave.uniform
  %result = arith.addi %tid_x, %tid_y : index
  return %result : index
}

// -----

// CHECK-LABEL: @subgroup_broadcast_uniform
func.func @subgroup_broadcast_uniform(%arg0: i32) -> i32 {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_i32 = arith.index_cast %tid : index to i32
  // CHECK: arith.addi
  // CHECK-NOT: wave.uniform
  %divergent = arith.addi %tid_i32, %arg0 : i32
  // CHECK: gpu.subgroup_broadcast {{.*}} {wave.uniform}
  %broadcast = gpu.subgroup_broadcast %divergent, first_active_lane : i32
  return %broadcast : i32
}

// -----

// CHECK-LABEL: @thread_id_div_subgroup_size
func.func @thread_id_div_subgroup_size() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %warp_id = arith.divui %tid, %c64 : index
  return %warp_id : index
}

// -----

// CHECK-LABEL: @thread_id_div_half_subgroup
func.func @thread_id_div_half_subgroup() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c32 = arith.constant 32 : index
  // CHECK: arith.divui
  // CHECK-NOT: wave.uniform
  %half_warp = arith.divui %tid, %c32 : index
  return %half_warp : index
}

// -----

// CHECK-LABEL: @thread_id_mul_then_div
func.func @thread_id_mul_then_div() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c2 = arith.constant 2 : index
  // CHECK: arith.muli
  // CHECK-NOT: wave.uniform
  %doubled = arith.muli %tid, %c2 overflow<nsw> : index
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui
  // CHECK-NOT: wave.uniform
  %result = arith.divui %doubled, %c64 : index
  return %result : index
}

// -----

// CHECK-LABEL: @thread_id_div_not_divisible
func.func @thread_id_div_not_divisible() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c48 = arith.constant 48 : index
  // CHECK: arith.divui
  // CHECK-NOT: wave.uniform
  %result = arith.divui %tid, %c48 : index
  return %result : index
}

// -----

// CHECK-LABEL: @lane_id_div_subgroup_size
func.func @lane_id_div_subgroup_size() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.lane_id
  // CHECK-NOT: wave.uniform
  %lid = gpu.lane_id
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %result = arith.divui %lid, %c64 : index
  return %result : index
}

// -----

// CHECK-LABEL: @thread_id_div_larger_divisor
func.func @thread_id_div_larger_divisor() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c128 = arith.constant 128 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %warp_id = arith.divui %tid, %c128 : index
  return %warp_id : index
}
