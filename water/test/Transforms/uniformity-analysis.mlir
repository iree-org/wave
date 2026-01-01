// RUN: water-opt %s --water-test-uniformity-analysis | FileCheck %s

// CHECK-LABEL: @constant_uniform
func.func @constant_uniform() -> i32 {
  // CHECK: arith.constant {wave.uniform}
  %c = arith.constant 42 : i32
  return %c : i32
}

// CHECK-LABEL: @thread_id_x_divergent
func.func @thread_id_x_divergent() -> index {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  return %tid : index
}

// CHECK-LABEL: @thread_id_y_uniform
func.func @thread_id_y_uniform() -> index {
  // CHECK: gpu.thread_id y {wave.uniform}
  %tid = gpu.thread_id y
  return %tid : index
}

// CHECK-LABEL: @thread_id_z_uniform
func.func @thread_id_z_uniform() -> index {
  // CHECK: gpu.thread_id z {wave.uniform}
  %tid = gpu.thread_id z
  return %tid : index
}

// CHECK-LABEL: @lane_id_divergent
func.func @lane_id_divergent() -> index {
  // CHECK: gpu.lane_id
  // CHECK-NOT: wave.uniform
  %lid = gpu.lane_id
  return %lid : index
}

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

// CHECK-LABEL: @uniform_propagation
func.func @uniform_propagation() -> index {
  // CHECK: gpu.thread_id y {wave.uniform}
  %tid = gpu.thread_id y
  // CHECK: arith.addi {{.*}} {wave.uniform}
  %result = arith.addi %tid, %tid : index
  return %result : index
}

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

// CHECK-LABEL: @thread_id_shl
func.func @thread_id_shl() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c1 = arith.constant 1 : index
  // CHECK: arith.shli
  // CHECK-NOT: wave.uniform
  %doubled = arith.shli %tid, %c1 overflow<nsw> : index
  return %doubled : index
}

// CHECK-LABEL: @thread_id_shru_uniform
func.func @thread_id_shru_uniform() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c6 = arith.constant 6 : index
  // CHECK: arith.shrui {{.*}} {wave.uniform}
  %warp_id = arith.shrui %tid, %c6 : index
  return %warp_id : index
}

// CHECK-LABEL: @thread_id_shru_subgroup_linear
func.func @thread_id_shru_subgroup_linear() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c1 = arith.constant 1 : index
  // CHECK: arith.shrui
  // CHECK-NOT: wave.uniform
  %half = arith.shrui %tid, %c1 : index
  return %half : index
}

// CHECK-LABEL: @thread_id_shl_then_shru
func.func @thread_id_shl_then_shru() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c1 = arith.constant 1 : index
  // CHECK: arith.shli
  // CHECK-NOT: wave.uniform
  %doubled = arith.shli %tid, %c1 overflow<nsw> : index
  // CHECK: arith.constant {wave.uniform}
  %c7 = arith.constant 7 : index
  // CHECK: arith.shrui {{.*}} {wave.uniform}
  %result = arith.shrui %doubled, %c7 : index
  return %result : index
}

// CHECK-LABEL: @thread_id_and_uniform
func.func @thread_id_and_uniform() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %mask = arith.constant -64 : index
  // CHECK: arith.andi {{.*}} {wave.uniform}
  %result = arith.andi %tid, %mask : index
  return %result : index
}

// CHECK-LABEL: @thread_id_and_divergent
func.func @thread_id_and_divergent() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %mask = arith.constant 31 : index
  // CHECK: arith.andi
  // CHECK-NOT: wave.uniform
  %result = arith.andi %tid, %mask : index
  return %result : index
}

// CHECK-LABEL: @index_cast_preserves_uniform
func.func @index_cast_preserves_uniform() -> i32 attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %warp_id = arith.divui %tid, %c64 : index
  // CHECK: arith.index_cast {{.*}} {wave.uniform}
  %result = arith.index_cast %warp_id : index to i32
  return %result : i32
}

// CHECK-LABEL: @trunci_preserves_subgroup_linear
func.func @trunci_preserves_subgroup_linear() -> i32 attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_i64 = arith.index_cast %tid : index to i64
  // CHECK: arith.trunci
  // CHECK-NOT: wave.uniform
  %tid_i32 = arith.trunci %tid_i64 : i64 to i32
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : i32
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %result = arith.divui %tid_i32, %c64 : i32
  return %result : i32
}

// CHECK-LABEL: @extui_preserves_subgroup_linear
func.func @extui_preserves_subgroup_linear() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_i32 = arith.index_cast %tid : index to i32
  // CHECK: arith.extui
  // CHECK-NOT: wave.uniform
  %tid_i64 = arith.extui %tid_i32 : i32 to i64
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_idx = arith.index_cast %tid_i64 : i64 to index
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %result = arith.divui %tid_idx, %c64 : index
  return %result : index
}

// CHECK-LABEL: @extsi_preserves_subgroup_linear
func.func @extsi_preserves_subgroup_linear() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_i32 = arith.index_cast %tid : index to i32
  // CHECK: arith.extsi
  // CHECK-NOT: wave.uniform
  %tid_i64 = arith.extsi %tid_i32 : i32 to i64
  // CHECK: arith.index_cast
  // CHECK-NOT: wave.uniform
  %tid_idx = arith.index_cast %tid_i64 : i64 to index
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.divui {{.*}} {wave.uniform}
  %result = arith.divui %tid_idx, %c64 : index
  return %result : index
}

// CHECK-LABEL: @loop_uniform_bounds
func.func @loop_uniform_bounds() -> index {
  // CHECK: arith.constant {wave.uniform}
  %c0 = arith.constant 0 : index
  // CHECK: arith.constant {wave.uniform}
  %c10 = arith.constant 10 : index
  // CHECK: arith.constant {wave.uniform}
  %c1 = arith.constant 1 : index
  // CHECK: arith.constant {wave.uniform}
  %c100 = arith.constant 100 : index
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %c100) -> index {
    // CHECK: arith.addi {{.*}} {wave.uniform}
    %sum = arith.addi %i, %arg : index
    scf.yield %sum : index
  }
  return %result : index
}

// CHECK-LABEL: @loop_divergent_bounds
func.func @loop_divergent_bounds() -> index {
  // CHECK: gpu.thread_id x
  // CHECK-NOT: wave.uniform
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c10 = arith.constant 10 : index
  // CHECK: arith.constant {wave.uniform}
  %c1 = arith.constant 1 : index
  // CHECK: arith.constant {wave.uniform}
  %c0 = arith.constant 0 : index
  // CHECK: scf.for
  %result = scf.for %i = %tid to %c10 step %c1 iter_args(%arg = %c0) -> index {
    // CHECK: arith.addi
    // CHECK-NOT: wave.uniform
    %sum = arith.addi %i, %arg : index
    scf.yield %sum : index
  }
  return %result : index
}

gpu.module @test_module1 {
  // CHECK-LABEL: gpu.func @kernel_args_uniform
  gpu.func @kernel_args_uniform(%arg0: index, %arg1: index) kernel {
    // Kernel arguments should be marked as uniform.
    // CHECK: arith.addi {{.*}} {wave.uniform}
    %sum = arith.addi %arg0, %arg1 : index
    gpu.return
  }
}

gpu.module @test_module2 {
  // CHECK-LABEL: gpu.func @kernel_arg_propagation
  gpu.func @kernel_arg_propagation(%arg0: index) kernel attributes {subgroup_size = 64 : i64} {
    // CHECK: gpu.thread_id x
    // CHECK-NOT: wave.uniform
    %tid = gpu.thread_id x
    // CHECK: arith.addi
    // CHECK-NOT: wave.uniform
    %result = arith.addi %tid, %arg0 : index
    gpu.return
  }
}

gpu.module @test_module3 {
  // CHECK-LABEL: gpu.func @kernel_arg_only_uniform_ops
  gpu.func @kernel_arg_only_uniform_ops(%arg0: index, %arg1: index) kernel {
    // CHECK: arith.addi {{.*}} {wave.uniform}
    %sum = arith.addi %arg0, %arg1 : index
    // CHECK: arith.constant {wave.uniform}
    %c10 = arith.constant 10 : index
    // CHECK: arith.muli {{.*}} {wave.uniform}
    %result = arith.muli %sum, %c10 : index
    gpu.return
  }
}

// CHECK-LABEL: @thread_id_rem_subgroup_size
func.func @thread_id_rem_subgroup_size() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x {wave.subgroup_linear = 64 : i64}
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c64 = arith.constant 64 : index
  // CHECK: arith.remui {{.*}} {wave.subgroup_linear = 64 : i64}
  %remainder = arith.remui %tid, %c64 : index
  return %remainder : index
}

// CHECK-LABEL: @thread_id_rem_double_subgroup
func.func @thread_id_rem_double_subgroup() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x {wave.subgroup_linear = 64 : i64}
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c128 = arith.constant 128 : index
  // CHECK: arith.remui {{.*}} {wave.subgroup_linear = 64 : i64}
  %remainder = arith.remui %tid, %c128 : index
  return %remainder : index
}

// CHECK-LABEL: @thread_id_rem_not_divisible
func.func @thread_id_rem_not_divisible() -> index attributes {subgroup_size = 64 : i64} {
  // CHECK: gpu.thread_id x {wave.subgroup_linear = 64 : i64}
  %tid = gpu.thread_id x
  // CHECK: arith.constant {wave.uniform}
  %c48 = arith.constant 48 : index
  // CHECK: arith.remui
  // CHECK-NOT: wave.uniform
  // CHECK-NOT: wave.subgroup_linear
  %remainder = arith.remui %tid, %c48 : index
  return %remainder : index
}
