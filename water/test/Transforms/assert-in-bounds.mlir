// RUN: water-opt %s --water-assert-in-bounds --split-input-file | FileCheck %s --check-prefixes=CHECK,PERDIM,INPLC,PERDIM-INPLC
// RUN: water-opt %s --water-assert-in-bounds='check-each-dim=false' --split-input-file | FileCheck %s --check-prefixes=CHECK,COMPOUND,INPLC
// RUN: water-opt %s --water-assert-in-bounds='include-vector-load-store=true' --split-input-file | FileCheck %s --check-prefixes=CHECK,VECTOR,INPLC
// RUN: water-opt %s --water-assert-in-bounds='create-speculative-funcs=true' --split-input-file --verify-diagnostics | FileCheck %s --check-prefixes=CHECK,SPEC,PERDIM

// Prefix key:
//   CHECK - for all cases
//   INPLC - in-place assertions only
//   PERDIM - only per-dimension checks
//   COMPOUND - compound checks across all dimensions
//   SPEC - speculative functions only
//   PERDIM-INPLC - in-place per-dimension checks only (speculative will fail)


// INPLC-LABEL: @all_static
// SPEC-LABEL: @__speculative_in_bounds_check_all_static
func.func @all_static(%memref: memref<2x2xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // PERDIM:         %[[TRUE1:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[TRUE:.+]] = arith.constant true
  // COMPOUND:       cf.assert %[[TRUE]], "memref access out of bounds"
  // INPLC:          memref.load
  // SPEC-NOT:       memref.load
  %2 = memref.load %memref[%0, %1] : memref<2x2xf32>
  return %2 : f32
}
// SPEC:       @all_static
// SPEC:       call @__speculative_in_bounds_check_all_static
// SPEC:       memref.load

// -----

// INPLC-LABEL: @shape_static_one_index_dynamic
// SPEC-LABEL:  @__speculative_in_bounds_check_shape_static_one_index_dynamic
// CHECK-SAME: (%[[ARG0:.+]]: memref<5x2xf32>, %[[ARG1:.+]]: index)
func.func @shape_static_one_index_dynamic(%memref: memref<5x2xf32>, %i: index) -> f32 {
  // CHECK:          arith.constant 0
  %0 = arith.constant 0 : index
  // CHECK:          %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK:          %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  //
  // CHECK:          %[[TWO:.+]] = arith.constant 2 : index
  // CHECK:          %[[LB:.+]] = arith.cmpi sge, %[[ARG1]], %[[ZERO]]
  // CHECK:          %[[UB:.+]] = arith.cmpi slt, %[[ARG1]], %[[TWO]]
  // CHECK:          %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
  // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       cf.assert %[[BOUND]], "memref access out of bounds"
  // INPLC:          memref.load
  // SPEC-NOT:       memref.load
  %1 = memref.load %memref[%0, %i] : memref<5x2xf32>
  return %1 : f32
}
// SPEC:      @shape_static_one_index_dynamic
// SPEC:      call @__speculative_in_bounds_check_shape_static_one_index_dynamic
// SPEC:      memref.load

// -----

// INPLC-LABEL: @shape_dynamic
// SPEC-LABEL:  @__speculative_in_bounds_check_shape_dynamic
func.func @shape_dynamic(%memref: memref<?x?xf32>) -> f32 {
  // SPEC:           %{{.*}} = arith.constant 0.0
  // CHECK:          %[[INDEX0:.+]] = arith.constant 0
  // CHECK:          %[[INDEX1:.+]] = arith.constant 1
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // CHECK:          %[[ZERO1:.+]] = arith.constant 0 : index
  // CHECK:          %[[ZERO2:.+]] = arith.constant 0 : index
  // CHECK:          %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO2]]
  // Note that folding changed index0 < dim0 into dim0 > index0.
  // CHECK:          %[[UB0:.+]] = arith.cmpi sgt, %[[DIM0]], %[[INDEX0]]
  // PERDIM:         cf.assert %[[UB0]], "memref access out of bounds along dimension 0"
  //
  // CHECK:          %[[ONE1:.+]] = arith.constant 1 : index
  // CHECK:          %[[DIM1:.+]] = memref.dim %{{.*}}, %[[ONE1]]
  // CHECK:          %[[UB1:.+]] = arith.cmpi sgt, %[[DIM1]], %[[INDEX1]]
  // PERDIM:         cf.assert %[[UB1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[COMPOUND:.+]] = arith.andi %[[UB0]], %[[UB1]]
  // COMPOUND:       cf.assert %[[COMPOUND]], "memref access out of bounds"
  //
  // INPLC:          memref.load
  // SPEC-NOT:       memref.load
  %2 = memref.load %memref[%0, %1] : memref<?x?xf32>
  return %2 : f32
}
// SPEC:      @shape_dynamic
// SPEC:      call @__speculative_in_bounds_check_shape_dynamic
// SPEC:      memref.load


// -----

// INPLC-LABEL: @out_of_bounds
// SPEC-LABEL:  @__speculative_in_bounds_check_out_of_bounds
func.func @out_of_bounds(%memref: memref<2x2xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = arith.constant 2 : index
  // Note that even though the access still hits allocated memory, it wraps
  // around the innermost dimension and is therefore considered out of bounds.
  //
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // CHECK:          %[[FALSE:.+]] = arith.constant false
  // PERDIM:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 1"
  // COMPOUND:       cf.assert %[[FALSE]], "memref access out of bounds"
  // INPLC:          memref.load
  // SPEC-NOT:       memref.load
  %2 = memref.load %memref[%0, %1] : memref<2x2xf32>
  return %2 : f32
}
// SPEC:      @out_of_bounds
// SPEC:      call @__speculative_in_bounds_check_out_of_bounds
// SPEC:      memref.load

// -----

// INPLC-LABEL: @zero_d
// SPEC-LABEL: @__speculative_in_bounds_check_zero_d
func.func @zero_d(%memref: memref<f32>) -> f32 {
  // INPLC:    memref.load
  // SPEC-NOT: memref.load
  %0 = memref.load %memref[] : memref<f32>
  return %0 : f32
}
// SPEC:      @zero_d
// SPEC:      call @__speculative_in_bounds_check_zero_d
// SPEC:      memref.load

// -----

// INPLC-LABEL: @vector_load_vector_static
// SPEC-LABEL: @__speculative_in_bounds_check_vector_load_vector_static
func.func @vector_load_vector_static(%memref: memref<2x2xvector<2xf32>>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // PERDIM:         %[[TRUE1:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[TRUE:.+]] = arith.constant true
  // COMPOUND:       cf.assert %[[TRUE]], "memref access out of bounds"
  // INPLC:          vector.load
  // SPEC-NOT:       vector.load
  %2 = vector.load %memref[%0, %1] : memref<2x2xvector<2xf32>>, vector<2xf32>
  return %2 : vector<2xf32>
}
// SPEC:      @vector_load_vector_static
// SPEC:      call @__speculative_in_bounds_check_vector_load_vector_static
// SPEC:      vector.load

// -----

// INPLC-LABEL: @vector_load_scalar_static_1d_in_bounds
// SPEC-LABEL: @__speculative_in_bounds_check_vector_load_scalar_static_1d_in_bounds
func.func @vector_load_scalar_static_1d_in_bounds(%memref: memref<5x9xf32>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  // PERDIM-NOT:     cf.assert
  // COMPOUND-NOT:   cf.assert
  // VECTOR:         %[[TRUE0:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE0]], "memref access out of bounds along dimension 0"
  // VECTOR:         %[[TRUE1:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  // INPLC:          vector.load
  // SPEC-NOT:       vector.load
  %2 = vector.load %memref[%0, %0] : memref<5x9xf32>, vector<2xf32>
  return %2 : vector<2xf32>
}
// SPEC:      @vector_load_scalar_static_1d_in_bounds
// SPEC:      call @__speculative_in_bounds_check_vector_load_scalar_static_1d_in_bounds
// SPEC:      vector.load

// -----

// INPLC-LABEL: @vector_load_scalar_static_1d_out_of_bounds
// SPEC-LABEL: @__speculative_in_bounds_check_vector_load_scalar_static_1d_out_of_bounds
func.func @vector_load_scalar_static_1d_out_of_bounds(%memref: memref<5x9xf32>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 8 : index
  // PERDIM-NOT:     cf.assert
  // COMPOUND-NOT:   cf.assert
  // VECTOR:         %[[TRUE:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // VECTOR:         %[[FALSE:.+]] = arith.constant false
  // VECTOR:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 1"
  // INPLC:          vector.load
  // SPEC-NOT:       vector.load
  %2 = vector.load %memref[%0, %1] : memref<5x9xf32>, vector<2xf32>
  return %2 : vector<2xf32>
}
// SPEC:      @vector_load_scalar_static_1d_out_of_bounds
// SPEC:      call @__speculative_in_bounds_check_vector_load_scalar_static_1d_out_of_bounds
// SPEC:      vector.load

// -----

// From this point, we only check the PERDIM/SPEC scenario as checking dimensions
// separately or together has no incidence on how the control flow is handled.

// INPLC-LABEL: @loop
// SPEC-LABEL: @__speculative_in_bounds_check_loop
func.func @loop(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) -> f32 {
  // SPEC:           %[[ITER_ARG_INIT:.+]] = arith.constant 0.0
  %0 = arith.constant 0.0 : f32
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  // SPEC-SAME:      iter_args(%{{.*}} = %[[ITER_ARG_INIT]])
  %1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %0) -> (f32) {
    // SPEC:           %[[YIELDED:.+]] = arith.constant 0.0
    // PERDIM:         %[[ZERO:.+]] = arith.constant 0 : index
    // PERDIM:         %[[TWO:.+]] = arith.constant 2 : index
    // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[I]], %[[ZERO]] : index
    // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[I]], %[[TWO]] : index
    // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
    // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
    // SPEC:           scf.yield %[[YIELDED]]
    %2 = memref.load %memref[%i] : memref<2xf32>
    %3 = arith.addf %acc, %2 : f32
    scf.yield %3 : f32
  }
  return %1 : f32
}

// -----

// INPLC-LABEL: @loop_store
// SPEC-LABEL: @__speculative_in_bounds_check_loop_store
func.func @loop_store(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    // PERDIM:         %[[ZERO:.+]] = arith.constant 0 : index
    // PERDIM:         %[[TWO:.+]] = arith.constant 2 : index
    // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[I]], %[[ZERO]] : index
    // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[I]], %[[TWO]] : index
    // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
    // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
    %2 = arith.index_cast %i : index to i64
    %3 = arith.sitofp %2 : i64 to f32
    // INPLC:          memref.store
    // SPEC-NOT:       memref.store
    memref.store %3, %memref[%i] : memref<2xf32>
    scf.yield
  }
  return
}
// SPEC:      @loop_store
// SPEC:      call @__speculative_in_bounds_check_loop_store
// SPEC:      memref.store

// -----

// INPLC-LABEL: @if
// SPEC-LABEL: @__speculative_in_bounds_check_if
func.func @if(%memref: memref<2xf32>, %val: f32, %cond: i1){
  // PERDIM:         scf.if
  scf.if %cond {
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %1 = arith.constant 0 : index
    memref.store %val, %memref[%1] : memref<2xf32>
  }
  return
}
// SPEC:      @if
// SPEC:      call @__speculative_in_bounds_check_if
// SPEC:      memref.store

// -----

// INPLC-LABEL: @if_else
func.func @if_else(%memref: memref<2xf32>, %cond: i1) -> f32 {
  // PERDIM:         scf.if
  %0 = scf.if %cond -> f32 {
    // SPEC:           %[[YIELDED_1:.+]] = arith.constant 0.0
    %1 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %2 = memref.load %memref[%1] : memref<2xf32>
    // SPEC:           scf.yield %[[YIELDED_1]]
    scf.yield %2 : f32
  // PERDIM:         else
  } else {
    // SPEC:           %[[YIELDED_2:.+]] = arith.constant 0.0
    %3 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %4 = memref.load %memref[%3] : memref<2xf32>
    // SPEC:           scf.yield %[[YIELDED_2]]
    scf.yield %4 : f32
  }
  return %0 : f32
}

// -----

// INPLC-LABEL: @if_else_store
// SPEC-LABEL: @__speculative_in_bounds_check_if_else_store
func.func @if_else_store(%memref: memref<2xf32>, %cond: i1, %value: f32) {
  // PERDIM:         scf.if
  scf.if %cond {
    %0 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    // INPLC:          memref.store
    // SPEC-NOT:       memref.store
    memref.store %value, %memref[%0] : memref<2xf32>
  } else {
    %1 = arith.constant 2 : index
    // PERDIM:         %[[FALSE:.+]] = arith.constant false
    // PERDIM:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 0"
    // INPLC:          memref.store
    // SPEC-NOT:       memref.store
    memref.store %value, %memref[%1] : memref<2xf32>
  }
  return
}
// SPEC:      @if_else_store
// SPEC:      call @__speculative_in_bounds_check_if_else_store
// SPEC:      memref.store

// -----

// INPLC-LABEL: @nested_for_indep
// SPEC-LABEL: @__speculative_in_bounds_check_nested_for_indep
func.func @nested_for_indep(%memref: memref<?xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    %c1 = arith.constant 1 : index
    %lb1 = arith.addi %i, %c1 : index
    // PERDIM:         scf.for %[[J:.+]] = %{{.*}} to
    scf.for %j = %lb1 to %ub step %step {
      // SPEC:           %{{.*}} = arith.constant 0.0
      %sum = arith.addi %i, %j : index
      %1 = arith.index_cast %sum : index to i64
      %2 = arith.uitofp %1 : i64 to f32
      // PERDIM:         %[[ZERO0:.+]] = arith.constant 0
      // PERDIM:         %[[ZERO1:.+]] = arith.constant 0
      // PERDIM:         %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO1]]
      // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[J]], %[[ZERO0]]
      // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[J]], %[[DIM0]]
      // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
      // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
      // INPLC:          memref.store
      // SPEC-NOT:       memref.store
      memref.store %2, %memref[%j] : memref<?xf32>
    }
  }
  return
}

// -----

// INPLC-LABEL: @nested_for
// SPEC-LABEL: @__speculative_in_bounds_check_nested_for
func.func @nested_for(%memref: memref<?x?xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    %c1 = arith.constant 1 : index
    %lb1 = arith.addi %i, %c1 : index
    // PERDIM:         scf.for %[[J:.+]] = %{{.*}} to
    scf.for %j = %lb1 to %ub step %step iter_args(%secondary = %i) -> index {
      %sum = arith.addi %secondary, %c1 : index
      %1 = arith.index_cast %sum : index to i64
      %2 = arith.uitofp %1 : i64 to f32
      // SPEC:           %[[DUMMY_YIELD:.+]] = arith.constant 0
      // SPEC:           %{{.*}} = arith.constant 0.0
      // PERDIM:         %[[ZERO0:.+]] = arith.constant 0
      // PERDIM:         %[[ZERO1:.+]] = arith.constant 0
      // PERDIM:         %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO1]]
      // PERDIM:         %[[LB0:.+]] = arith.cmpi sge, %[[I]], %[[ZERO0]]
      // PERDIM:         %[[UB0:.+]] = arith.cmpi slt, %[[I]], %[[DIM0]]
      // PERDIM:         %[[BOUND0:.+]] = arith.andi %[[LB0]], %[[UB0]]
      // PERDIM:         cf.assert %[[BOUND0]], "memref access out of bounds along dimension 0"
      //
      // PERDIM:         %[[ONE:.+]] = arith.constant 1
      // PERDIM:         %[[DIM1:.+]] = memref.dim %{{.*}}, %[[ONE]]
      // PERDIM:         %[[LB1:.+]] = arith.cmpi sge, %[[J]], %[[ZERO0]]
      // PERDIM:         %[[UB1:.+]] = arith.cmpi slt, %[[J]], %[[DIM1]]
      // PERDIM:         %[[BOUND1:.+]] = arith.andi %[[LB1]], %[[UB1]]
      // PERDIM:         cf.assert %[[BOUND1]], "memref access out of bounds along dimension 1"
      //
      // INPLC:          memref.store
      // SPEC-NOT:       memref.store
      //
      // SPEC:           scf.yield %[[DUMMY_YIELD]]
      memref.store %2, %memref[%i, %j] : memref<?x?xf32>
      scf.yield %sum : index
    }
  }
  return
}
// SPEC:      @nested_for
// SPEC:      call @__speculative_in_bounds_check_nested_for
// SPEC:      memref.store

// -----

// INPLC-LABEL: @dynamic_bound
func.func @dynamic_bound(%lb: index, %ub: memref<index>, %step: index, %data: memref<?xf32>) {
  // expected-error @below {{in-bounds check generation requires speculating this operation, but it is not speculatable}}
  %0 = memref.load %ub[] : memref<index>
  scf.for %i = %lb to %0 step %step {
    // PERDIM-INPLC:    cf.assert
    memref.load %data[%i] : memref<?xf32>
  }
  return
}

// -----

// VECTOR-LABEL: @oob
func.func @oob(%arg0: memref<256x256xf16, strided<[256, 1], offset: ?>>, %arg1: memref<256x256xf16, strided<[256, 1], offset: ?>>, %arg2: memref<256xf16, strided<[1], offset: ?>>) {
  %c32_i32 = arith.constant 32 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c4_i32 = arith.constant 4 : i32
  %c2_i32 = arith.constant 2 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  // VECTOR:         %[[BLOCK_ID_Y:.+]] = gpu.block_id  y upper_bound 256
  %block_id_y = gpu.block_id  y upper_bound 256
  // VECTOR:         %[[THREAD_ID_X:.+]] = gpu.thread_id  x upper_bound 128
  %thread_id_x = gpu.thread_id  x upper_bound 128
  // VECTOR:         %[[IDX_DIM1:.+]] = affine.apply {{.*}}[%[[THREAD_ID_X]]]
  %0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%thread_id_x]
  // VECTOR:         %[[LOWER_BOUND_CHECK:.+]] = arith.cmpi sge, %[[IDX_DIM1]], %c0
  // VECTOR:         %[[C3:.+]] = arith.constant 3 : index
  // VECTOR:         %[[UPPER_BOUND:.+]] = arith.addi %0, %c3 : index
  // VECTOR:         %[[UPPER_BOUND_CHECK:.+]] = arith.cmpi slt, %[[UPPER_BOUND]], %c256
  // VECTOR:         %[[COND:.+]] = arith.andi %[[LOWER_BOUND_CHECK]], %[[UPPER_BOUND_CHECK]] : i1
  // VECTOR:         cf.assert %[[COND]], "memref access out of bounds along dimension 1"
  %1 = vector.load %arg0[%block_id_y, %0] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
  %2 = vector.load %arg1[%block_id_y, %0] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
  %3 = arith.mulf %1, %2 : vector<4xf16>
  %4 = vector.extract %3[0] : f16 from vector<4xf16>
  %5 = vector.extract %3[1] : f16 from vector<4xf16>
  %6 = arith.addf %4, %5 : f16
  %7 = vector.extract %3[2] : f16 from vector<4xf16>
  %8 = arith.addf %6, %7 : f16
  %9 = vector.extract %3[3] : f16 from vector<4xf16>
  %10 = arith.addf %8, %9 : f16
  %11 = vector.broadcast %10 : f16 to vector<1xf16>
  %shuffleResult, %valid = gpu.shuffle  xor %11, %c1_i32, %c64_i32 : vector<1xf16>
  %12 = arith.addf %11, %shuffleResult : vector<1xf16>
  %shuffleResult_0, %valid_1 = gpu.shuffle  xor %12, %c2_i32, %c64_i32 : vector<1xf16>
  %13 = arith.addf %12, %shuffleResult_0 : vector<1xf16>
  %shuffleResult_2, %valid_3 = gpu.shuffle  xor %13, %c4_i32, %c64_i32 : vector<1xf16>
  %14 = arith.addf %13, %shuffleResult_2 : vector<1xf16>
  %shuffleResult_4, %valid_5 = gpu.shuffle  xor %14, %c8_i32, %c64_i32 : vector<1xf16>
  %15 = arith.addf %14, %shuffleResult_4 : vector<1xf16>
  %shuffleResult_6, %valid_7 = gpu.shuffle  xor %15, %c16_i32, %c64_i32 : vector<1xf16>
  %16 = arith.addf %15, %shuffleResult_6 : vector<1xf16>
  %shuffleResult_8, %valid_9 = gpu.shuffle  xor %16, %c32_i32, %c64_i32 : vector<1xf16>
  %17 = arith.addf %16, %shuffleResult_8 : vector<1xf16>
  vector.store %17, %arg2[%block_id_y] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>
  return
}
