// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-control-flow --mlir-print-local-scope --split-input-file --verify-diagnostics | FileCheck %s

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @lower_iterate() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 64, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<64xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<64xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @lower_iterate
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     %{{.*}} = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (!wave.tensor<[@M] of f32, <shared>>) {
    // CHECK:       scf.yield %{{.*}} : !wave.tensor<[@M] of f32, <shared>>
    // CHECK:     } {iterator = #wave.symbol<"K">}
    %result = wave.iterate @K iter_args(%0) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @lower_iterate_with_operations() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 32, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<32xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @lower_iterate_with_operations
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     %{{.*}} = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (!wave.tensor<[@M] of f32, <shared>>) {
    // CHECK:       %{{.*}} = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    // CHECK:       scf.yield %{{.*}} : !wave.tensor<[@M] of f32, <shared>>
    // CHECK:     } {iterator = #wave.symbol<"K">}
    %result = wave.iterate @K iter_args(%0) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      %alloc_new = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @lower_iterate_with_iter_args() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, M = 32, BLOCK_K = 16}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = memref.alloc() : memref<32xf32, #gpu.address_space<workgroup>>
    %0 = builtin.unrealized_conversion_cast %alloc : memref<32xf32, #gpu.address_space<workgroup>> to !wave.tensor<[@M] of f32, <shared>>
    %cst = arith.constant 0.0 : f32
    %1 = wave.register %cst : vector<4xf32>

    // CHECK-LABEL: func.func @lower_iterate_with_iter_args
    // CHECK-NOT: wave.iterate
    // CHECK:     %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!wave.tensor<[@M] of f32, <shared>>, vector<4xf32>) {
    // CHECK:       scf.yield %{{.*}}, %{{.*}} : !wave.tensor<[@M] of f32, <shared>>, vector<4xf32>
    // CHECK:     } {iterator = #wave.symbol<"K">}
    %result:2 = wave.iterate @K iter_args(%0, %1) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>, %arg1: vector<4xf32>):
      wave.yield %arg0, %arg1 : !wave.tensor<[@M] of f32, <shared>>, vector<4xf32>
    } : (!wave.tensor<[@M] of f32, <shared>>, vector<4xf32>) -> (!wave.tensor<[@M] of f32, <shared>>, vector<4xf32>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @lower_iterate_allocate() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 128, BLOCK_K = 32, M = 64}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
      : !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @lower_iterate_allocate
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     %{{.*}} = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (!wave.tensor<[@M] of f32, <shared>>) {
    // CHECK:       scf.yield %{{.*}} : !wave.tensor<[@M] of f32, <shared>>
    // CHECK:     } {iterator = #wave.symbol<"K">}
    %result = wave.iterate @K iter_args(%alloc) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}

// -----

module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>} {
  func.func @lower_iterate_with_allocate_inside() attributes {
    wave.hyperparameters = #wave.hyperparameters<{K = 64, BLOCK_K = 16, M = 32}>,
    wave.constraints = [
      #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
    ]
  } {
    %alloc = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
      : !wave.tensor<[@M] of f32, <shared>>

    // CHECK-LABEL: func.func @lower_iterate_with_allocate_inside
    // CHECK-NOT: wave.iterate
    // CHECK:     %[[LB:.*]] = arith.constant 0 : index
    // CHECK:     %[[UB:.*]] = arith.constant 4 : index
    // CHECK:     %[[STEP:.*]] = arith.constant 1 : index
    // CHECK:     %{{.*}} = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%{{.*}} = %{{.*}}) -> (!wave.tensor<[@M] of f32, <shared>>) {
    // CHECK:       %{{.*}} = wave.allocate {distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)>} : <[@M] of f32, <shared>>
    // CHECK:       scf.yield %{{.*}} : !wave.tensor<[@M] of f32, <shared>>
    // CHECK:     } {iterator = #wave.symbol<"K">}
    %result = wave.iterate @K iter_args(%alloc) {
    ^bb0(%arg0: !wave.tensor<[@M] of f32, <shared>>):
      %temp = wave.allocate { distributed_shape = #wave.expr_list<[#wave.symbol<"M">] -> (M)> }
        : !wave.tensor<[@M] of f32, <shared>>
      wave.yield %arg0 : !wave.tensor<[@M] of f32, <shared>>
    } : (!wave.tensor<[@M] of f32, <shared>>) -> (!wave.tensor<[@M] of f32, <shared>>)

    return
  }
}