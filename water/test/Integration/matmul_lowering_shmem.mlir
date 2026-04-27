// RUN: water-opt %s --water-middle-end-lowering | FileCheck %s

// This test is a LIT counterpart to $WAVE_DIR/tests/kernel/wave/water_e2e_test.py
// with the `minimize_shared_allocs` option on and was obtained by printing
// the MLIR produced by `emit_wave_dialect` before it is passed into
// `override_mlir`.
// It only checks that wave-related dialect have been successfully lowered: the
// dialects are missing and there are no errors meaning the exit code 0. Unit
// tests cover individual transformations and verifiers.

// CHECK-NOT: wave.
// CHECK-NOT: normalform

#idxmap = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.index_symbol<GPR_NUM>, #wave.symbol<"BLOCK_M">] -> (((T0 mod 64) floordiv 32) * 4 + ((GPR_NUM floordiv 4) mod 4) * 8 + WG0 * BLOCK_M + (BLOCK_M floordiv 2) * (T0 floordiv 64) + GPR_NUM mod 4, 16, 32)>
#idxmap1 = #wave.index_mapping<[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * (BLOCK_N floordiv 2) + WG1 * BLOCK_N + T0 mod 32, 1, 1)>
#idxmap2 = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4, 4, 1)>
#idxmap3 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + (BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 32, 1, 1)>
#idxmap4 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> ((T0 mod 4) * 8 + _Iter_K * BLOCK_K, 8, 1)>
#idxmap5 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_M">] -> (WG0 * BLOCK_M + (T1 * 32 + T0 floordiv 4) mod 64, 1, 1)>
#idxmap6 = #wave.index_mapping<[#wave.index_symbol<T0>] -> ((T0 mod 4) * 8, 8, 1)>
#idxmap7 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.index_symbol<T1>] -> ((T1 * 32 + T0 floordiv 4) mod 64, 1, 1)>
#idxmap8 = #wave.index_mapping<[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (WG1 * BLOCK_N + (T1 * 32 + T0 floordiv 4) mod 64, 1, 1)>
#idxmap9 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.index_symbol<T1>, #wave.symbol<"BLOCK_N">] -> (T1 * (BLOCK_N floordiv 2) + T0 mod 32, 1, 1)>
#idxmap10 = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + 8, 4, 1)>
#idxmap11 = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + 16, 4, 1)>
#idxmap12 = #wave.index_mapping<[#wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + 24, 4, 1)>
#idxmap13 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> ((BLOCK_M floordiv 2) * (T0 floordiv 64) + T0 mod 32, 1, 1)>
#idxmap14 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 32) * 4 + _Iter_K * BLOCK_K, 4, 1)>
#idxmap15 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 32) * 4 + _Iter_K * BLOCK_K + 8, 4, 1)>
#idxmap16 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 32) * 4 + _Iter_K * BLOCK_K + 16, 4, 1)>
#idxmap17 = #wave.index_mapping<[#wave.index_symbol<T0>, #wave.iter<"K">, #wave.symbol<"BLOCK_K">] -> (((T0 mod 64) floordiv 32) * 4 + _Iter_K * BLOCK_K + 24, 4, 1)>
#idxmap18 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64, 1, 1)>
#idxmap19 = #wave.index_mapping<[#wave.index_symbol<WG1>, #wave.index_symbol<T0>, #wave.index_symbol<T1>] -> (T1 * 32 + WG1 * 64 + T0 mod 32, 1, 1)>
#idxmap20 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 1, 1, 1)>
#idxmap21 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 2, 1, 1)>
#idxmap22 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 3, 1, 1)>
#idxmap23 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 8, 1, 1)>
#idxmap24 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 9, 1, 1)>
#idxmap25 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 10, 1, 1)>
#idxmap26 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 11, 1, 1)>
#idxmap27 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 16, 1, 1)>
#idxmap28 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 17, 1, 1)>
#idxmap29 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 18, 1, 1)>
#idxmap30 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 19, 1, 1)>
#idxmap31 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 24, 1, 1)>
#idxmap32 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 25, 1, 1)>
#idxmap33 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 26, 1, 1)>
#idxmap34 = #wave.index_mapping<[#wave.index_symbol<WG0>, #wave.index_symbol<T0>] -> (((T0 mod 64) floordiv 32) * 4 + (T0 floordiv 64) * 32 + WG0 * 64 + 27, 1, 1)>
module {
  func.func @kernel(%arg0: !wave.tensor<[@M, @K] of f16, <global>>, %arg1: !wave.tensor<[@N, @K] of f16, <global>>, %arg2: !wave.tensor<[@M, @N] of f32, <global>>) attributes {wave.constraints = [#wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>, #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>, #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>, #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>, #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N floordiv 2)>>, #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1], mma_type = <f32_32x32x8_f16>>], wave.hyperparameters = #wave.hyperparameters<@"$GLOBAL_MEMORY_UNITS" = 4 : i64, @"$GLOBAL_TO_SHARED_DELAY" = 1 : i64, @"$MMA_DELAY" = 1 : i64, @"$MMA_UNITS" = 4 : i64, @"$READ_GLOBAL_DELAY" = 2 : i64, @"$READ_SHARED_DELAY" = 1 : i64, @"$SHARED_MEMORY_UNITS" = 4 : i64, @"$SHUFFLE_DELAY" = 1 : i64, @"$SHUFFLE_UNITS" = 2 : i64, @"$VALU_DELAY" = 1 : i64, @"$VALU_UNITS" = 2 : i64, @"$WRITE_GLOBAL_DELAY" = 2 : i64, @"$WRITE_SHARED_DELAY" = 1 : i64, @BLOCK_K = 32 : i64, @BLOCK_M = 64 : i64, @BLOCK_N = 64 : i64, @K = 640 : i64, @M = 1024 : i64, @N = 5120 : i64>} {
    %0 = wave.allocate {distributed_shape = #wave.expr_list<[] -> (9216)>} : !wave.tensor<[@N, @K, @M] of i8, <shared>>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = wave.register %cst index [#wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>] : !wave.tensor<[@M, @N] of f32, <register>>
    %2 = wave.allocate in %0 : !wave.tensor<[@N, @K, @M] of i8, <shared>> index [#wave.symbol_mapping<@K = #idxmap2, @N = #idxmap1>] {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_K">, #wave.symbol<"BLOCK_N">] -> (BLOCK_N, BLOCK_K)>, offset = 0 : i64, padding = 4 : i64} : !wave.tensor<[@N, @K] of f16, <shared>>
    %3 = wave.allocate in %0 : !wave.tensor<[@N, @K, @M] of i8, <shared>> index [#wave.symbol_mapping<@K = #idxmap2, @M = #idxmap3>] {distributed_shape = #wave.expr_list<[#wave.symbol<"BLOCK_K">, #wave.symbol<"BLOCK_M">] -> (BLOCK_M, BLOCK_K)>, offset = 4608 : i64, padding = 4 : i64} : !wave.tensor<[@M, @K] of f16, <shared>>
    %4 = wave.iterate @K iter_args(%1) {
    ^bb0(%arg3: !wave.tensor<[@M, @N] of f32, <register>>):
      %21 = wave.read %arg0 index [#wave.symbol_mapping<@K = #idxmap4, @M = #idxmap5>] {elements_per_thread = 8 : i64} : (!wave.tensor<[@M, @K] of f16, <global>>) -> !wave.tensor<[@M, @K] of f16, <register>>
      amdgpu.lds_barrier
      wave.write %21, %3 index [#wave.symbol_mapping<@K = #idxmap6, @M = #idxmap7>] {elements_per_thread = 8 : i64} : !wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@M, @K] of f16, <shared>>
      %22 = wave.read %arg1 index [#wave.symbol_mapping<@K = #idxmap4, @N = #idxmap8>] {elements_per_thread = 8 : i64} : (!wave.tensor<[@N, @K] of f16, <global>>) -> !wave.tensor<[@N, @K] of f16, <register>>
      wave.write %22, %2 index [#wave.symbol_mapping<@K = #idxmap6, @N = #idxmap7>] {elements_per_thread = 8 : i64} : !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <shared>>
      amdgpu.lds_barrier
      %23 = wave.read %2 index [#wave.symbol_mapping<@K = #idxmap2, @N = #idxmap9>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@N, @K] of f16, <shared>>) -> !wave.tensor<[@N, @K] of f16, <register>>
      %24 = wave.read %2 index [#wave.symbol_mapping<@K = #idxmap10, @N = #idxmap9>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@N, @K] of f16, <shared>>) -> !wave.tensor<[@N, @K] of f16, <register>>
      %25 = wave.read %2 index [#wave.symbol_mapping<@K = #idxmap11, @N = #idxmap9>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@N, @K] of f16, <shared>>) -> !wave.tensor<[@N, @K] of f16, <register>>
      %26 = wave.read %2 index [#wave.symbol_mapping<@K = #idxmap12, @N = #idxmap9>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@N, @K] of f16, <shared>>) -> !wave.tensor<[@N, @K] of f16, <register>>
      %27 = wave.read %3 index [#wave.symbol_mapping<@K = #idxmap2, @M = #idxmap13>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@M, @K] of f16, <shared>>) -> !wave.tensor<[@M, @K] of f16, <register>>
      %28 = wave.read %3 index [#wave.symbol_mapping<@K = #idxmap10, @M = #idxmap13>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@M, @K] of f16, <shared>>) -> !wave.tensor<[@M, @K] of f16, <register>>
      %29 = wave.read %3 index [#wave.symbol_mapping<@K = #idxmap11, @M = #idxmap13>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@M, @K] of f16, <shared>>) -> !wave.tensor<[@M, @K] of f16, <register>>
      %30 = wave.read %3 index [#wave.symbol_mapping<@K = #idxmap12, @M = #idxmap13>] {elements_per_thread = 4 : i64} : (!wave.tensor<[@M, @K] of f16, <shared>>) -> !wave.tensor<[@M, @K] of f16, <register>>
      %31 = wave.mma %27, %23, %arg3 index [#wave.symbol_mapping<@K = #idxmap14, @M = #idxmap3>, #wave.symbol_mapping<@K = #idxmap14, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>] {kind = #wave.mma_kind<f32_32x32x8_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
      %32 = wave.mma %28, %24, %31 index [#wave.symbol_mapping<@K = #idxmap15, @M = #idxmap3>, #wave.symbol_mapping<@K = #idxmap15, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>] {kind = #wave.mma_kind<f32_32x32x8_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
      %33 = wave.mma %29, %25, %32 index [#wave.symbol_mapping<@K = #idxmap16, @M = #idxmap3>, #wave.symbol_mapping<@K = #idxmap16, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>] {kind = #wave.mma_kind<f32_32x32x8_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
      %34 = wave.mma %30, %26, %33 index [#wave.symbol_mapping<@K = #idxmap17, @M = #idxmap3>, #wave.symbol_mapping<@K = #idxmap17, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>, #wave.symbol_mapping<@M = #idxmap, @N = #idxmap1>] {kind = #wave.mma_kind<f32_32x32x8_f16>} : (!wave.tensor<[@M, @K] of f16, <register>>, !wave.tensor<[@N, @K] of f16, <register>>, !wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
      wave.yield %34 : !wave.tensor<[@M, @N] of f32, <register>>
    } : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    %5 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap18, @N = #idxmap19>], offset = #wave.expr_list<[] -> (0)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %5, %arg2 index [#wave.symbol_mapping<@M = #idxmap18, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %6 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap20, @N = #idxmap19>], offset = #wave.expr_list<[] -> (1)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %6, %arg2 index [#wave.symbol_mapping<@M = #idxmap20, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %7 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap21, @N = #idxmap19>], offset = #wave.expr_list<[] -> (2)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %7, %arg2 index [#wave.symbol_mapping<@M = #idxmap21, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %8 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap22, @N = #idxmap19>], offset = #wave.expr_list<[] -> (3)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %8, %arg2 index [#wave.symbol_mapping<@M = #idxmap22, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %9 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap23, @N = #idxmap19>], offset = #wave.expr_list<[] -> (4)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %9, %arg2 index [#wave.symbol_mapping<@M = #idxmap23, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %10 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap24, @N = #idxmap19>], offset = #wave.expr_list<[] -> (5)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %10, %arg2 index [#wave.symbol_mapping<@M = #idxmap24, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %11 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap25, @N = #idxmap19>], offset = #wave.expr_list<[] -> (6)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %11, %arg2 index [#wave.symbol_mapping<@M = #idxmap25, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %12 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap26, @N = #idxmap19>], offset = #wave.expr_list<[] -> (7)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %12, %arg2 index [#wave.symbol_mapping<@M = #idxmap26, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %13 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap27, @N = #idxmap19>], offset = #wave.expr_list<[] -> (8)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %13, %arg2 index [#wave.symbol_mapping<@M = #idxmap27, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %14 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap28, @N = #idxmap19>], offset = #wave.expr_list<[] -> (9)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %14, %arg2 index [#wave.symbol_mapping<@M = #idxmap28, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %15 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap29, @N = #idxmap19>], offset = #wave.expr_list<[] -> (10)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %15, %arg2 index [#wave.symbol_mapping<@M = #idxmap29, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %16 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap30, @N = #idxmap19>], offset = #wave.expr_list<[] -> (11)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %16, %arg2 index [#wave.symbol_mapping<@M = #idxmap30, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %17 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap31, @N = #idxmap19>], offset = #wave.expr_list<[] -> (12)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %17, %arg2 index [#wave.symbol_mapping<@M = #idxmap31, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %18 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap32, @N = #idxmap19>], offset = #wave.expr_list<[] -> (13)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %18, %arg2 index [#wave.symbol_mapping<@M = #idxmap32, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %19 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap33, @N = #idxmap19>], offset = #wave.expr_list<[] -> (14)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %19, %arg2 index [#wave.symbol_mapping<@M = #idxmap33, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    %20 = wave.extract_slice %4 {index = [#wave.symbol_mapping<@M = #idxmap34, @N = #idxmap19>], offset = #wave.expr_list<[] -> (15)>, size = #wave.expr_list<[] -> (1)>, stride = #wave.expr_list<[] -> (1)>} : (!wave.tensor<[@M, @N] of f32, <register>>) -> !wave.tensor<[@M, @N] of f32, <register>>
    wave.write %20, %arg2 index [#wave.symbol_mapping<@M = #idxmap34, @N = #idxmap19>] {elements_per_thread = 1 : i64} : !wave.tensor<[@M, @N] of f32, <register>>, !wave.tensor<[@M, @N] of f32, <global>>
    return
  }
}
