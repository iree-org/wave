// RUN: water-opt %s | FileCheck %s

#hyperparams = #wave.hyperparameters<{M = 1024, N = 1024, K = 1024, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128, DEVICE_M = 512, DEVICE_N = 512, DEVICE_K = 512}>

// CHECK-LABEL: @test_hw1
// CHECK: #wave.hardware_constraint<threads_per_wave = 64>
func.func private @test_hw1() attributes { wave.constraints = [#wave.hardware_constraint<threads_per_wave = 64>] }

// CHECK-LABEL: @test_hw2
// CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [1, 1, 1], mma_type = <f32_16x16x16_f16>, vector_shapes = {K = 64 : i64, M = 1 : i64, N = 1 : i64}>
#hw_constraint2 = #wave.hardware_constraint<threads_per_wave = 64,
                                            waves_per_block = [1, 1, 1],
                                            mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                            vector_shapes = {M = 1, N = 1, K = 64},
                                            max_bits_per_load = 128>
func.func private @test_hw2() attributes { wave.hyperparameters = #wave.hyperparameters<{M = 1024, N = 1024, K = 1024}>,
                                           wave.constraints = [#hw_constraint2] }


// CHECK-LABEL: @test_wg1
// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wg_constraint1 = #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wg_hyperparams1 = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
func.func private @test_wg1() attributes { wave.hyperparameters = #wg_hyperparams1, wave.constraints = [#wg_constraint1] }

// CHECK-LABEL: @test_wg2
// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wg_constraint2 = #wave.workgroup_constraint<dim = <"M">,
                                             tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>,
                                             workgroup_dim = <x>,
                                             primary = true>
func.func private @test_wg2() attributes { wave.hyperparameters = #wg_hyperparams1, wave.constraints = [#wg_constraint2] }

// CHECK-LABEL: @test_wg3
// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
// CHECK: #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
#wg_constraint3 = #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
#wg_hyperparams3 = #wave.hyperparameters<{M = 1024,N = 1024, BLOCK_M = 128, BLOCK_N = 128}>
func.func private @test_wg3() attributes { wave.hyperparameters = #wg_hyperparams3, wave.constraints = [#wg_constraint2, #wg_constraint3] }

// CHECK-LABEL: @test_wg4
// CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>,
// CHECKL #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <x>, primary = false>
#wg_constraint4 = #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <x>, primary=false>
func.func private @test_wg4() attributes { wave.hyperparameters = #wg_hyperparams3, wave.constraints = [#wg_constraint2, #wg_constraint4] }

// CHECK-LABEL: @test_tiling
// CHECK: #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
#tl_constraint = #wave.tiling_constraint<dim = <"K">, tile_size = <[#wave.symbol<"BLOCK_K">] -> (BLOCK_K)>>
#tl_hyperparams = #wave.hyperparameters<{K = 1024, BLOCK_K = 128}>
func.func private @test_tiling() attributes { wave.hyperparameters = #tl_hyperparams, wave.constraints = [#tl_constraint] }

// CHECK-LABEL: @test_wave1
// CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#wv_constraint1 = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#wv_hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
func.func private @test_wave1() attributes { wave.hyperparameters = #wv_hyperparams, wave.constraints = [#wg_constraint1, #wv_constraint1] }

// CHECK-LABEL: @test_device
// CHECK: #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"DEVICE_M">] -> (DEVICE_M)>, device_dim = 0>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"DEVICE_M">] -> (DEVICE_M)>, device_dim = 0>
#dv_hyperparams = #wave.hyperparameters<{M = 1024, DEVICE_M = 512}>
func.func private @test_device() attributes { wave.hyperparameters = #dv_hyperparams, wave.constraints = [#dv_constraint] }
