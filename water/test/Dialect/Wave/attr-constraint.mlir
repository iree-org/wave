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

// CHECK-LABEL: @test_wave_divisible_by_2
// CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>
#wv_constraint_div2 = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>
#wv_hyperparams_div2 = #wave.hyperparameters<{M = 1024, BLOCK_M = 64}>
func.func private @test_wave_divisible_by_2() attributes { wave.hyperparameters = #wv_hyperparams_div2, wave.constraints = [#wg_constraint1, #wv_constraint_div2] }

// CHECK-LABEL: @test_wave_multiple_dims
// CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
// CHECK: #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N ceildiv 2)>>
#wv_constraint_m = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#wv_constraint_n = #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N ceildiv 2)>>
#wv_hyperparams_multi = #wave.hyperparameters<{M = 1024, N = 1024, BLOCK_M = 128, BLOCK_N = 64}>
func.func private @test_wave_multiple_dims() attributes { wave.hyperparameters = #wv_hyperparams_multi, wave.constraints = [#wg_constraint2, #wg_constraint3, #wv_constraint_m, #wv_constraint_n] }

// CHECK-LABEL: @test_device
// CHECK: #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"DEVICE_M">] -> (DEVICE_M)>, device_dim = 0>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"DEVICE_M">] -> (DEVICE_M)>, device_dim = 0>
#dv_hyperparams = #wave.hyperparameters<{M = 1024, DEVICE_M = 512}>
func.func private @test_device() attributes { wave.hyperparameters = #dv_hyperparams, wave.constraints = [#dv_constraint] }

// CHECK-LABEL: @test_waves_per_block_match_single_dim
// CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [4, 1, 1]>
#hyperparams_wpb_valid1 = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint_wpb_valid1 = #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wv_constraint_wpb_valid1 = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#hw_constraint_wpb_valid1 = #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [4, 1, 1]>
func.func private @test_waves_per_block_match_single_dim() attributes { wave.hyperparameters = #hyperparams_wpb_valid1, wave.constraints = [#wg_constraint_wpb_valid1, #wv_constraint_wpb_valid1, #hw_constraint_wpb_valid1] }

// CHECK-LABEL: @test_waves_per_block_match_multi_dim
// CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 4, 1]>
#hyperparams_wpb_valid2 = #wave.hyperparameters<{M = 1024, N = 512, BLOCK_M = 128, BLOCK_N = 64}>
#wg_constraint_wpb_valid2_m = #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wg_constraint_wpb_valid2_n = #wave.workgroup_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N)>, workgroup_dim = <y>>
#wv_constraint_wpb_valid2_m = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 2)>>
#wv_constraint_wpb_valid2_n = #wave.wave_constraint<dim = <"N">, tile_size = <[#wave.symbol<"BLOCK_N">] -> (BLOCK_N floordiv 4)>>
#hw_constraint_wpb_valid2 = #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 4, 1]>
func.func private @test_waves_per_block_match_multi_dim() attributes { wave.hyperparameters = #hyperparams_wpb_valid2, wave.constraints = [#wg_constraint_wpb_valid2_m, #wg_constraint_wpb_valid2_n, #wv_constraint_wpb_valid2_m, #wv_constraint_wpb_valid2_n, #hw_constraint_wpb_valid2] }

// CHECK-LABEL: @test_waves_per_block_no_wave_constraints
// CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 1, 1]>
#hyperparams_wpb_valid3 = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint_wpb_valid3 = #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#hw_constraint_wpb_valid3 = #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 1, 1]>
func.func private @test_waves_per_block_no_wave_constraints() attributes { wave.hyperparameters = #hyperparams_wpb_valid3, wave.constraints = [#wg_constraint_wpb_valid3, #hw_constraint_wpb_valid3] }

// CHECK-LABEL: @test_wave_constraints_no_waves_per_block
// CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#hyperparams_wpb_valid4 = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint_wpb_valid4 = #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M)>, workgroup_dim = <x>>
#wv_constraint_wpb_valid4 = #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"BLOCK_M">] -> (BLOCK_M floordiv 4)>>
#hw_constraint_wpb_valid4 = #wave.hardware_constraint<threads_per_wave = 64>
func.func private @test_wave_constraints_no_waves_per_block() attributes { wave.hyperparameters = #hyperparams_wpb_valid4, wave.constraints = [#wg_constraint_wpb_valid4, #wv_constraint_wpb_valid4, #hw_constraint_wpb_valid4] }
