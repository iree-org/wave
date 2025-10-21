// RUN: water-opt %s --allow-unregistered-dialect --water-test-wave-dialect-constructors --split-input-file --verify-diagnostics

// expected-error @below {{waves_per_block (1) does should have the same size as vector_shapes ({M = 1 : i64, N = 64 : i64})}}
#hw_constraint = #wave.hardware_constraint<threads_per_wave = 64,
                                           waves_per_block = [1],
                                           mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                           vector_shapes = {M = 1, N = 64}>
func.func private @test_num_dimensions_mismatch1() attributes { wave.constraints = [#hw_constraint] }

// -----

// expected-error @below {{"M" is not an IntegerAttr: "BLOCK_M"}}
#hw_constraint = #wave.hardware_constraint<threads_per_wave = 64,
                                           waves_per_block = [1, 1],
                                           mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                                           vector_shapes = {M = "BLOCK_M", N = 64}>
func.func private @test_num_dimensions_mismatch2() attributes { wave.constraints = [#hw_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, K = 1024, BLOCK_M = 128, BLOCK_K = 128}>
#wg_constraint1 = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M), workgroup_dim = <x>>
#wg_constraint2 = #wave.workgroup_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K), workgroup_dim = <x>>
// expected-error @below {{the dimension of the workgroup constraint in wg_constraint: #wave.symbol<"K"> should match the dimension of the wave constraint: #wave.symbol<"M">}}
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4),
                                       wg_constraint = #wg_constraint2>
func.func private @test_wrong_wg_attr() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint1, #wg_constraint2, #wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, K = 1024, BLOCK_M = 128, BLOCK_K = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K), workgroup_dim = <x>>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{missing corresponding workgroup constraint for dimension: #wave.symbol<"M">}}
func.func private @test_wrong_wg_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint, #wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{missing corresponding workgroup constraint for dimension: #wave.symbol<"M">}}
func.func private @test_missing_wg_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M), workgroup_dim = <x>, primary=false>
// expected-error @below {{missing primary workgroup constraint for workgroup dimension: #wave.workgroup_dim<x>}}
func.func private @test_no_primary_wg_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{K = 512, M = 1024, BLOCK_K = 64, BLOCK_M = 128}>
#wg_constraint1 = #wave.workgroup_constraint<dim = <"K">, tile_size = [BLOCK_K] -> (BLOCK_K), workgroup_dim = <x>>
#wg_constraint2 = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
// expected-error @below {{workgroup dimension #wave.workgroup_dim<x> has more than one primary workgroup constraint}}
func.func private @test_duplicate_primary_wg_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint1, #wg_constraint2] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, DEVICE_M = 2048}>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
// expected-error @below {{invalid number of devices: 0 for dimension: #wave.symbol<"M">}}
func.func private @test_wrong_device_size() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#dv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 2048}>
#constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M), workgroup_dim = <x>>
// expected-error @below {{invalid number of workgroups: 0 for dimension: #wave.symbol<"M">}}
func.func private @test_wrong_wg_size() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
// expected-error @below {{invalid number of waves: 0 for dimension: #wave.symbol<"M">}}
func.func private @test_wrong_wave_size() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint, #wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 2048}>
#constraint = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
// expected-error @below {{invalid number of tiles: 0 for dimension: #wave.symbol<"M">}}
func.func private @test_wrong_tile_size() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, DEVICE_M = 128}>
#dv_constraint1 = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
#dv_constraint2 = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M floordiv 4), device_dim = 0>
// expected-error @below {{more than one device constraint for dimension: #wave.symbol<"M">}}
func.func private @test_duplicate_device_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#dv_constraint1, #dv_constraint2] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint1 = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
#wg_constraint2 = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 2), workgroup_dim = <x>>
// expected-error @below {{more than one workgroup constraint for dimension: #wave.symbol<"M">}}
func.func private @test_duplicate_wg_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint1, #wg_constraint2] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M), workgroup_dim = <x>>
#wv_constraint1 = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
#wv_constraint2 = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 2)>
// expected-error @below {{more than one wave constraint for dimension: #wave.symbol<"M">}}
func.func private @test_duplicate_wave_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint, #wv_constraint1, #wv_constraint2] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_M = 128}>
#tl_constraint1 = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
#tl_constraint2 = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{more than one tiling constraint for dimension: #wave.symbol<"M">}}
func.func private @test_duplicate_tile_dim() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#tl_constraint1, #tl_constraint2] }

// -----

#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
// expected-error @below {{missing hyperparameters attribute}}
func.func private @test_dev_missing_hyperparams1() attributes { wave.constraints = [#dv_constraint] }

// -----

#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
// expected-error @below {{missing hyperparameters attribute}}
func.func private @test_wg_missing_hyperparams1() attributes { wave.constraints = [#wg_constraint] }

// -----

#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M), workgroup_dim = <x>>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{missing hyperparameters attribute}}
func.func private @test_wave_missing_hyperparams1() attributes { wave.constraints = [#wg_constraint, #wv_constraint] }

// -----

#tl_constraint = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
// expected-error @below {{missing hyperparameters attribute}}
func.func private @test_tile_missing_hyperparams1() attributes { wave.constraints = [#tl_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{K = 1024, BLOCK_K = 128}>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
// expected-error @below {{uses symbolic value #wave.symbol<"M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, K}}
func.func private @test_dev_missing_hyperparams2() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#dv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{K = 1024, BLOCK_K = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
// expected-error @below {{uses symbolic value #wave.symbol<"M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, K}}
func.func private @test_wg_missing_hyperparams2() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{K = 1024, BLOCK_K = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_K] -> (BLOCK_K), workgroup_dim = <x>>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{uses symbolic value #wave.symbol<"M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, K}}
func.func private @test_wave_missing_hyperparams2() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint, #wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{K = 1024, BLOCK_K = 128}>
#tl_constraint = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
// expected-error @below {{uses symbolic value #wave.symbol<"M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, K}}
func.func private @test_tile_missing_hyperparams2() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#tl_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_K = 128}>
#dv_constraint = #wave.device_constraint<dim = <"M">, tile_size = [DEVICE_M] -> (DEVICE_M), device_dim = 0>
// expected-error @below {{uses symbolic value #wave.symbol<"DEVICE_M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, M}}
func.func private @test_dev_missing_hyperparams3() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#dv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_K = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4), workgroup_dim = <x>>
// expected-error @below {{uses symbolic value #wave.symbol<"BLOCK_M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, M}}
func.func private @test_wg_missing_hyperparams3() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_K = 128}>
#wg_constraint = #wave.workgroup_constraint<dim = <"M">, tile_size = [BLOCK_K] -> (BLOCK_K), workgroup_dim = <x>>
#wv_constraint = #wave.wave_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M floordiv 4)>
// expected-error @below {{uses symbolic value #wave.symbol<"BLOCK_M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, M}}
func.func private @test_wave_missing_hyperparams3() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#wg_constraint, #wv_constraint] }

// -----

#hyperparams = #wave.hyperparameters<{M = 1024, BLOCK_K = 128}>
#tl_constraint = #wave.tiling_constraint<dim = <"M">, tile_size = [BLOCK_M] -> (BLOCK_M)>
// expected-error @below {{uses symbolic value #wave.symbol<"BLOCK_M"> not provided as a hyperparameter}}
// expected-note @below {{available symbols: BLOCK_K, M}}
func.func private @test_tile_missing_hyperparams3() attributes { wave.hyperparameters = #hyperparams, wave.constraints = [#tl_constraint] }

// -----
