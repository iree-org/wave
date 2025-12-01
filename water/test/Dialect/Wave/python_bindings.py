# REQUIRES: water_python
# RUN: env PYTHONPATH=%py_pkg_root:%PYTHONPATH %python %s | FileCheck %s


from water_mlir import ir
from water_mlir.dialects import wave
import water_mlir.ir as ir2
from water_mlir.dialects import wave as wave2

# Verify both import styles resolve to the same modules/symbols.
if ir2.Context is not ir.Context:
    raise RuntimeError("module import path for ir differs")
if wave2.register_dialect is not wave.register_dialect:
    raise RuntimeError("module import path for wave differs")
with ir.Context() as ctx:
    wave.register_dialect(ctx)

    # CHECK: wave.constraints
    print(wave.WAVE_CONSTRAINTS_ATTR_NAME)

    # CHECK: #wave.symbol<"test">
    print(wave.WaveSymbolAttr.get("test"))

    # CHECK: #wave.index_symbol<WG0>
    index_symbol_attr = wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.WORKGROUP_0)
    print(index_symbol_attr)

    # CHECK: WaveIndexSymbol.WORKGROUP_0
    print(index_symbol_attr.value)

    # CHECK: #wave.index_symbol<T0>
    print(wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.THREAD_0))

    try:
        wave.WaveIndexSymbolAttr.get(100)
    except TypeError as e:
        assert "incompatible function arguments" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    # CHECK: #wave<index_mapping[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.index_symbol<T0>] -> (WG0 * 3, WG0 + BLOCK_M, T0 mod WG0)>
    symbols = [
        wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.WORKGROUP_0),
        wave.WaveSymbolAttr.get("BLOCK_M"),
        wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.THREAD_0),
    ]
    s0 = ir.AffineSymbolExpr.get(0)
    s1 = ir.AffineSymbolExpr.get(1)
    s2 = ir.AffineSymbolExpr.get(2)
    start_map = ir.AffineMap.get(0, 3, [s0 * 3])
    step_map = ir.AffineMap.get(0, 3, [s0 + s1])
    stride_map = ir.AffineMap.get(0, 3, [s2 % s0])
    print(wave.WaveIndexMappingAttr.get(symbols, start_map, step_map, stride_map))

    try:
        wave.WaveIndexMappingAttr.get([], start_map, step_map, stride_map)
    except ValueError as e:
        assert "co-indexed" in str(e)
    else:
        assert False, "Expected to fail with ValueError."

    try:
        dimension_map = ir.AffineMap.get(1, 0, [])
        wave.WaveIndexMappingAttr.get([], dimension_map, dimension_map, dimension_map)
    except ValueError as e:
        assert "not involve dimensions" in str(e)
    else:
        assert False, "Expected to fail with ValueError."

    try:
        no_result_map = ir.AffineMap.get(0, 3, [])
        wave.WaveIndexMappingAttr.get(symbols, start_map, no_result_map, stride_map)
    except ValueError as e:
        assert "same number of results" in str(e)
    else:
        assert False, "Expected to fail with ValueError."

    try:
        wave.WaveIndexMappingAttr.get(
            ["string", "instead", "of", "attrs"], start_map, step_map, stride_map
        )
    except TypeError as e:
        assert "ir.Attribute" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    # CHECK: #wave.hyperparameters<{A = 1 : i64, B = 2 : i64, C = 3 : i64}>
    print(wave.WaveHyperparameterAttr.get({"A": 1, "B": 2, "C": 3}))

    try:
        wave.WaveHyperparameterAttr.get({"A": 1.0})
    except TypeError as e:
        assert "must be an integer" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    try:
        wave.WaveHyperparameterAttr.get({3: 1})
    except TypeError as e:
        assert "must be a string" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    try:
        wave.WaveHyperparameterAttr.get({"A": 10**100})
    except ValueError as e:
        assert "too large" in str(e)
    else:
        assert False, "Expected to fail with ValueError."

    # CHECK: #wave.address_space<shared>
    addr_attr = wave.WaveAddressSpaceAttr.get(wave.WaveAddressSpace.Shared)
    print(addr_attr)

    # CHECK: WaveAddressSpace.Shared
    print(addr_attr.value)

    try:
        wave.WaveAddressSpaceAttr.get(5)
    except TypeError as e:
        assert "incompatible function arguments" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    # CHECK: #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.index_symbol<T0>] -> (WG0 * 3)>
    symbol_attrs_for_expr = [
        wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.WORKGROUP_0),
        wave.WaveSymbolAttr.get("BLOCK_M"),
        wave.WaveIndexSymbolAttr.get(wave.WaveIndexSymbol.THREAD_0),
    ]
    expr_attr = wave.WaveExprListAttr.get(symbol_attrs_for_expr, start_map)
    print(expr_attr)

    try:
        wave.WaveExprListAttr.get(symbol_attrs_for_expr[:-1], start_map)
    except ValueError as e:
        assert "as many entries as map have symbols" in str(e)
    else:
        assert False, "Expected to fail with ValueError."

    # CHECK: #wave.read_write_bounds<{M = #wave.expr_list<[#wave.index_symbol<WG0>, #wave.symbol<"BLOCK_M">, #wave.index_symbol<T0>] -> (WG0 * 3)>}>
    print(wave.WaveReadWriteBoundsAttr.get({"M": expr_attr}))

    try:
        wave.WaveReadWriteBoundsAttr.get({3: expr_attr})
    except TypeError as e:
        assert "must be a string" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    try:
        wave.WaveReadWriteBoundsAttr.get({"A": 1.0})
    except TypeError as e:
        assert "must be an attribute" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    try:
        wave.WaveReadWriteBoundsAttr.get({"A": addr_attr})
    except TypeError as e:
        assert "must be a WaveExprListAttr" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    # CHECK: #wave.mma_kind<f32_16x16x16_f16>
    mma_type_attr = wave.WaveMmaKindAttr.get(wave.WaveMmaKind.F32_16x16x16_F16)
    print(mma_type_attr)

    # CHECK: #wave.workgroup_dim<x>
    wg_dim_x = wave.WaveWorkgroupDimAttr.get(wave.WaveWorkgroupDim.X)
    print(wg_dim_x)

    # CHECK: #wave.symbol<"M">
    M = wave.WaveSymbolAttr.get("M")
    print(M)

    # CHECK: #wave.expr_list<[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>
    symbol_attrs = [
        wave.WaveSymbolAttr.get("M"),
        wave.WaveSymbolAttr.get("BLOCK_M"),
    ]
    s0 = ir.AffineSymbolExpr.get(0)
    s1 = ir.AffineSymbolExpr.get(1)
    expr_map = ir.AffineMap.get(0, 2, [ir.AffineExpr.get_floor_div(s0, s1)])
    expr_attr = wave.WaveExprListAttr.get(symbol_attrs, expr_map)
    print(expr_attr)

    # CHECK: #wave.expr_list<[] -> (512)>
    int_expr_map = ir.AffineMap.get(0, 0, [ir.AffineExpr.get_constant(512)])
    int_expr_attr = wave.WaveExprListAttr.get([], int_expr_map)
    print(int_expr_attr)

    # CHECK: {M = 512 : i64, N = 512 : i64}
    i64 = ir.IntegerType.get_signless(64)
    shape_dict = ir.DictAttr.get(
        {"M": ir.IntegerAttr.get(i64, 512), "N": ir.IntegerAttr.get(i64, 512)}
    )
    print(shape_dict)

    # CHECK: #wave.hardware_constraint<threads_per_wave = 64, mma_type = <f32_16x16x16_f16>>
    print(
        wave.HardwareConstraintAttr.get(
            threads_per_wave=64, mma_type=mma_type_attr, max_bits_per_load=128
        )
    )

    # CHECK: #wave.hardware_constraint<threads_per_wave = 64, vector_shapes = {M = 512 : i64, N = 512 : i64}>
    print(
        wave.HardwareConstraintAttr.get(
            threads_per_wave=64, vector_shapes=shape_dict, max_bits_per_load=128
        )
    )

    # CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1], vector_shapes = {M = 512 : i64, N = 512 : i64}>
    print(
        wave.HardwareConstraintAttr.get(
            threads_per_wave=64,
            waves_per_block=[2, 2, 1],
            vector_shapes=shape_dict,
            max_bits_per_load=128,
        )
    )

    # CHECK: #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, device_dim = 0>
    print(wave.DeviceConstraintAttr.get(dim="M", tile_size=expr_attr, device_dim=0))

    # CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, workgroup_dim = <x>>
    print(
        wave.WorkgroupConstraintAttr.get(
            dim="M", tile_size=expr_attr, workgroup_dim=wg_dim_x
        )
    )

    # CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, workgroup_dim = <x>, primary = false>
    print(
        wave.WorkgroupConstraintAttr.get(
            dim="M", tile_size=expr_attr, workgroup_dim=wg_dim_x, primary=False
        )
    )

    # CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>>
    print(wave.WaveConstraintAttr.get(dim="M", tile_size=expr_attr))

    # CHECK: #wave.tiling_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>>
    print(wave.TilingConstraintAttr.get(dim="M", tile_size=expr_attr))

# CHECK: wave_ok
print("wave_ok")
