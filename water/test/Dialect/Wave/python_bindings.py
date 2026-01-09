# REQUIRES: water_python
# RUN: env PYTHONPATH=%py_pkg_root:%PYTHONPATH %python %s | FileCheck %s


from water_mlir import ir
from water_mlir.dialects import wave
from water_mlir.dialects.transform import interpreter
import water_mlir.ir as ir2
from water_mlir.dialects import wave as wave2

# Verify both import styles resolve to the same modules/symbols.
if ir2.Context is not ir.Context:
    raise RuntimeError("module import path for ir differs")
if wave2.register_dialect is not wave.register_dialect:
    raise RuntimeError("module import path for wave differs")
with ir.Context() as ctx:
    wave.register_dialect(ctx)
    wave.register_passes()

    # CHECK: wave.constraints
    print(wave.WAVE_CONSTRAINTS_ATTR_NAME)

    # CHECK: #wave.symbol<"test">
    symbol_attr = wave.WaveSymbolAttr.get("test")
    print(symbol_attr)

    # CHECK: test
    print(symbol_attr.name)

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

    # CHECK: #wave<index_mapping[#wave.index_symbol<WG0>, #wave.index_symbol<T0>, #wave.symbol<"BLOCK_M">] -> (WG0 * 3, WG0 + BLOCK_M, T0 mod WG0)>
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
    index_mapping_attr = wave.WaveIndexMappingAttr.get(
        symbols, start_map, step_map, stride_map
    )
    print(index_mapping_attr)

    # CHECK: ()[s0, s1, s2] -> (s0 * 3)
    print(index_mapping_attr.start)

    # Note that symbols were reordered to ensure a consistent order in the
    # mapping, in particular s1 and s2 were swapped.
    # CHECK: ()[s0, s1, s2] -> (s0 + s2)
    print(index_mapping_attr.step)

    # CHECK: ()[s0, s1, s2] -> (s1 mod s0)
    print(index_mapping_attr.stride)

    # CHECK: 3
    retrieved_symbols = index_mapping_attr.symbols
    print(len(retrieved_symbols))

    # CHECK: #wave.index_symbol<WG0>
    print(retrieved_symbols[0])

    # CHECK: #wave.index_symbol<T0>
    print(retrieved_symbols[1])

    # CHECK: #wave.symbol<"BLOCK_M">
    print(retrieved_symbols[2])

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
    hyper_param = wave.WaveHyperparameterAttr.get({"A": 1, "B": 2, "C": 3})
    print(hyper_param)
    # CHECK: {A = 1 : i64, B = 2 : i64, C = 3 : i64}
    print(hyper_param.mapping)

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

    # CHECK: M
    print(M.name)

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
    # CHECK: ()[s0, s1] -> (s0 floordiv s1)
    print(expr_attr.map)

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
    hardware_constr_1 = wave.HardwareConstraintAttr.get(
        threads_per_wave=64, mma_type=mma_type_attr, max_bits_per_load=128
    )
    print(hardware_constr_1)
    # CHECK: 64
    print(hardware_constr_1.threads_per_wave)
    # CHECK: #wave.mma_kind<f32_16x16x16_f16>
    print(hardware_constr_1.mma_type)
    # CHECK: 128
    print(hardware_constr_1.max_bits_per_load)
    # CHECK: #wave.hardware_constraint<threads_per_wave = 64, vector_shapes = {M = 512 : i64, N = 512 : i64}>
    hardware_constr_2 = wave.HardwareConstraintAttr.get(
        threads_per_wave=64, vector_shapes=shape_dict, max_bits_per_load=128
    )
    print(hardware_constr_2)
    # CHECK: 64
    print(hardware_constr_2.threads_per_wave)
    # CHECK: None
    print(hardware_constr_2.mma_type)
    # CHECK: {M = 512 : i64, N = 512 : i64}
    print(hardware_constr_2.vector_shapes)
    # CHECK: 128
    print(hardware_constr_2.max_bits_per_load)

    # CHECK: #wave.hardware_constraint<threads_per_wave = 64, waves_per_block = [2, 2, 1], vector_shapes = {M = 512 : i64, N = 512 : i64}>
    hardware_constr_3 = wave.HardwareConstraintAttr.get(
        threads_per_wave=64,
        waves_per_block=[2, 2, 1],
        vector_shapes=shape_dict,
        max_bits_per_load=128,
    )
    print(hardware_constr_3)
    # CHECK: [2, 2, 1]
    print(hardware_constr_3.waves_per_block)

    # CHECK: #wave.device_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, device_dim = 0>
    device_constr = wave.DeviceConstraintAttr.get(
        dim="M", tile_size=expr_attr, device_dim=0
    )
    print(device_constr)
    # CHECK: #wave.symbol<"M">
    print(device_constr.dim)
    # CHECK: #wave.expr_list<[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>
    print(device_constr.tile_size)
    # CHECK: 0
    print(device_constr.device_dim)

    # CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, workgroup_dim = <x>>
    wg_constr = wave.WorkgroupConstraintAttr.get(
        dim="M", tile_size=expr_attr, workgroup_dim=wg_dim_x
    )
    print(wg_constr)

    # CHECK: #wave.workgroup_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>, workgroup_dim = <x>, primary = false>
    wg_constr_1 = wave.WorkgroupConstraintAttr.get(
        dim="M", tile_size=expr_attr, workgroup_dim=wg_dim_x, primary=False
    )
    print(wg_constr_1)
    # CHECK: #wave.symbol<"M">
    print(wg_constr_1.dim)
    # CHECK: #wave.expr_list<[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>
    print(wg_constr_1.tile_size)
    # CHECK: #wave.workgroup_dim<x>
    print(wg_constr_1.workgroup_dim)
    # CHECK: False
    print(wg_constr_1.primary)

    # CHECK: #wave.wave_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>>
    wg_constr_2 = wave.WaveConstraintAttr.get(dim="M", tile_size=expr_attr)
    print(wg_constr_2)
    # CHECK: #wave.symbol<"M">
    print(wg_constr_2.dim)
    # CHECK: #wave.expr_list<[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>
    print(wg_constr_2.tile_size)

    # CHECK: #wave.tiling_constraint<dim = <"M">, tile_size = <[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>>
    tiling_constr = wave.TilingConstraintAttr.get(dim="M", tile_size=expr_attr)
    print(tiling_constr)
    # CHECK: #wave.symbol<"M">
    print(tiling_constr.dim)
    # CHECK: #wave.expr_list<[#wave.symbol<"M">, #wave.symbol<"BLOCK_M">] -> (M floordiv BLOCK_M)>
    print(tiling_constr.tile_size)

    # CHECK: #wave.normal_form<none>
    normal_form_attr = wave.WaveNormalFormAttr.get(wave.WaveNormalForm.None_)
    print(normal_form_attr)

    # CHECK: WaveNormalForm.None_
    print(normal_form_attr.value)

    # CHECK: #wave.normal_form<full_func_boundary>
    print(wave.WaveNormalFormAttr.get(wave.WaveNormalForm.FunctionBoundarySpecified))

    # CHECK: #wave.normal_form<full_op_types>
    print(wave.WaveNormalFormAttr.get(wave.WaveNormalForm.OpTypesSpecified))

    # CHECK: #wave.normal_form<index_exprs>
    print(wave.WaveNormalFormAttr.get(wave.WaveNormalForm.IndexExprsSpecified))

    # CHECK: #wave.normal_form<memory_only_types>
    print(wave.WaveNormalFormAttr.get(wave.WaveNormalForm.MemoryOnlyTypes))

    # CHECK: #wave.normal_form<full_types>
    print(wave.WaveNormalFormAttr.get(wave.WaveNormalForm.AllTypesSpecified))

    try:
        wave.WaveNormalFormAttr.get(100)
    except TypeError as e:
        assert "incompatible function arguments" in str(e)
    else:
        assert False, "Expected to fail with TypeError."

    # Invoke a pass using transform dialect. This should not fail.
    transform_module = ir.Module.parse(
        """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_registered_pass "water-wave-detect-normal-forms" to %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}"""
    )
    payload_module = ir.Module.parse("module {}")
    ops = list(transform_module.body.operations)
    entry_op = ops[0]
    interpreter.apply_named_sequence(
        payload_module,
        entry_op,
        transform_module,
    )

    # The pass must have applied and inferred normal forms. We don't care which
    # ones here, this is tested separately, just checking the fact that the pass
    # applied.
    # CHECK: module attributes {wave.normal_form = #wave.normal_form<
    print(payload_module)


# CHECK: wave_ok
print("wave_ok")
