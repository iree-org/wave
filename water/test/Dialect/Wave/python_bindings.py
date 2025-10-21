# REQUIRES: water_python
# RUN: env PYTHONPATH=%py_pkg_root:%PYTHONPATH %python %s | FileCheck %s

try:
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

        # CHECK: #wave.symbol<"test">
        print(wave.WaveSymbolAttr.get("test"))

        # CHECK: #wave<index_mapping[$WG0, BLOCK_M, $T0] -> ($WG0 * 3, $WG0 + BLOCK_M, $T0 mod $WG0)>
        symbol_names = ["$WG0", "BLOCK_M", "$T0"]
        s0 = ir.AffineSymbolExpr.get(0)
        s1 = ir.AffineSymbolExpr.get(1)
        s2 = ir.AffineSymbolExpr.get(2)
        start_map = ir.AffineMap.get(0, 3, [s0 * 3])
        step_map = ir.AffineMap.get(0, 3, [s0 + s1])
        stride_map = ir.AffineMap.get(0, 3, [s2 % s0])
        print(
            wave.WaveIndexMappingAttr.get(symbol_names, start_map, step_map, stride_map)
        )

        try:
            wave.WaveIndexMappingAttr.get([], start_map, step_map, stride_map)
        except ValueError as e:
            assert "co-indexed" in str(e)
        else:
            assert False, "Expected to fail with ValueError."

        try:
            dimension_map = ir.AffineMap.get(1, 0, [])
            wave.WaveIndexMappingAttr.get(
                [], dimension_map, dimension_map, dimension_map
            )
        except ValueError as e:
            assert "not involve dimensions" in str(e)
        else:
            assert False, "Expected to fail with ValueError."

        try:
            no_result_map = ir.AffineMap.get(0, 3, [])
            wave.WaveIndexMappingAttr.get(
                symbol_names, start_map, no_result_map, stride_map
            )
        except ValueError as e:
            assert "same number of results" in str(e)
        else:
            assert False, "Expected to fail with ValueError."

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
        print(addr_attr.value())

        try:
            wave.WaveAddressSpaceAttr.get(5)
        except TypeError as e:
            assert "incompatible function arguments" in str(e)
        else:
            assert False, "Expected to fail with TypeError."

        # CHECK: #wave.expr<[$WG0, BLOCK_M, $T0] -> ($WG0 * 3)>
        expr_attr = wave.WaveExprAttr.get(symbol_names, start_map)
        print(expr_attr)

        try:
            wave.WaveExprAttr.get(symbol_names[:-1], start_map)
        except ValueError as e:
            assert "as many entries as map have symbols" in str(e)
        else:
            assert False, "Expected to fail with ValueError."

        # CHECK: #wave.read_write_bounds<{M = #wave.expr<[$WG0, BLOCK_M, $T0] -> ($WG0 * 3)>}>
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
            assert "must be a WaveExprAttr" in str(e)
        else:
            assert False, "Expected to fail with TypeError."

    # CHECK: wave_ok
    print("wave_ok")
except Exception as e:
    print("error:", e)
