# REQUIRES: water_python
# RUN: env PYTHONPATH=%py_pkg_root:%PYTHONPATH %python %s | FileCheck %s
# CHECK: wave_ok

try:
    from water_mlir import ir
    from water_mlir.dialects import wave
    import water_mlir.ir as ir2
    import water_mlir.dialects as dialects2
    from water_mlir.dialects import wave as wave2

    # Verify both import styles resolve to the same modules/symbols.
    if ir2.Context is not ir.Context:
        raise RuntimeError("module import path for ir differs")
    if wave2.register_dialect is not wave.register_dialect:
        raise RuntimeError("module import path for wave differs")
    with ir.Context() as ctx:
        wave.register_dialect(ctx)
    print("wave_ok")
except Exception as e:
    print("error:", e)
