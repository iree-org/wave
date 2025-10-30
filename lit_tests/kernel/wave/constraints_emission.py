# REQUIRES: water
# RUN: python %s | FileCheck %s


import sympy
from typing import Any


from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_workgroup_and_wave_constraints():
    """Test emission of WorkgroupConstraint and WaveConstraint."""

    # Define constraints
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 16, N: 16}),
    ]

    @wave.wave(constraints)
    def simple_kernel(
        a: Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        c_reg = a_reg + b_reg
        wave.write(c_reg, b)

    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 128,
        M: 256,
        N: 512,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, simple_kernel)
    trace = compiled_kernel.get_compiled_graph()

    mlir_output, _ = emit_wave_dialect(trace, simple_kernel.constraints, options, False)
    print(mlir_output)

    # CHECK-LABEL: test_workgroup_and_wave_constraints
    # CHECK: module
    # CHECK: func.func @kernel
    # CHECK-SAME: wave.constraints = [
    # CHECK-SAME: #wave.workgroup_constraint<@M, 64, 0>
    # CHECK-SAME: #wave.workgroup_constraint<@N, 128, 1>
    # CHECK-SAME: #wave.wave_constraint<@M, 32>
    # CHECK-SAME: #wave.wave_constraint<@N, 64>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64
    # CHECK-SAME: ]


@run_test
def test_tiling_constraint():
    """Test emission of TilingConstraint."""

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(threads_per_wave=64, mma_type=MMAType.F32_16x16x16_F16),
    ]

    @tkw.wave(constraints)
    def matmul_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

    subs = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 16,
        M: 128,
        N: 128,
        K: 64,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, matmul_kernel)
    trace = compiled_kernel.compiled_graph

    mlir_output, _ = emit_wave_dialect(trace, matmul_kernel.constraints, options, False)
    print(mlir_output)

    # CHECK-LABEL: test_tiling_constraint
    # CHECK: module
    # CHECK: func.func @kernel
    # CHECK-SAME: wave.constraints = [
    # CHECK-SAME: #wave.workgroup_constraint<@M, 64, 0>
    # CHECK-SAME: #wave.workgroup_constraint<@N, 64, 1>
    # CHECK-SAME: #wave.tiling_constraint<@K, 16>
    # CHECK-SAME: #wave.wave_constraint<@M, 32>
    # CHECK-SAME: #wave.wave_constraint<@N, 32>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64
    # CHECK-SAME: mma_type = #wave.mma_kind<f32_16x16x16_f16>
    # CHECK-SAME: ]


@run_test
def test_hardware_constraint_with_vector_shapes():
    """Test emission of HardwareConstraint with vector_shapes."""

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 4)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 4)),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 32, N: 32},
        ),
    ]

    @wave.wave(constraints)
    def kernel_with_vector_shapes(
        a: Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = wave.read(a)
        b_reg = wave.read(b)
        c_reg = a_reg * b_reg
        wave.write(c_reg, a)

    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 128,
        BLOCK_N: 128,
        M: 512,
        N: 512,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, kernel_with_vector_shapes)
    trace = compiled_kernel.get_compiled_graph()

    mlir_output, _ = emit_wave_dialect(
        trace, kernel_with_vector_shapes.constraints, options, False
    )
    print(mlir_output)

    # CHECK-LABEL: test_hardware_constraint_with_vector_shapes
    # CHECK: module
    # CHECK: func.func @kernel
    # CHECK-SAME: wave.constraints = [
    # CHECK-SAME: #wave.workgroup_constraint<@M, 128, 0>
    # CHECK-SAME: #wave.workgroup_constraint<@N, 128, 1>
    # CHECK-SAME: #wave.wave_constraint<@M, 32>
    # CHECK-SAME: #wave.wave_constraint<@N, 32>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64
    # CHECK-SAME: vector_shapes = {@M = 32, @N = 32}
    # CHECK-SAME: ]


@run_test
def test_multiple_workgroup_dimensions():
    """Test emission of multiple WorkgroupConstraints on different dimensions."""

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(K, BLOCK_K, 2),
        tkw.WaveConstraint(M, 16),
        tkw.WaveConstraint(N, 16),
        tkw.HardwareConstraint(threads_per_wave=64),
    ]

    @wave.wave(constraints)
    def kernel_3d_workgroup(
        a: Memory[M, N, K, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = wave.read(a)
        b_reg = a_reg + a_reg
        wave.write(b_reg, a)

    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 32,
        M: 64,
        N: 64,
        K: 64,
    }

    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)

    compiled_kernel = wave_compile(options, kernel_3d_workgroup)
    trace = compiled_kernel.get_compiled_graph()

    mlir_output, _ = emit_wave_dialect(
        trace, kernel_3d_workgroup.constraints, options, False
    )
    print(mlir_output)

    # CHECK-LABEL: test_multiple_workgroup_dimensions
    # CHECK: module
    # CHECK: func.func @kernel
    # CHECK-SAME: wave.constraints = [
    # CHECK-SAME: #wave.workgroup_constraint<@M, 32, 0>
    # CHECK-SAME: #wave.workgroup_constraint<@N, 32, 1>
    # CHECK-SAME: #wave.workgroup_constraint<@K, 32, 2>
    # CHECK-SAME: #wave.wave_constraint<@M, 16>
    # CHECK-SAME: #wave.wave_constraint<@N, 16>
    # CHECK-SAME: #wave.hardware_constraint<threads_per_wave = 64
    # CHECK-SAME: ]
