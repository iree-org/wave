# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import sympy

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType


##################################################################
#  Misc Kernels
##################################################################


def get_broadcast_scaled_add(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]

    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def broadcast_scaled_add(
        a: tkl.Memory[M, N / 2, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N / 2, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        rhs = tkw.broadcast(rhs, [M, N / 2])
        res = lhs + rhs
        tkw.write(res, c)

    hyperparams = {
        M: shape[0],
        N: shape[1],
        ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
    }

    return broadcast_scaled_add, hyperparams


##################################################################
#  MXFP GEMM Related Kernels
##################################################################


# This kernel slowly updated to have fused dynamic quantization.
def get_broadcasted_scale_gemm_mxfp4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def broadcasted_scale_scaled_mma(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.broadcast(a_scale_reg, [M, K / 32])
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())
    return broadcasted_scale_scaled_mma, hyperparams


##################################################################
#  Kernels with explicit schedule
##################################################################


def get_gemm_prefetch_kernel_and_schedule(
    shape: tuple[int, int, int] = (128, 128, 128),
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16,
    compile_to_mlir: bool = True,
):
    """
    Returns a GEMM kernel with prefetch schedule that demonstrates pipelining.

    Args:
        shape: Tuple of (M, N, K) dimensions for the matrix multiplication
        mfma_variant: The MMA type to use for hardware constraint
        compile_to_mlir: Whether to compile to MLIR IR

    Returns:
        tuple: (kernel_function, schedule_function, compile_options)
    """
    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Basic constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel
    @tkw.wave(constraints)
    def gemm_prefetch(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the schedule
    @wave_schedule.wave_schedule()
    def prefetch_schedule():
        # Get nodes to be manipulated in the schedule.
        k_loop = tkw.get_node_by_tag("k_loop")
        load_a = tkw.get_node_by_tag_and_type("read_a", tkw.Read)
        global_load_a, shared_load_a = tkw.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = tkw.get_node_by_tag_and_type("read_a", tkw.Write)
        load_b = tkw.get_node_by_tag_and_type("read_b", tkw.Read)
        global_load_b, shared_load_b = tkw.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = tkw.get_node_by_tag_and_type("read_b", tkw.Write)
        mma = tkw.get_node_by_tag("mma")

        # Create a pipeline with 2 stages and specify the operations that are overlapping.
        with tkw.pipeline(k_loop) as pipelined_loop:
            pipelined_loop.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            pipelined_loop.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=True,
        compile_to_mlir=compile_to_mlir,
    )

    return gemm_prefetch, prefetch_schedule, options
