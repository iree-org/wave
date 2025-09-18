# RUN: python %s | FileCheck %s

# This file contains location tests that check exact source line and column numbers.
# These tests are sensitive to minor edits and line number changes.

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)

M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_reduce_op_location():
    K = tkl.sym.K
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, K: 128},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(K, BLOCK_N)]

    subs = {
        M: 256,
        K: 128,
        BLOCK_M: 1,
        BLOCK_N: 128,
        ELEMS_PER_THREAD: 2,
        ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
    }
    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(
            LocationCaptureLevel.FILE_LINE_COL
        ),
        use_local_scope=True,
        canonicalize=False,
    )
    from wave_lang.kernel.wave.utils.compile_utils import set_default_compile_config

    options = set_default_compile_config(options)

    @tkw.wave(constraints)
    def reduce_sum_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res_f32 = tkw.cast(res, tkl.f32)
        reduced = tkw.sum(res_f32, dim=K)
        tkw.write(reduced, c, elements_per_thread=1)

    reduce_sum_kernel = wave_compile(options, reduce_sum_kernel)
    print(reduce_sum_kernel.asm)

    # CHECK-LABEL: @reduce_sum_kernel
    # CHECK: vector.load {{.*}} loc("{{.*}}specific_location.py":66
    # CHECK: vector.load {{.*}} loc("{{.*}}specific_location.py":67

    # multiply
    # CHECK: arith.mulf {{.*}} loc("{{.*}}specific_location.py":68
    # cast
    # CHECK: arith.extf {{.*}} loc("{{.*}}specific_location.py":69

    # reduce
    # CHECK: arith.addf {{.*}} loc("{{.*}}specific_location.py":70
    # CHECK: gpu.shuffle {{.*}} loc("{{.*}}specific_location.py":70
    # CHECK: arith.addf {{.*}} loc("{{.*}}specific_location.py":70
    # CHECK: gpu.shuffle {{.*}} loc("{{.*}}specific_location.py":70

    # write
    # CHECK: vector.store {{.*}} loc("{{.*}}specific_location.py":71
