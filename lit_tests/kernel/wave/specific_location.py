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
BLOCK_M = tkl.sym.BLOCK_M
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@run_test
def test_reduce_op_location():
    K = tkl.sym.K
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, K: 128},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    subs = {
        M: 256,
        K: 128,
        BLOCK_M: 1,
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
        drop_debug_info_before_mlir=False,
    )
    from wave_lang.kernel.wave.utils.compile_utils import set_default_compile_config

    options = set_default_compile_config(options)

    @tkw.wave(constraints)
    def reduce_sum_kernel(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        res = lhs * rhs
        res_f32 = tkw.cast(res, tkl.f32)
        reduced = tkw.sum(res_f32, dim=K)
        tkw.write(reduced, c)

    reduce_sum_kernel = wave_compile(options, reduce_sum_kernel)
    print(reduce_sum_kernel.asm)

    # CHECK-LABEL: @reduce_sum_kernel

    # Test that placeholder locations are placed on function arguments.
    # CHECK: @reduce_sum_kernel(%{{.*}} loc("a"("{{.*}}specific_location.py":54{{.*}})))

    # load
    # CHECK: vector.load {{.*}} loc("{{.*}}specific_location.py":60
    # CHECK: vector.load {{.*}} loc("{{.*}}specific_location.py":61

    # multiply
    # CHECK: arith.mulf {{.*}} loc("{{.*}}specific_location.py":62
    # cast
    # CHECK: arith.extf {{.*}} loc("{{.*}}specific_location.py":63

    # reduce
    # CHECK: arith.addf {{.*}} loc("{{.*}}specific_location.py":64
    # CHECK: gpu.shuffle {{.*}} loc("{{.*}}specific_location.py":64
    # CHECK: arith.addf {{.*}} loc("{{.*}}specific_location.py":64
    # CHECK: gpu.shuffle {{.*}} loc("{{.*}}specific_location.py":64

    # write
    # CHECK: vector.store {{.*}} loc("{{.*}}specific_location.py":65
