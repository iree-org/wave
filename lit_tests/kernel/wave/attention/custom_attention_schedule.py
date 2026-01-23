# RUN: python %s | FileCheck %s

"""
Lit test for tagged attention kernel with manual prefetch schedule.

This test verifies that the tagged attention kernel template from
templates.tagged_attention compiles correctly with the manual
attention prefetch schedule from schedules.attention_prefetch.

The tagged attention kernel provides operation tags for custom scheduling:
- "n_kv_loop": Main iteration loop over KV sequence
- "read_q": Q read from global memory (hoisted outside loop)
- "read_k": K read (GatherToLDS + shared read)
- "read_v": V read (GatherToLDS + shared read)
- "mma_qk": QK MMA operation (GEMM0)
- "mma_pv": PV MMA operation (GEMM1)
- "softmax0_*": First group of softmax ops (max, exp2, sum)
- "softmax1_*": Second group of softmax ops (delta_max scaling)
"""

import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.templates.tagged_attention import (
    get_tagged_bshd_attention_kernel,
)
from wave_lang.kernel.wave.schedules.attention_prefetch import (
    get_attention_prefetch_schedule,
)
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)


@run_test
def test_tagged_attention_kernel():
    """
    Test that the tagged BSHD attention kernel compiles with manual schedule.

    This verifies:
    1. The tagged_attention kernel from templates module can be created
    2. The attention_prefetch_schedule from schedules module works
    3. The kernel + schedule compiles successfully with GatherToLDS and hardware transpose
    """
    # Configuration from manual_scheduled_attention_test.py
    shape = AttentionShape(
        num_query_heads=4,
        num_kv_heads=4,
        query_seq_len=256,
        kv_seq_len=256,
        head_size=64,
        head_size_kv=64,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16, tkw.MMAType.F32_16x16x16_F16)

    # Get the tagged BSHD attention kernel from the templates module
    tagged_attention, hyperparams, _ = get_tagged_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
        is_causal=False,
        num_waves=8,
    )
    hyperparams.update(get_default_scheduling_params())

    # Get the manual attention prefetch schedule
    attention_schedule = get_attention_prefetch_schedule()

    # Compile with manual scheduling using GatherToLDS
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.MANUAL,
        use_scheduling_barriers=False,
        use_global_to_shared=True,
        compile_to_mlir=True,
        target="gfx950",
    )

    kernel = wave_compile(options, tagged_attention, attention_schedule)

    # Print the MLIR for FileCheck
    print("=== TAGGED ATTENTION KERNEL WITH MANUAL SCHEDULE ===")
    print(kernel.asm)

    # CHECK-LABEL: === TAGGED ATTENTION KERNEL WITH MANUAL SCHEDULE ===
    # CHECK: func.func @tagged_attention

    # Verify GatherToLDS operations for K and V matrices (global -> shared memory)
    # CHECK: amdgpu.gather_to_lds
    # CHECK: amdgpu.gather_to_lds

    # Verify hardware transpose loads for V matrix (transposed read from shared memory)
    # CHECK: amdgpu.transpose_load
    # CHECK: amdgpu.transpose_load

    # Verify MMA operations are present (QK and PV GEMMs)
    # CHECK: amdgpu.mfma

    # Verify softmax operations (shuffle for reductions)
    # CHECK: gpu.shuffle


if __name__ == "__main__":
    pass
