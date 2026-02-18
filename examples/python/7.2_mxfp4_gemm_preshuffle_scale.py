"""
MXFP4 GEMM with Pre-shuffled Scales - Scheduled

Double-buffered MXFP4 GEMM with pre-shuffled e8m0 scale matrices.
Scale matrices are pre-shuffled on the host and read from global memory
with an IndexMapping that encodes the shuffle pattern. Data tensors still
go through shared memory (LDS) for double-buffered prefetch.

Usage:
    python 7.2_mxfp4_gemm_preshuffle_scale.py --test test_basic_preshuffle
    python 7.2_mxfp4_gemm_preshuffle_scale.py --test test_dbuf_4wave_preshuffle
    python 7.2_mxfp4_gemm_preshuffle_scale.py --test test_dbuf_8wave_preshuffle
    python 7.2_mxfp4_gemm_preshuffle_scale.py --test test_dbuf_8wave_preshuffle --debug
    python 7.2_mxfp4_gemm_preshuffle_scale.py --test test_vanilla_mxfp_gemm
    python 7.2_mxfp4_gemm_preshuffle_scale.py --benchmark
    python 7.2_mxfp4_gemm_preshuffle_scale.py --benchmark --shape 2048,2048,8192
    python 7.2_mxfp4_gemm_preshuffle_scale.py --list_tests
"""

import torch

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.scheduling.schedule_enums import SchedulingType
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
)

# Non-preshuffle kernel and schedule (from 7.1)
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm
from wave_lang.kernel.wave.schedules import get_mxfp4_dbuf_schedule

from utils import list_tests, run_test


# ---------------------------------------------------------------------------
# Host-side scale shuffle
# ---------------------------------------------------------------------------


def e8m0_shuffle(scale):
    """Shuffle scale tensor into the e8m0 hardware layout.

    Coordinate transform from
    https://github.com/ROCm/rocm-libraries/blob/4348901528fe100a84975b89c247eece553a2a2d/shared/mxdatagenerator/lib/include/mxDataGenerator/PreSwizzle.hpp#L403

    1. Pad to ((m+255)//256*256, (n+7)//8*8)
    2. Reshape to (sm//32, 2, 16, sn//8, 2, 4)
    3. Permute dims: (0, 3, 5, 2, 4, 1)
    4. Flatten back to (sm, sn)
    """
    if scale is None or scale.dtype == torch.float32:
        return scale
    assert scale.ndim == 2, "scale must be a 2D tensor"
    m, n = scale.shape
    scale_padded = torch.zeros(
        (m + 255) // 256 * 256,
        (n + 7) // 8 * 8,
        dtype=scale.dtype,
        device=scale.device,
    )
    scale_padded[:m, :n] = scale
    scale = scale_padded
    sm, sn = scale.shape
    scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
    scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale


# ---------------------------------------------------------------------------
# Tagged preshuffle kernel
# ---------------------------------------------------------------------------


def get_tagged_preshuffle_mxfp4_gemm(
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block_shape: tuple[int, int, int] = (256, 256, 256),
    mfma_variant: ScaledMMAType = ScaledMMAType.F32_16x16x128_F8F6F4,
    num_waves: int = 8,
):
    """Return a tagged preshuffle MXFP4 GEMM kernel + compile options.

    Like get_tagged_mxfp4_gemm, but scale tensors use GLOBAL_ADDRESS_SPACE
    (not shared/LDS) and reads use IndexMapping for the shuffled layout.

    All ops are tagged for use with schedule functions.

    Returns:
        (kernel_function, WaveCompileOptions)
    """
    from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    if num_waves == 8:
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    else:
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # IndexMapping for shuffled A scales: logical (K, M) -> physical shuffled layout
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            M: (
                (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                + (i // 8) * 256
                + ((i % 8) % 4) * 64
                + ((j % 32) % 16) * 4
                + (((i % 8) // 4) * 2)
                + ((j % 32) // 16)
            )
            // K_SCALE_SHUFFLED,
            K: (
                (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                + (i // 8) * 256
                + ((i % 8) % 4) * 64
                + ((j % 32) % 16) * 4
                + (((i % 8) // 4) * 2)
                + ((j % 32) // 16)
            )
            % K_SCALE_SHUFFLED,
        },
        outputs={K: i, M: j},
    )

    # IndexMapping for shuffled B scales: logical (K, N) -> physical shuffled layout
    k = tkw.IndexMapping.iterator(0)
    n = tkw.IndexMapping.iterator(1)

    b_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: (
                (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                + (k // 8) * 256
                + ((k % 8) % 4) * 64
                + ((n % 32) % 16) * 4
                + (((k % 8) // 4) * 2)
                + ((n % 32) // 16)
            )
            // K_SCALE_SHUFFLED,
            K: (
                (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                + (k // 8) * 256
                + ((k % 8) % 4) * 64
                + ((n % 32) % 16) * 4
                + (((k % 8) // 4) * 2)
                + ((n % 32) // 16)
            )
            % K_SCALE_SHUFFLED,
        },
        outputs={K: k, N: n},
    )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")
            b_reg = tkw.read(b, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, mapping=b_scale_mapping, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")
            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    k_scale_shuffled = (((shape[2] // 32) + 7) // 8) * 8

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_shape[0],
        BLOCK_N: block_shape[1],
        BLOCK_K: block_shape[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
        K_SCALE_SHUFFLED: k_scale_shuffled,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        use_global_to_shared=True,
        minimize_shared_allocs=False,
    )

    return gemm, options


# ---------------------------------------------------------------------------
# Double-buffer schedule adapted for preshuffle (scales bypass LDS)
# ---------------------------------------------------------------------------


def get_preshuffle_dbuf_schedule(use_stagger: bool = True):
    """Return a double-buffered schedule for preshuffle MXFP4 kernels.

    Adapted from get_mxfp4_dbuf_schedule for kernels where scale tensors are
    read directly from global memory (no LDS).  Only data tensors (A, B) use
    GatherToLDS async prefetch; scale reads are direct global loads scheduled
    alongside compute.

    Args:
        use_stagger: Enable wave staggering + WorkgroupBarrier in cluster 0.
            Recommended for 8-wave configs; disable for 4-wave.
    """
    K = tkl.sym.K

    @wave_schedule.wave_schedule()
    def preshuffle_dbuf_schedule():
        k_loop = tkw.get_node_by_tag("k_loop")

        # Data tensors go through LDS (GatherToLDS + shared Read)
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        # Scale tensors are direct global reads (no LDS, no GatherToLDS)
        read_a_scale = tkw.get_node_by_tag("read_a_scale")
        read_b_scale = tkw.get_node_by_tag("read_b_scale")

        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")

        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        # =================================================================
        # 2-stage pipeline (double buffering)
        # =================================================================
        pipeline_loop = tkw.pipeline(k_loop)

        with pipeline_loop as pl:
            # Stage 0: Prefetch data via GatherToLDS (scales bypass LDS)
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                    (),
                ],
            )
            # Stage 1: Shared loads for data + global reads for scales
            #           + bitcasts + MMA
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_b,
                        read_a_scale,
                        read_b_scale,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ],
            )

        # =================================================================
        # KERNEL: Main loop body with custom cluster ordering
        # =================================================================

        # Filter nodes for KERNEL stage
        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        ) + tkw.filter_nodes(global_to_shared_b, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_b = tkw.filter_nodes(
            shared_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_read_a_scale = tkw.filter_nodes(
            read_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_read_b_scale = tkw.filter_nodes(
            read_b_scale, subgraph=pipeline_loop.KERNEL
        )

        loop_bitcast_a = tkw.filter_nodes(bitcast_a, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_a_scale = tkw.filter_nodes(
            bitcast_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_bitcast_b = tkw.filter_nodes(bitcast_b, subgraph=pipeline_loop.KERNEL)
        loop_bitcast_b_scale = tkw.filter_nodes(
            bitcast_b_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_scaled_mma = tkw.filter_nodes(scaled_mma, subgraph=pipeline_loop.KERNEL)

        # Partition by K dimension for interleaving compute with memory ops
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_b_0, loop_shared_load_b_1 = tkw.partition_by_dim(
            loop_shared_load_b, dim=K, num_partitions=2
        )
        loop_read_a_scale_0, loop_read_a_scale_1 = tkw.partition_by_dim(
            loop_read_a_scale, dim=K, num_partitions=2
        )
        loop_read_b_scale_0, loop_read_b_scale_1 = tkw.partition_by_dim(
            loop_read_b_scale, dim=K, num_partitions=2
        )
        loop_bitcast_a_0, loop_bitcast_a_1 = tkw.partition_by_dim(
            loop_bitcast_a, dim=K, num_partitions=2
        )
        loop_bitcast_a_scale_0, loop_bitcast_a_scale_1 = tkw.partition_by_dim(
            loop_bitcast_a_scale, dim=K, num_partitions=2
        )
        loop_bitcast_b_0, loop_bitcast_b_1 = tkw.partition_by_dim(
            loop_bitcast_b, dim=K, num_partitions=2
        )
        loop_bitcast_b_scale_0, loop_bitcast_b_scale_1 = tkw.partition_by_dim(
            loop_bitcast_b_scale, dim=K, num_partitions=2
        )

        independent_global_count = len(loop_global_to_shared)

        # Cluster 0: First K-partition loads/bitcasts + async GatherToLDS
        cluster_0_ops = [
            loop_shared_load_a_0,
            loop_read_a_scale_0,
            loop_shared_load_b_0,
            loop_read_b_scale_0,
            loop_bitcast_a_0,
            loop_bitcast_a_scale_0,
            loop_bitcast_b_0,
            loop_bitcast_b_scale_0,
            tkw.SchedulingBarrier([]),
            loop_global_to_shared,
            tkw.SchedulingBarrier([]),
        ]
        if use_stagger:
            cluster_0_ops.extend(
                [
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            )

        clusters = [
            # Cluster 0: First K-partition shared loads/bitcasts + async prefetch
            tkw.cluster(cluster_0_ops),
            # Cluster 1: First K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: Second K-partition shared loads/bitcasts
            tkw.cluster(
                [
                    loop_shared_load_a_1,
                    loop_read_a_scale_1,
                    loop_shared_load_b_1,
                    loop_read_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Second K-partition scaled_mma (high priority)
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Shared memory barriers at loop boundaries
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        if use_stagger:
            tkw.stagger(pipeline_loop.KERNEL)

    return preshuffle_dbuf_schedule


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run_mxfp_gemm(gemm, shape):
    """Run compiled non-preshuffle GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_preshuffle_mxfp_gemm(gemm, shape):
    """Run compiled preshuffle GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x_scales_shuffled = e8m0_shuffle(x_scales)
    w_scales_shuffled = e8m0_shuffle(w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales_shuffled = x_scales_shuffled.cuda()
    w_scales_shuffled = w_scales_shuffled.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales_shuffled, w.T.contiguous(), w_scales_shuffled, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench_kernel(kernel_func, inputs, warmup=5, iters=20):
    """Time a kernel using CUDA events. Returns mean time in microseconds."""
    for _ in range(warmup):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        kernel_func(*inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def _make_inputs_non_preshuffle(shape):
    """Create GPU inputs for non-preshuffle MXFP4 kernels."""
    m, n, k = shape
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    return (x, x_scales, w.T.contiguous(), w_scales, out)


def _make_inputs_preshuffle(shape):
    """Create GPU inputs for preshuffle MXFP4 kernels."""
    m, n, k = shape
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    x_scales_shuffled = e8m0_shuffle(x_scales)
    w_scales_shuffled = e8m0_shuffle(w_scales)
    x, w = x.cuda(), w.cuda()
    x_scales_shuffled = x_scales_shuffled.cuda()
    w_scales_shuffled = w_scales_shuffled.cuda()
    out = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    return (x, x_scales_shuffled, w.T.contiguous(), w_scales_shuffled, out)


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------


def _compile_vanilla(shape, block):
    """Vanilla MXFP4 GEMM: no manual schedule, compiler defaults."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)
    options.schedule = SchedulingType.NONE
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def _compile_scheduled_4wave(shape, block):
    """Non-preshuffle MXFP4 GEMM with double-buffer schedule, 4 waves."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=4)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)
    options = set_default_run_config(options)
    return wave_compile(options, gemm, schedule)


def _compile_scheduled_8wave(shape, block):
    """Non-preshuffle MXFP4 GEMM with double-buffer schedule, 8 waves."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, num_waves=8)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True)
    options = set_default_run_config(options)
    return wave_compile(options, gemm, schedule)


def _compile_preshuffle_default(shape, block):
    """Preshuffle MXFP4 GEMM with compiler-default scheduling."""
    gemm, options = get_tagged_preshuffle_mxfp4_gemm(shape, block, num_waves=8)
    options.schedule = SchedulingType.NONE
    options = set_default_run_config(options)
    return wave_compile(options, gemm)


def _compile_preshuffle_scheduled_4wave(shape, block):
    """Preshuffle MXFP4 GEMM with double-buffer schedule, 4 waves."""
    gemm, options = get_tagged_preshuffle_mxfp4_gemm(shape, block, num_waves=4)
    schedule = get_preshuffle_dbuf_schedule(use_stagger=False)
    options = set_default_run_config(options)
    return wave_compile(options, gemm, schedule)


def _compile_preshuffle_scheduled_8wave(shape, block):
    """Preshuffle MXFP4 GEMM with double-buffer schedule, 8 waves."""
    gemm, options = get_tagged_preshuffle_mxfp4_gemm(shape, block, num_waves=8)
    schedule = get_preshuffle_dbuf_schedule(use_stagger=True)
    options = set_default_run_config(options)
    return wave_compile(options, gemm, schedule)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vanilla_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Vanilla MXFP4 GEMM with compiler-default scheduling (no manual schedule)."""
    compiled = _compile_vanilla(shape, block)

    if is_debug:
        with open("gemm_mxfp4_vanilla.mlir", "w") as f:
            f.write(compiled.asm)
        print("MLIR written to gemm_mxfp4_vanilla.mlir")

    _run_mxfp_gemm(compiled, shape)
    print("Vanilla MXFP GEMM test passed!")


def test_basic_preshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Preshuffle MXFP4 GEMM with compiler-default scheduling (no manual schedule)."""
    compiled = _compile_preshuffle_default(shape, block)

    if is_debug:
        with open("gemm_mxfp4_preshuffle_basic.mlir", "w") as f:
            f.write(compiled.asm)
        print("MLIR written to gemm_mxfp4_preshuffle_basic.mlir")

    _run_preshuffle_mxfp_gemm(compiled, shape)
    print("Basic preshuffle MXFP GEMM test passed!")


def test_dbuf_4wave_preshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered preshuffle MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_preshuffle_mxfp4_gemm(shape, block, num_waves=4)
    schedule = get_preshuffle_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_preshuffle_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_preshuffle_mxfp_gemm(gemm, shape)
    print("Preshuffle MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_preshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered preshuffle MXFP4 GEMM, 8 waves, with stagger."""
    gemm, options = get_tagged_preshuffle_mxfp4_gemm(shape, block, num_waves=8)
    schedule = get_preshuffle_dbuf_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_preshuffle_mxfp_gemm(gemm, shape)
    print("Preshuffle MXFP GEMM double-buffer 8-wave test passed!")


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

VARIANTS = [
    ("vanilla", _compile_vanilla, _make_inputs_non_preshuffle),
    ("scheduled 4wave", _compile_scheduled_4wave, _make_inputs_non_preshuffle),
    ("scheduled 8wave", _compile_scheduled_8wave, _make_inputs_non_preshuffle),
    (
        "preshuffle (default sched)",
        _compile_preshuffle_default,
        _make_inputs_preshuffle,
    ),
    (
        "preshuffle sched 4wave",
        _compile_preshuffle_scheduled_4wave,
        _make_inputs_preshuffle,
    ),
    (
        "preshuffle sched 8wave",
        _compile_preshuffle_scheduled_8wave,
        _make_inputs_preshuffle,
    ),
]


def run_benchmark(shape=(1024, 1024, 8192), block=(256, 256, 256), warmup=5, iters=20):
    """Compile and benchmark all MXFP4 GEMM variants, printing a timing table."""
    m, n, k = shape
    flops = 2.0 * m * n * k

    print(
        f"MXFP4 GEMM Benchmark  shape=({m}, {n}, {k})  block=({block[0]}, {block[1]}, {block[2]})"
    )
    print(f"  warmup={warmup}  iters={iters}")
    print("-" * 72)
    print(f"{'Variant':<35} {'Time (us)':>10} {'TFLOPs':>10}")
    print("-" * 72)

    for name, compile_fn, make_inputs_fn in VARIANTS:
        try:
            compiled = compile_fn(shape, block)
        except Exception as e:
            print(f"{name:<35} {'COMPILE FAIL':>10}  {e}")
            continue

        inputs = make_inputs_fn(shape)
        try:
            mean_us = _bench_kernel(compiled, inputs, warmup=warmup, iters=iters)
            tflops = (flops / 1e12) / (mean_us / 1e6)
            print(f"{name:<35} {mean_us:>10.1f} {tflops:>10.2f}")
        except Exception as e:
            print(f"{name:<35} {'RUN FAIL':>10}  {e}")

    print("-" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="MXFP4 GEMM preshuffle tests and benchmark"
    )
    parser.add_argument("--test", type=str, help="Name of the test to run")
    parser.add_argument(
        "--list_tests", action="store_true", help="List all available tests"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the test"
    )
    parser.add_argument(
        "--shape", type=str, default=None, help="Shape, e.g. 1024,1024,8192"
    )
    parser.add_argument(
        "--block", type=str, default=None, help="Block size, e.g. 256,256,256"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing all MXFP4 variants",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Benchmark warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="Benchmark measurement iterations"
    )

    args = parser.parse_args()

    if isinstance(args.shape, str):
        args.shape = tuple(map(int, args.shape.split(",")))
    if isinstance(args.block, str):
        args.block = tuple(map(int, args.block.split(",")))

    if args.list_tests:
        list_tests(globals())
        sys.exit(0)

    if args.benchmark:
        shape = args.shape or (1024, 1024, 8192)
        block = args.block or (256, 256, 256)
        run_benchmark(shape=shape, block=block, warmup=args.warmup, iters=args.iters)
        sys.exit(0)

    if not args.test:
        print("Error: --test or --benchmark argument is required")
        print("Use --list_tests to see available tests")
        sys.exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    sys.exit(0 if success else 1)
