import torch

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.constraints import ScaledMMAType

from utils import parse_args, list_tests, run_test


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def preshuffle_b_aiter(b: torch.Tensor) -> torch.Tensor:
    """
    Preshuffle B using the aiter ``shuffle_weight`` permutation (see
    ``preshuffle_B_analysis.md``).

    Within each 16-row x 32-byte tile the original layout is
    ``[n, k_sub, k_elem]``  (shape ``[16, 2, 16]``).
    The permutation reorders it to ``[k_sub, n, k_elem]``, so that a
    contiguous 256-byte read fetches one K-chunk for all 16 N-rows.

    For the full MFMA B operand (16 N x 128 K FP4 = 1024 packed bytes),
    four such 256-byte chunks are contiguous, giving::

        addr(lane) = tile_base + lane * 16   (64 lanes x 16 B = 1024 B contiguous)

    Equivalent to ``aiter/ops/shuffle.py::shuffle_weight`` with
    ``layout=(16, 16), use_int4=False``:

        x.view(N//16, 16, K_packed//32, 2, 16)
         .permute(0, 2, 3, 1, 4)
         .contiguous()
         .view(N, K_packed)

    Input:
      - b: ``[N, K/2]`` packed MXFP4 bytes (``uint8``)
    Output:
      - b_ps: same shape/dtype, rearranged for PRESHUFFLEB
    """
    if b.dtype is not torch.uint8:
        raise TypeError(f"Expected uint8 packed FP4 weights, got {b.dtype}")
    if b.ndim != 2:
        raise ValueError(f"Expected 2D [N, K/2] tensor, got shape={tuple(b.shape)}")

    N, K_packed = b.shape
    if N % 16 != 0:
        raise ValueError(f"N ({N}) must be divisible by 16")
    if K_packed % 32 != 0:
        raise ValueError(f"K/2 ({K_packed}) must be divisible by 32")

    # 5D view:  [n_blk, n, k_blk, k_sub, k_elem]
    #            N//16   16  K_packed//32  2    16
    b_5d = b.view(N // 16, 16, K_packed // 32, 2, 16)

    # permute: swap n (dim 1) with (k_blk, k_sub) (dims 2,3)
    #   [n_blk, n, k_blk, k_sub, k_elem]  →  [n_blk, k_blk, k_sub, n, k_elem]
    b_ps = b_5d.permute(0, 2, 3, 1, 4).contiguous()

    return b_ps.view(N, K_packed)


def generate_gemm_afp4wfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N, K = shape
    torch.manual_seed(5)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)

    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)

    # Matches other examples: w is stored transposed first.
    w = w.T

    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(
    x: torch.Tensor, w: torch.Tensor, x_scales: torch.Tensor, w_scales: torch.Tensor
) -> torch.Tensor:
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T).T
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_f32 = x_f32 * e8m0_to_f32(x_scales)
    w_f32 = w_f32 * e8m0_to_f32(w_scales).T
    return torch.mm(x_f32, w_f32)


def test_preshuffleB_direct_global_b_8wave(
    is_debug: bool = False,
    shape: tuple[int, int, int] = (1024, 1024, 8192),
    block: tuple[int, int, int] = (256, 256, 256),
):
    """
    GEMM with:
    - A + A_scales via LDS (GatherToLDS) for bandwidth/latency hiding
    - B + B_scales via direct GLOBAL reads (no LDS) using preshuffled B layout
    """
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    m_iter = tkl.sym.m_iter
    n_iter = tkl.sym.n_iter
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
    K_PACKED = tkl.sym.K_PACKED  # K/2 (byte count in K dim)

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.IteratorBindings({m_iter: M, n_iter: N})]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Same pattern as the scale-shuffle IndexMapping:
    #   compute a flat byte offset in the preshuffled layout,
    #   then divmod by K_PACKED to get the 2D (N, K/2) physical position.
    #
    # After aiter shuffle_weight the 16×32-byte tile is [k_sub, n, k_elem].
    # Flat byte offset within one n_blk:
    #   within = k_blk*512 + k_sub*256 + n*16 + k_elem
    # where k_blk = k_byte//32, k_sub = (k_byte//16)%2, k_elem = k_byte%16
    #
    # IndexMapping iterators are in the memory-dimension space:
    #   n_it indexes N  (0..N-1)
    #   k_it indexes K/2 (0..K_packed-1, byte indices)
    n_it = tkw.IndexMapping.iterator(0)  # N dimension
    k_it = tkw.IndexMapping.iterator(1)  # K/2 dimension (byte index)

    # TODO: simplify below, if possible. It is correct but can be simplified.
    # aiter shuffle_weight reshapes each 16×32-byte tile as [k_sub, n, k_elem]
    # where k_sub is in {0,1} selects the 16-byte half,  n is in [0,15],  k_elem is in [0,15].
    # Decompose k_it (byte index into K/2) into these sub-fields:
    #   k_blk  = k_it // 32       which 32-byte block (strides by 512 = 32*16 bytes)
    #   k_sub  = (k_it // 16) % 2 upper/lower 16-byte half within the block (stride 256 = 16*16)
    #   k_elem = k_it % 16        byte position within the 16-byte element group
    # n_it % 16 selects the row within a 16-row N-block (stride 16 bytes).
    #
    # Combining: flat byte offset within one N-block of 16 rows =
    #   k_blk*512 + k_sub*256 + (n_it%16)*16 + k_elem
    #
    within_nblk = (
        (k_it // 32) * 512 + ((k_it // 16) % 2) * 256 + (n_it % 16) * 16 + k_it % 16
    )

    b_preshuffle_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: (n_it // 16) * 16 + within_nblk // K_PACKED,
            K: within_nblk % K_PACKED,
        },
        outputs={N: n_it, K: k_it},
        contiguous=True,
    )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE_A, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE_A, tkl.i8],
        # B matrix stays in global memory; preshuffled layout enables contiguous wave reads.
        b: tkl.Memory[N, K / 2, GLOBAL_ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn, tag="bitcast_a")
            a_scale_reg = tkw.read(a_scale, tag="read_a_scale")
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu, tag="bitcast_a_scale")

            b_reg = tkw.read(b, mapping=b_preshuffle_mapping, tag="read_b")
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn, tag="bitcast_b")
            b_scale_reg = tkw.read(b_scale, tag="read_b_scale")
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu, tag="bitcast_b_scale")

            acc = tkw.scaled_mma(
                a_reg, a_scale_reg, b_reg, b_scale_reg, acc, tag="scaled_mma"
            )
            return acc

        tkw.write(repeat, c)

    @wave_schedule.wave_schedule()
    def preshuffleB_schedule():
        k_loop = tkw.get_node_by_tag("k_loop")

        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        all_read_a_scale = tkw.get_node_by_tag("read_a_scale")
        global_to_shared_a_scale = tkw.filter_nodes(
            all_read_a_scale, node_type=tkw.GatherToLDS
        )
        shared_load_a_scale = tkw.filter_nodes(all_read_a_scale, node_type=tkw.Read)

        # B: direct global reads (no GatherToLDS)
        all_read_b = tkw.get_node_by_tag("read_b")
        global_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        all_read_b_scale = tkw.get_node_by_tag("read_b_scale")
        global_load_b_scale = tkw.filter_nodes(all_read_b_scale, node_type=tkw.Read)

        bitcast_a = tkw.get_node_by_tag("bitcast_a")
        bitcast_a_scale = tkw.get_node_by_tag("bitcast_a_scale")
        bitcast_b = tkw.get_node_by_tag("bitcast_b")
        bitcast_b_scale = tkw.get_node_by_tag("bitcast_b_scale")
        scaled_mma = tkw.get_node_by_tag("scaled_mma")

        pipeline_loop = tkw.pipeline(k_loop)
        with pipeline_loop as pl:
            # Stage 0: async global->shared for A/A_scale only
            pl.set_stage([(global_to_shared_a, global_to_shared_a_scale), (), ()])
            # Stage 1: shared loads for A, direct global loads for B + compute
            pl.set_stage(
                [
                    (
                        shared_load_a,
                        shared_load_a_scale,
                        global_load_b,
                        global_load_b_scale,
                    ),
                    (bitcast_a, bitcast_a_scale, bitcast_b, bitcast_b_scale),
                    (scaled_mma,),
                ]
            )

        loop_global_to_shared = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        ) + tkw.filter_nodes(global_to_shared_a_scale, subgraph=pipeline_loop.KERNEL)

        loop_shared_load_a = tkw.filter_nodes(
            shared_load_a, subgraph=pipeline_loop.KERNEL
        )
        loop_shared_load_a_scale = tkw.filter_nodes(
            shared_load_a_scale, subgraph=pipeline_loop.KERNEL
        )
        loop_global_load_b = tkw.filter_nodes(
            global_load_b, subgraph=pipeline_loop.KERNEL
        )
        loop_global_load_b_scale = tkw.filter_nodes(
            global_load_b_scale, subgraph=pipeline_loop.KERNEL
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

        # Partition by K to build two half-iterations.
        loop_scaled_mma_0, loop_scaled_mma_1 = tkw.partition_by_dim(
            loop_scaled_mma, dim=K, num_partitions=2
        )
        loop_shared_load_a_0, loop_shared_load_a_1 = tkw.partition_by_dim(
            loop_shared_load_a, dim=K, num_partitions=2
        )
        loop_shared_load_a_scale_0, loop_shared_load_a_scale_1 = tkw.partition_by_dim(
            loop_shared_load_a_scale, dim=K, num_partitions=2
        )
        loop_global_load_b_0, loop_global_load_b_1 = tkw.partition_by_dim(
            loop_global_load_b, dim=K, num_partitions=2
        )
        loop_global_load_b_scale_0, loop_global_load_b_scale_1 = tkw.partition_by_dim(
            loop_global_load_b_scale, dim=K, num_partitions=2
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

        clusters = [
            # Cluster 1: first half loads + bitcasts + issue async gathers for next iter
            tkw.cluster(
                [
                    loop_shared_load_a_0,
                    loop_shared_load_a_scale_0,
                    loop_global_load_b_0,
                    loop_global_load_b_scale_0,
                    loop_bitcast_a_0,
                    loop_bitcast_a_scale_0,
                    loop_bitcast_b_0,
                    loop_bitcast_b_scale_0,
                    tkw.SchedulingBarrier([]),
                    loop_global_to_shared,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ]
            ),
            # Cluster 2: first half MMAs with higher priority; wait for async gathers
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=independent_global_count),
                    tkw.SchedulingBarrier([]),
                ]
            ),
            # Cluster 3: second half loads + bitcasts
            tkw.cluster(
                [
                    loop_shared_load_a_1,
                    loop_shared_load_a_scale_1,
                    loop_global_load_b_1,
                    loop_global_load_b_scale_1,
                    loop_bitcast_a_1,
                    loop_bitcast_a_scale_1,
                    loop_bitcast_b_1,
                    loop_bitcast_b_scale_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWaitBarrier(load=0),
                    tkw.SchedulingBarrier([]),
                ]
            ),
            # Cluster 4: second half MMAs
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    loop_scaled_mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ]
            ),
        ]

        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)
        tkw.stagger(pipeline_loop.KERNEL)

    hyperparams = {
        ADDRESS_SPACE_A: SHARED_ADDRESS_SPACE,
        BLOCK_M: block[0],
        BLOCK_N: block[1],
        BLOCK_K: block[2],
        M: shape[0],
        N: shape[1],
        K: shape[2],
        K_PACKED: shape[2] // 2,  # K / 2 (byte count)
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
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
        use_buffer_ops=True,
        dump_intermediates="tmp_files/mxfp4_wave8_preshuffleB_schedule/", # uncomment to dump intermediate files
    )
    options = set_default_run_config(options)
    compiled = wave_compile(options, gemm, preshuffleB_schedule)

    with open("tmp_files/mxfp4_wave8_preshuffleB_schedule/temp.mlir", "w") as f:
        f.write(compiled.asm)

    # Testing
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(
        shape, device=torch.device("cpu")
    )

    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    # Kernel expects B as [N, K/2]. w is [K/2, N], so transpose back.
    w_t = w.T.contiguous()
    w_t_ps = preshuffle_b_aiter(w_t)

    x = x.cuda()
    x_scales = x_scales.cuda()
    w_t_ps = w_t_ps.cuda()
    w_scales = w_scales.cuda()

    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=torch.float32, device="cuda")

    compiled(x, x_scales, w_t_ps, w_scales, out)

    torch.testing.assert_close(torch_out, out.cpu(), check_dtype=False)
    print("PreshuffleB direct-global-B scheduled GEMM test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
