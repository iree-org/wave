"""
MXFP4 GEMM Example: Unshuffled vs Shuffled Scales

Tests two MXFP4 GEMM implementations:
1. Unshuffled: Scales in normal [M, K//32] or [N, K//32] layout
2. Shuffled: Scales pre-shuffled using e8m0_shuffle for hardware efficiency

Both kernels are verified against a PyTorch reference implementation.
"""

import torch
import argparse

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.constraints import ScaledMMAType

SCALE_GROUP_SIZE = 32  # Hardware constant: 1 scale per 32 data elements


def e8m0_shuffle(scale):
    """
    Shuffle the scale tensor for e8m0 format.

    This particular shuffle is taken from
    https://github.com/ROCm/rocm-libraries/blob/4348901528fe100a84975b89c247eece553a2a2d/shared/mxdatagenerator/lib/include/mxDataGenerator/PreSwizzle.hpp#L403

    The e8m0_shuffle operation transforms a matrix with shape (m, n) as follows:
    1. Pads to shape ((m+255)//256*256, (n+7)//8*8)
    2. Reshapes to (sm//32, 2, 16, sn//8, 2, 4)
    3. Permutes dimensions: (0, 3, 5, 2, 4, 1)
    4. Flattens back to (sm, sn)

    Args:
        scale: A 2D tensor to be shuffled

    Returns:
        Shuffled tensor with the same padded shape
    """
    if scale is None:
        return scale
    if scale.dtype == torch.float32:
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


def generate_mxfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random MXFP4 inputs for scaled GEMM."""
    M, N, K = shape
    torch.manual_seed(5)

    # Generate packed MXFP4 data (2 values per byte)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)

    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)

    # Generate E8M0 scales (random values near 1.0)
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )

    x_scales = x_scales.T.contiguous()  # [M, K//32]
    w_scales = w_scales.T.contiguous()  # [N, K//32]

    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert packed MXFP4 (e2m1fn) values to float32."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF  # Low nibble
    x[:, 1::2] = x[:, 1::2] >> 4  # High nibble

    mxfp4_lut = [
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
    mxfp4_f32 = torch.tensor(mxfp4_lut, dtype=torch.float32, device=x.device)
    return mxfp4_f32[x.long()]


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 (exponent-only) scale values to float32."""
    x_f32 = 2 ** ((x.to(torch.float32) - 127))
    x_f32[x == 255] = float("nan")
    return x_f32


def reference_mxfp4_gemm(
    x: torch.Tensor, w: torch.Tensor, x_scales: torch.Tensor, w_scales: torch.Tensor
) -> torch.Tensor:
    """PyTorch reference implementation for scaled MXFP4 GEMM: C = (x * x_scales) @ (w * w_scales)^T"""
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)

    x_scales_expanded = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = e8m0_to_f32(x_scales_expanded)

    w_scales_expanded = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = e8m0_to_f32(w_scales_expanded)

    x_scaled = x_f32 * x_scales_f32
    w_scaled = w_f32 * w_scales_f32

    return torch.mm(x_scaled, w_scaled.T)


def get_vanilla_kernel():
    """Return the vanilla (unshuffled) MXFP4 GEMM kernel definition."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 1),
        tkw.WaveConstraint(N, BLOCK_N / 4),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    @tkw.wave(constraints)
    def mxfp4_gemm_vanilla(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)

            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)

            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    return mxfp4_gemm_vanilla


def get_preshuffle_kernel():
    """Return the pre-shuffled MXFP4 GEMM kernel definition with IndexMapping for shuffled scales."""
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 1),
        tkw.WaveConstraint(N, BLOCK_N / 4),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=ScaledMMAType.F32_16x16x128_F8F6F4,
        ),
    ]

    # Create IndexMapping for shuffled A scales
    # The e8m0_shuffle coordinate transformation maps logical (K, M) iterators
    # to physical shuffled memory layout
    i = tkw.IndexMapping.iterator(0)  # K iterator
    j = tkw.IndexMapping.iterator(1)  # M iterator

    a_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            M: (
                (
                    (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (i // 8) * 256
                    + ((i % 8) % 4) * 64
                    + ((j % 32) % 16) * 4
                    + (((i % 8) // 4) * 2)
                    + ((j % 32) // 16)
                )
                // K_SCALE_SHUFFLED
            ),
            K: (
                (
                    (j // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (i // 8) * 256
                    + ((i % 8) % 4) * 64
                    + ((j % 32) % 16) * 4
                    + (((i % 8) // 4) * 2)
                    + ((j % 32) // 16)
                )
                % K_SCALE_SHUFFLED
            ),
        },
        outputs={
            K: i,
            M: j,
        },
    )

    # Create IndexMapping for shuffled B scales
    k = tkw.IndexMapping.iterator(0)  # K iterator
    n = tkw.IndexMapping.iterator(1)  # N iterator

    b_scale_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: (
                (
                    (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (k // 8) * 256
                    + ((k % 8) % 4) * 64
                    + ((n % 32) % 16) * 4
                    + (((k % 8) // 4) * 2)
                    + ((n % 32) // 16)
                )
                // K_SCALE_SHUFFLED
            ),
            K: (
                (
                    (n // 32) * ((K_SCALE_SHUFFLED // 8) * 256)
                    + (k // 8) * 256
                    + ((k % 8) % 4) * 64
                    + ((n % 32) % 16) * 4
                    + (((k % 8) // 4) * 2)
                    + ((n % 32) // 16)
                )
                % K_SCALE_SHUFFLED
            ),
        },
        outputs={
            K: k,
            N: n,
        },
    )

    # TODO: The read side matches aiter. The only difference is on the write side for a_scale
    #     (how data gets into LDS). Wave uses 8 byte loads + 8 byte stores, while AITER
    #     uses 2 DMA dword loads.
    # The analysis report covers what's needed to close that gap.
    @tkw.wave(constraints)
    def mxfp4_gemm_preshuffle(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale, mapping=a_scale_mapping)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)

            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale, mapping=b_scale_mapping)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)

            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    return mxfp4_gemm_preshuffle


def run_all_tests():
    """Run both vanilla and pre-shuffled tests and compare results."""
    m, n, k = 1024, 1024, 8192  # 2048, 57344, 16384 #512, 512, 2048
    block_m, block_n, block_k = 256, 256, 256

    print("=" * 70)
    print("MXFP4 GEMM COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Problem size: M={m}, N={n}, K={k}")
    print(f"Block sizes: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}")

    # Define symbolic dimensions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Calculate shuffled dimensions for pre-shuffle kernel
    k_scale_shuffled = (((k // 32) + 7) // 8) * 8
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    # Get kernel definitions
    print("\nGetting kernel definitions...")
    vanilla_kernel = get_vanilla_kernel()
    preshuffle_kernel = get_preshuffle_kernel()

    # Set up hyperparameters (shared by both kernels)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        K_SCALE_SHUFFLED: k_scale_shuffled,
    }

    # Compile options (shared by both kernels)
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_global_to_shared=True,
        use_buffer_ops=True,
    )
    options = set_default_run_config(options)

    # Compile both kernels
    compiled_vanilla = wave_compile(options, vanilla_kernel)

    compiled_preshuffle = wave_compile(options, preshuffle_kernel)

    # Generate test data
    x, w, x_scales, w_scales = generate_mxfp4_inputs(
        (m, n, k), device=torch.device("cpu")
    )

    # Compute PyTorch reference
    torch_result = reference_mxfp4_gemm(x, w, x_scales, w_scales)

    # Shuffle scales for pre-shuffle kernel
    x_scales_shuffled = e8m0_shuffle(x_scales)
    w_scales_shuffled = e8m0_shuffle(w_scales)

    # Move data to GPU
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    x_scales_gpu = x_scales.cuda()
    w_scales_gpu = w_scales.cuda()
    x_scales_shuffled_gpu = x_scales_shuffled.cuda()
    w_scales_shuffled_gpu = w_scales_shuffled.cuda()

    # Run vanilla kernel
    print("\n" + "=" * 60)
    print("TEST 1: Vanilla MXFP4 GEMM")
    print("=" * 60)
    print("Running vanilla Wave kernel...")
    c_vanilla_gpu = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    compiled_vanilla(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_vanilla_gpu)
    wave_vanilla_result = c_vanilla_gpu.cpu()

    print("Verifying vanilla results...")
    try:
        torch.testing.assert_close(
            torch_result, wave_vanilla_result, rtol=1e-3, atol=1e-3, check_dtype=False
        )
        print("✓ VANILLA TEST PASSED! Results match PyTorch reference.")
    except AssertionError as e:
        print("✗ VANILLA TEST FAILED!")
        print(f"Error: {e}")
        max_diff = torch.max(torch.abs(torch_result - wave_vanilla_result))
        mean_diff = torch.mean(torch.abs(torch_result - wave_vanilla_result))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        print(f"Reference output range: [{torch_result.min()}, {torch_result.max()}]")
        print(
            f"Wave output range: [{wave_vanilla_result.min()}, {wave_vanilla_result.max()}]"
        )

    # Run pre-shuffle kernel
    print("\n" + "=" * 60)
    print("TEST 2: Pre-Shuffled MXFP4 GEMM")
    print("=" * 60)
    print("Running pre-shuffle Wave kernel...")
    c_preshuffle_gpu = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    compiled_preshuffle(
        x_gpu, x_scales_shuffled_gpu, w_gpu, w_scales_shuffled_gpu, c_preshuffle_gpu
    )
    wave_preshuffle_result = c_preshuffle_gpu.cpu()

    print("Verifying pre-shuffle results...")
    try:
        torch.testing.assert_close(
            torch_result,
            wave_preshuffle_result,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
        )
        print("✓ PRE-SHUFFLE TEST PASSED! Results match PyTorch reference.")
    except AssertionError as e:
        print("✗ PRE-SHUFFLE TEST FAILED!")
        print(f"Error: {e}")
        max_diff = torch.max(torch.abs(torch_result - wave_preshuffle_result))
        mean_diff = torch.mean(torch.abs(torch_result - wave_preshuffle_result))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        print(f"Reference output range: [{torch_result.min()}, {torch_result.max()}]")
        print(
            f"Wave output range: [{wave_preshuffle_result.min()}, {wave_preshuffle_result.max()}]"
        )

    # Final comparison: verify both Wave results are identical
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: Vanilla vs Pre-Shuffled Wave Results")
    print("=" * 60)
    try:
        torch.testing.assert_close(
            wave_vanilla_result,
            wave_preshuffle_result,
            rtol=1e-6,
            atol=1e-6,
            check_dtype=False,
        )
        print("✓ Both Wave kernels produce IDENTICAL results!")
    except AssertionError as e:
        print("✗ Wave kernels produce DIFFERENT results!")
        print(f"Error: {e}")
        max_diff = torch.max(torch.abs(wave_vanilla_result - wave_preshuffle_result))
        mean_diff = torch.mean(torch.abs(wave_vanilla_result - wave_preshuffle_result))
        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")


def run_benchmark(
    kernel_type="vanilla",
    m=2048,
    n=57344,
    k=16384,
    block_m=256,
    block_n=256,
    block_k=256,
    warmup_iters=50,
    bench_iters=100,
):
    """
    Benchmark a single kernel configuration.

    Args:
        kernel_type: Either "vanilla" or "preshuffle"
        m, n, k: Problem dimensions
        block_m, block_n, block_k: Block dimensions
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations

    Returns:
        Dictionary containing benchmark results
    """
    print("=" * 70)
    print(f"BENCHMARKING: {kernel_type.upper()} MXFP4 GEMM")
    print("=" * 70)
    print(f"Problem size: M={m}, N={n}, K={k}")
    print(f"Block sizes: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")

    # Define symbolic dimensions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Get the appropriate kernel
    if kernel_type == "vanilla":
        kernel = get_vanilla_kernel()
    elif kernel_type == "preshuffle":
        kernel = get_preshuffle_kernel()
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # Set up hyperparameters
    k_scale_shuffled = (((k // 32) + 7) // 8) * 8
    K_SCALE_SHUFFLED = tkl.sym.K_SCALE_SHUFFLED

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        K_SCALE_SHUFFLED: k_scale_shuffled,
    }

    dump_dir = f"tmp_files/mxfp4_preshuffle_scale_{kernel_type}_{m}x{n}x{k}/"

    # Compile kernel
    print("\nCompiling kernel...")
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_global_to_shared=True,
        use_buffer_ops=True,
        dump_intermediates=dump_dir,
    )
    options = set_default_run_config(options)
    compiled_kernel = wave_compile(options, kernel)

    # Generate test data
    print("Generating test data...")
    x, w, x_scales, w_scales = generate_mxfp4_inputs(
        (m, n, k), device=torch.device("cpu")
    )

    # Shuffle scales if needed
    if kernel_type == "preshuffle":
        x_scales = e8m0_shuffle(x_scales)
        w_scales = e8m0_shuffle(w_scales)

    # Move data to GPU
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    x_scales_gpu = x_scales.cuda()
    w_scales_gpu = w_scales.cuda()
    c_gpu = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Warmup
    print(f"\nWarming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        compiled_kernel(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_gpu)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({bench_iters} iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(bench_iters):
        compiled_kernel(x_gpu, x_scales_gpu, w_gpu, w_scales_gpu, c_gpu)
    end_event.record()
    torch.cuda.synchronize()

    # Calculate metrics
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / bench_iters
    flops = 2 * m * n * k  # GEMM FLOPs
    tflops = (flops / (avg_time_ms * 1e-3)) / 1e12

    results = {
        "kernel_type": kernel_type,
        "m": m,
        "n": n,
        "k": k,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "avg_time_ms": avg_time_ms,
        "total_time_ms": elapsed_ms,
        "tflops": tflops,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
    }

    # Print results (same format as 7.3)
    print(f"  Problem size: M={m}, N={n}, K={k}")
    print(f"  Avg time:     {avg_time_ms:.3f} ms  ({bench_iters} iters)")
    print(f"  TFLOP/s:      {tflops:.2f}")
    print(f"  Dump dir:     {dump_dir}")

    # Quick NaN check
    result_cpu = c_gpu.cpu()
    nan_mask = torch.isnan(result_cpu)
    nan_count = nan_mask.sum().item()
    print(
        f"  NaN count:    {nan_count}/{result_cpu.numel()} ({100*nan_count/result_cpu.numel():.1f}%)"
    )
    if nan_count > 0:
        nan_rows = torch.where(nan_mask.any(dim=1))[0]
        nan_cols = torch.where(nan_mask.any(dim=0))[0]
        print(f"  NaN rows:     {nan_rows.tolist()[:20]}")
        print(f"  NaN cols:     {nan_cols.tolist()[:20]}")
    valid = result_cpu[~nan_mask]
    if valid.numel() > 0:
        print(f"  Valid range:  [{valid.min():.4f}, {valid.max():.4f}]")

    return results


def run_all_benchmarks():
    """Run benchmarks for both vanilla and preshuffle kernels and compare."""
    # Default configuration
    m, n, k = 512, 512, 2048
    block_m, block_n, block_k = 256, 256, 256
    warmup_iters = 10
    bench_iters = 100

    vanilla_results = run_benchmark(
        kernel_type="vanilla",
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )

    print("\n\n")

    preshuffle_results = run_benchmark(
        kernel_type="preshuffle",
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )

    # Comparison
    print("\n\n")
    print("=" * 70)
    print("COMPARISON: Vanilla vs Pre-Shuffled")
    print("=" * 70)
    print(
        f"Vanilla avg time:      {vanilla_results['avg_time_ms']:.4f} ms ({vanilla_results['tflops']:.2f} TFLOPS)"
    )
    print(
        f"Pre-shuffle avg time:  {preshuffle_results['avg_time_ms']:.4f} ms ({preshuffle_results['tflops']:.2f} TFLOPS)"
    )

    speedup = vanilla_results["avg_time_ms"] / preshuffle_results["avg_time_ms"]
    if speedup > 1.0:
        print(
            f"\nPre-shuffle is {speedup:.2f}x FASTER than vanilla ({(speedup-1)*100:.1f}% improvement)"
        )
    elif speedup < 1.0:
        print(
            f"\nPre-shuffle is {1/speedup:.2f}x SLOWER than vanilla ({(1-speedup)*100:.1f}% regression)"
        )
    else:
        print("\n= Performance is identical")
    print("=" * 70)

    return vanilla_results, preshuffle_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MXFP4 GEMM: Preshuffle Scale Benchmark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="preshuffle",
        choices=["test", "bench", "vanilla", "preshuffle"],
        help="Mode: test (correctness), bench (both), vanilla, preshuffle (default)",
    )
    parser.add_argument("--M", type=int, default=2048, help="M dimension")
    parser.add_argument("--N", type=int, default=57344, help="N dimension")
    parser.add_argument("--K", type=int, default=16384, help="K dimension")
    parser.add_argument("--block_m", type=int, default=256, help="Block M dimension")
    parser.add_argument("--block_n", type=int, default=256, help="Block N dimension")
    parser.add_argument("--block_k", type=int, default=256, help="Block K dimension")
    parser.add_argument(
        "--warmup", type=int, default=50, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    if args.mode == "test":
        run_all_tests()
    elif args.mode == "bench":
        run_all_benchmarks()
    else:
        run_benchmark(
            kernel_type=args.mode,
            m=args.M,
            n=args.N,
            k=args.K,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            warmup_iters=args.warmup,
            bench_iters=args.iters,
        )
