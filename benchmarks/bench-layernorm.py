import itertools

import torch
import triton
import triton.language as tl

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

def layernorm_triton(x, weight, bias, eps: float = 1e-6):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    orig_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M, )](
        x_arg, y, weight, bias,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, 
        num_warps=num_warps,
        num_ctas=1
        )
    y_reshape = torch.reshape(y, orig_shape)
    return y_reshape

def get_layernorm_wave_kernel(shape, eps: float = 1e-6):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def layernorm(
        x: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        bias: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        output: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        x_reg = tkw.read(x)
        x_reg = tkw.cast(x_reg, tkl.f32)
        mean = tkw.sum(x_reg, dim=N, block=False) / length_embedding
        mean_bc = tkw.broadcast(mean, [M, N])
        xmm = x_reg - mean_bc
        sq = xmm * xmm
        var = tkw.sum(sq, dim=N, block=False) / length_embedding
        rstd = tkw.sqrt(var + eps_reg)
        rstd_bc = tkw.broadcast(rstd, [M, N])
        x_hat = xmm / rstd_bc
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_bc = tkw.broadcast(w_reg, [M, N])
        b_reg = tkw.read(bias)
        b_reg = tkw.cast(b_reg, tkl.f32)
        b_bc = tkw.broadcast(b_reg, [M, N])
        y = x_hat * w_bc + b_bc
        y = tkw.cast(y, tkl.bf16)
        tkw.write(y, output)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: shape[1],
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        use_buffer_ops=False,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    return wave_compile(options, layernorm)

def layernorm_wave(kernel, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    orig_shape = x.shape
    x_arg = x.reshape(-1, x.shape[-1])
    c = torch.empty_like(x_arg)
    kernel(x, weight, bias, c)
    c_reshape = torch.reshape(c, orig_shape)
    return c_reshape


def layernorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    dtype=torch.bfloat16
):
    w_shape = (x.shape[-1], )
    output = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype=dtype, device=x.device)
    return output


def calculate_diff(batch_size, sentence_length, embedding_dim, dtype=torch.bfloat16):
    x = torch.randn(batch_size, sentence_length, embedding_dim, dtype=dtype, device="cuda")
    w_shape = (x.shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    bias = torch.rand(w_shape, dtype=dtype, device="cuda")

    shape = [x.shape[0] * x.shape[1], x.shape[2]]
    kernel = get_layernorm_wave_kernel(shape)

    output_torch = layernorm_torch(x.clone(), weight.clone(), bias.clone())
    output_wave = layernorm_wave(kernel, x.clone(), weight.clone(), bias.clone())
    output_triton = layernorm_triton(x.clone(), weight.clone(), bias.clone())

    print(f"Torch output={output_torch}")
    print(f"Wave output={output_wave}")
    print(f"Triton output={output_triton}")

    if torch.allclose(output_torch, output_wave, atol=1e-2, rtol=1e-2):
        print("✅ Wave implementation matches") 
    else:
        print("❌ Wave implementation differs")
    
    if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=1e-2):
        print("✅ Triton implementation matches")
    else:
        print("❌ Triton implementation differs")


batch_size_range = [16]
sentence_length_range = [1536]
embedding_dim_range = [576]
configs = list(itertools.product(batch_size_range, sentence_length_range, embedding_dim_range))


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "sentence_length", "embedding_dim",],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["torch", "wave", "triton"],
            line_names=["PyTorch", "Wave", "Triton"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"layernorm-performance",
            args={},
        )
    )
    def benchmark(batch_size, sentence_length, embedding_dim, provider, dtype=torch.bfloat16):
        x = torch.randn(batch_size, sentence_length, embedding_dim, dtype=dtype, device="cuda")
        w_shape = (x.shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device="cuda")
        bias = torch.rand(w_shape, dtype=dtype, device="cuda")

        shape = [x.shape[0] * x.shape[1], x.shape[2]]
        kernel = get_layernorm_wave_kernel(shape)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: layernorm_torch(x.clone(), weight.clone(), bias.clone()),
                quantiles=quantiles,
            )
        elif provider == "wave":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: layernorm_wave(kernel, x.clone(), weight.clone(), bias.clone()),
                quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: layernorm_triton(x.clone(), weight.clone(), bias.clone()),
                quantiles=quantiles
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save layernorm benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(batch_size=16, sentence_length=1536, embedding_dim=576)

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
