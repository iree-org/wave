import itertools
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import nn
from vllm import _custom_ops as vllm_ops


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_zeros,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps: float = 1e-6, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }
 
    fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output


def get_rmsnorm_wave(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    EMB_SIZE = tkl.sym.EMB_SIZE
    TOKENS_PER_WK = tkl.sym.TOKENS_PER_WK

    num_waves = 4
    wave_size = 64
    BLOCK_N = N
    BLOCK_M = TOKENS_PER_WK

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: ELEMS_PER_THREAD * wave_size},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / num_waves)]

    @tkw.wave(constraints)
    def rmsnorm(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
        gamma: tkl.Memory[N, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.bf16],
    ):
        length_embedding = tkl.Register[M, tkl.bf16](EMB_SIZE)
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        lhs_pow = lhs * lhs
        red = tkw.sum(lhs_pow, dim=N, block=True)
        result = red / length_embedding
        rms = tkw.sqrt(result)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = lhs / rms_broad
        gamma_reg = tkw.read(gamma, elements_per_thread=ELEMS_PER_THREAD)
        gamma_broad = tkw.broadcast(gamma_reg, [M, N])
        output = a_scaled * gamma_broad
        tkw.write(output, c, elements_per_thread=ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            TOKENS_PER_WK: 1,
            EMB_SIZE: shape[1],
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        wave_runtime=False,
    )
    options = set_default_run_config(options)
    return wave_compile(options, rmsnorm)


def rmsnorm_wave(
    kernel,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    c = device_zeros(x.shape, dtype=torch.bfloat16)
    kernel(x, weight, c)

    return c


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.bfloat16
    x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None
    wave_kernel = get_rmsnorm_wave(x.shape)

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_wave = rmsnorm_wave(
        wave_kernel,
        x.clone(),
        weight,
        # residual.clone() if residual is not None else None,
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_triton = fused_rmsnorm(
        x.clone(), weight
    )

    if use_residual:
        output_naive = output_naive[0]
        output_wave = output_wave[0]
        output_vllm = output_vllm[0]
        output_triton = output_triton[0]

    print(f"Naive output={output_naive}")
    print(f"Wave output={output_wave}")
    print(f"VLLM output={output_vllm}")
    print(f"Triton output={output_triton}")

    if torch.allclose(
        output_naive, output_wave, atol=1e-2, rtol=1e-2
    ) and torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2) and torch.allclose(output_naive, output_triton, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1]  # [2**i for i in range(0, 7, 2)]
seq_length_range = [2**i for i in range(6, 11, 1)] + [5120]
head_num_range = [1, 32, 48, 128]
configs = list(itertools.product(head_num_range, batch_size_range, seq_length_range))


def get_benchmark(use_residual):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "wave", "vllm", "triton"],
            line_names=["HuggingFace", "Wave", "vLLM", "Triton"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name=f"rmsnorm-performance-{'with' if use_residual else 'without'}-residual",
            args={},
        )
    )
    def benchmark(head_num, batch_size, seq_len, provider):
        dtype = torch.bfloat16
        hidden_size = head_num * 128  # assuming head_dim = 128

        x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        wave_kernel = get_rmsnorm_wave(x.shape)

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "wave":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_rmsnorm(
                    x.clone(),
                    weight,
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_vllm(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_residual", action="store_true", help="Whether to use residual connection"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save rmsnorm benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(
        batch_size=4, seq_len=128, hidden_size=4096, use_residual=args.use_residual
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
