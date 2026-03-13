"""
Tests for the MOE reduce sum kernel.

This kernel reduces a 3D tensor [B, K, D] over the K dimension,
weighted by a 2D tensor [B, K]. It uses source/target transpose
to make K the fast dimension for reduction.
"""

import pytest
import torch

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.templates.moe import get_moe_reduce_sum_kernel
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config


def run_moe_reduce_sum_test(b_size: int, k_size: int, d_size: int, dtype: torch.dtype):
    """
    Run MOE reduce sum kernel and compare against reference.

    Args:
        b_size: Batch dimension (number of tokens)
        k_size: Reduction dimension (topk)
        d_size: Output dimension (hidden size)
        dtype: Data type for computation
    """
    # Get kernel and hyperparams
    wave_dtype = tkl.f32 if dtype == torch.float32 else tkl.f16
    kernel, subs = get_moe_reduce_sum_kernel(b_size, k_size, d_size, wave_dtype)

    options = WaveCompileOptions(subs=subs)
    options = set_default_run_config(options)
    kernel = wave_compile(options, kernel)

    # Create test data
    a = torch.randn(b_size, k_size, d_size, dtype=dtype, device="cuda")
    weights = torch.randn(b_size, k_size, dtype=dtype, device="cuda")
    c = torch.zeros(b_size, d_size, dtype=dtype, device="cuda")

    # Reference computation: (a * weights.unsqueeze(-1)).sum(dim=1)
    weights_broadcasted = weights.unsqueeze(-1)  # [B, K, 1]
    ref = (a * weights_broadcasted).sum(dim=1)  # [B, D]

    # Run kernel
    kernel(a, weights, c)

    # Compare results
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2

    torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)


# Test combined sizes similar to real MoE scenarios
@pytest.mark.parametrize(
    "b_size,k_size,d_size",
    [
        (32, 2, 64),  # Small: 32 tokens, topk=2, hidden=64
        (64, 2, 128),  # Medium: 64 tokens, topk=2, hidden=128
        (128, 4, 256),  # Large: 128 tokens, topk=4, hidden=256
        (256, 8, 512),  # XLarge: 256 tokens, topk=8, hidden=512
        (256, 64, 512),  # XLarge: 256 tokens, topk=8, hidden=512
        (256, 128, 512),  # XLarge: 256 tokens, topk=8, hidden=512
        (256, 250, 512),  # XLarge: 256 tokens, topk=8, hidden=512
    ],
)
def test_moe_reduce_sum_combined(b_size: int, k_size: int, d_size: int):
    """Test reduce sum with combined size configurations."""
    run_moe_reduce_sum_test(
        b_size=b_size, k_size=k_size, d_size=d_size, dtype=torch.float32
    )


if __name__ == "__main__":
    test_moe_reduce_sum_combined(b_size=256, k_size=250, d_size=512)
