import torch
import pytest

# Handle both direct execution and pytest imports
try:
    from .common.utils import require_e2e
except ImportError:
    # When running directly, create a no-op decorator
    # (wanted to be able to run the test file directly)
    def require_e2e(func):
        return func


torch.manual_seed(42)


def get_mxfp6_e2m3_lut():
    """64-value LUT for E2M3 (1-2-3) MXFP6."""
    lut = torch.zeros(64, dtype=torch.float32)
    for i in range(64):
        sign = -1.0 if (i >> 5) & 0x1 else 1.0
        exponent = (i >> 3) & 0x3  # 2 bits
        mantissa = i & 0x7  # 3 bits

        if exponent == 0:
            # Subnormal: val = sign * (mantissa / 8.0) * 2^(0)
            val = sign * (mantissa / 8.0)
        else:
            # Normal: val = sign * (1.0 + mantissa / 8.0) * 2^(exponent - 1)
            # Exponent 1 -> 2^0, Exponent 2 -> 2^1, Exponent 3 -> 2^2 (Value 4.0+)
            val = sign * (1.0 + mantissa / 8.0) * (2 ** (exponent - 1))
        lut[i] = val
    return lut


MXFP6_E2M3_LUT = get_mxfp6_e2m3_lut()
MAX_E2M3 = 8.0  # power-of-two ceiling of (1 + 7/8) * 2^2 = 7.5 -> 8.0


# ---------------------------------------------------------------------------
# Reference MXFP quantizer (pure PyTorch, no Brevitas dependency)
# Used as ground truth for validating the LUT-based pack/unpack path.
# ---------------------------------------------------------------------------


def safe_frexp(x: torch.Tensor) -> torch.Tensor:
    """torch.frexp returns unbiased exponent 0 for 0.0, which is not what we want."""
    if x.is_cuda and x.dtype not in (torch.float32, torch.float16):
        x = x.float()
    return torch.where(
        x == 0.0, torch.tensor(-126, dtype=torch.int32), x.frexp().exponent - 1
    )


class MXFP:
    """
    MXFP - Quantize OCP MXFP floating point types.
    A type is defined as ebits, mbits, bias, and inf/nan handling.
    """

    CONFIG = dict(
        e5m2=(5, 2, 15, "ieee"),
        e4m3=(4, 3, 7, "fn"),
        e3m2=(3, 2, 3, "fnuz"),
        e2m3=(2, 3, 1, "fnuz"),
        e2m1=(2, 1, 1, "fnuz"),
    )

    def __init__(self, name, tile_size=32):
        self.name = name.lower()
        assert self.name in self.CONFIG
        self.ebits, self.mbits, self.bias, self.infnan = self.CONFIG[self.name]
        self.tile_size = tile_size

    @property
    def emax(self) -> int:
        return 2**self.ebits - 1 - self.bias - int(self.infnan == "ieee")

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def maxval(self) -> float:
        return 2**self.emax * (
            2.0 - (1 + int(self.infnan == "fn")) * 2 ** (-self.mbits)
        )

    def quantize(self, tensor: torch.Tensor, axis: int = -1):
        exp = safe_frexp(tensor)
        shared = exp.amax(axis, keepdim=True)
        scale = (
            self.mbits - (shared - exp - (self.emax - self.emin)).clamp_min(0) - exp
        ).exp2()
        maxval = self.maxval * (shared - self.emax).exp2()
        return ((tensor * scale).round() / scale).clamp(-maxval, maxval)


def f32_to_mxfp6(x, lut, block_size=32):
    """
    Packs FP32 tensor into MXFP6 E2M3 format.
    x: [N, K]
    Returns: packed_uint8 [N, (K*6)/8], scales [N, K/block_size]
    """

    # Step 1: Quantize using the analytical MXFP reference quantizer.
    # This uses torch.round() (round-half-to-even) for correct rounding behavior,
    # exactly matching the Brevitas/OCP MX spec.
    x_reshaped = x.view(-1, block_size)
    _quantizer = MXFP("e2m3")
    x_q = _quantizer.quantize(x_reshaped)

    # Step 2: Recover the block scales (power-of-two, per OCP MX spec).
    # Scale = 2^(shared - emax), where shared is the max per-element exponent
    # in the block and emax = 2 for E2M3.
    shared = safe_frexp(x_reshaped).amax(dim=1, keepdim=True)
    scales = torch.pow(2.0, (shared - 2).float())

    # Step 3: Normalize quantized values to LUT range and map to 6-bit indices.
    # Since x_q is already quantized to E2M3-representable values, dividing by
    # the block scale yields values that are exactly in the LUT.
    x_norm_q = x_q / scales
    dists = torch.abs(x_norm_q.view(-1).unsqueeze(-1) - lut.unsqueeze(0))
    indices = torch.argmin(dists, dim=-1).to(torch.uint8)

    # pack bytes using indexing
    v = indices.view(-1, 4)
    v0, v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]

    # b0: v0 [7:2], v1_head [1:0]
    b0 = (v0 << 2) | (v1 >> 4)
    # b1: v1_tail [7:4], v2_head [3:0]
    b1 = ((v1 & 0x0F) << 4) | (v2 >> 2)
    # b2: v2_tail [7:6], v3 [5:0]
    b2 = ((v2 & 0x03) << 6) | v3

    packed = torch.stack([b0, b1, b2], dim=1).view(x.shape[0], -1)
    return packed, scales.view(x.shape[0], -1)


def mxfp6_to_f32(packed, scales, lut, block_size=32):
    """
    Unpacks MXFP6 E2M3 bytes back to FP32.
    packed: [N, (K*6)/8]
    scales: [N, K/block_size]
    """

    # Unpack bits, slice 3 bytes into 4 6-bit values
    b = packed.view(-1, 3)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]

    # Extract v0 (top 6 bits of b0)
    v0 = b0 >> 2
    # Extract v1 (bottom 2 of b0 + top 4 of b1)
    v1 = ((b0 & 0x03) << 4) | (b1 >> 4)
    # Extract v2 (bottom 4 of b1 + top 2 of b2)
    v2 = ((b1 & 0x0F) << 2) | (b2 >> 6)
    # Extract v3 (bottom 6 of b2)
    v3 = b2 & 0x3F

    indices = torch.stack([v0, v1, v2, v3], dim=1).view(-1).long()

    # Dequantize by mapping indices to LUT and multiply by scale
    x_unscaled = lut[indices].view(-1, block_size)
    x_final = x_unscaled * scales.view(-1, 1)

    return x_final.view(packed.shape[0], -1)


# Test parameters: (m, n, k, block_size, max_rel_error_tensor, max_rel_error_matmul)
test_cases = [
    (64, 64, 64, 32, 0.20, 0.15),
    (128, 128, 128, 32, 0.20, 0.15),
    (64, 64, 128, 32, 0.20, 0.15),
]


@require_e2e
@pytest.mark.parametrize(
    "m, n, k, block_size, max_rel_error_tensor, max_rel_error_matmul",
    test_cases,
)
def test_mxfp6_tensor_quantization(
    m, n, k, block_size, max_rel_error_tensor, max_rel_error_matmul
):
    """Test that LUT-based quantization matches the analytical MXFP reference."""
    torch.manual_seed(0)

    # Create test tensors
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)

    # Generate MXFP6 LUT
    lut = get_mxfp6_e2m3_lut()

    # LUT-based quantize and dequantize
    a_packed, a_scales = f32_to_mxfp6(a, lut, block_size=block_size)
    b_packed, b_scales = f32_to_mxfp6(b, lut, block_size=block_size)

    a_dequant = mxfp6_to_f32(a_packed, a_scales, lut, block_size=block_size)
    b_dequant = mxfp6_to_f32(b_packed, b_scales, lut, block_size=block_size)

    # Reference analytical quantization (must match f32_to_mxfp6's block_size)
    quantizer = MXFP("e2m3")
    a_ref = quantizer.quantize(a.view(-1, block_size)).view(m, k)
    b_ref = quantizer.quantize(b.view(-1, block_size)).view(n, k)

    # LUT output must match the analytical reference
    assert torch.allclose(
        a_dequant, a_ref, atol=1e-8
    ), f"Tensor A: LUT vs reference max diff {(a_dequant - a_ref).abs().max().item():.8f}"
    assert torch.allclose(
        b_dequant, b_ref, atol=1e-8
    ), f"Tensor B: LUT vs reference max diff {(b_dequant - b_ref).abs().max().item():.8f}"


@require_e2e
@pytest.mark.parametrize(
    "m, n, k, block_size, max_rel_error_tensor, max_rel_error_matmul",
    test_cases,
)
def test_mxfp6_matmul_quantization(
    m, n, k, block_size, max_rel_error_tensor, max_rel_error_matmul
):
    """Test that matmul with quantized tensors produces results within expected error bounds."""
    torch.manual_seed(0)
    device = "cuda"

    # Create test tensors
    a = torch.randn(m, k, dtype=torch.float32, device=device)
    b = torch.randn(n, k, dtype=torch.float32, device=device)

    # Generate MXFP6 LUT
    lut = get_mxfp6_e2m3_lut().to(device)

    # Quantize and dequantize
    a_packed, a_scales = f32_to_mxfp6(a, lut, block_size=block_size)
    b_packed, b_scales = f32_to_mxfp6(b, lut, block_size=block_size)

    a_dequant = mxfp6_to_f32(a_packed, a_scales, lut, block_size=block_size)
    b_dequant = mxfp6_to_f32(b_packed, b_scales, lut, block_size=block_size)

    # Reference matmul with original tensors
    expected = torch.matmul(a, b.transpose(0, 1))

    # Matmul with quantized tensors
    fake_quantize = torch.matmul(a_dequant, b_dequant.transpose(0, 1))

    # Calculate matmul errors using normalized RMSE (more robust than relative error)
    # This avoids division by near-zero values
    matmul_abs_error = (fake_quantize - expected).abs()
    rmse = torch.sqrt((matmul_abs_error**2).mean())
    expected_rms = torch.sqrt((expected**2).mean())
    normalized_rmse = rmse / (expected_rms + 1e-12)

    # Assert matmul errors are within bounds
    assert normalized_rmse.item() < max_rel_error_matmul, (
        f"Matmul normalized RMSE {normalized_rmse.item():.4f} "
        f"exceeds threshold {max_rel_error_matmul}"
    )


@require_e2e
def test_mxfp6_roundtrip_exact_values():
    """Test that values exactly representable in E2M3 round-trip perfectly."""
    lut = get_mxfp6_e2m3_lut()

    # Create a tensor with exactly representable values (from the LUT)
    # Use 32 values (one block) that are in the LUT
    exact_values = lut[:32].view(1, 32)

    # Quantize and dequantize
    packed, scales = f32_to_mxfp6(exact_values, lut, block_size=32)
    recovered = mxfp6_to_f32(packed, scales, lut, block_size=32)

    # For exactly representable values, the round-trip should be nearly exact
    # (only scale-related error)
    error = (exact_values - recovered).abs()

    assert (
        error.max().item() < 0.1
    ), f"Round-trip error for exact values {error.max().item():.6f} is too large"


@require_e2e
def test_mxfp6_weight_quantization():
    """Test that LUT-based weight quantization matches the analytical MXFP reference."""
    torch.manual_seed(0)
    linear = torch.nn.Linear(32, 1, bias=False)

    x = torch.randn(1, 32) * 1e4
    linear.weight.data = x

    # LUT-based quantize and dequantize the weight
    lut = get_mxfp6_e2m3_lut()
    packed, scales = f32_to_mxfp6(linear.weight.data, lut, block_size=32)
    qx_weight = mxfp6_to_f32(packed, scales, lut, block_size=32)

    # Reference analytical quantization
    quantizer = MXFP("e2m3")
    y = quantizer.quantize(x)

    assert torch.allclose(qx_weight, y, atol=1e-8), (
        f"Weight quantization: LUT vs reference max diff "
        f"{(qx_weight - y).abs().max().item():.8f}"
    )


def simple_quantize_matmul_test():
    """Basic matrix multiplication kernel with detailed output."""

    # Create test matrices
    m, n, k = 64, 64, 64  # Small dimensions for testing
    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Generate MXFP6 LUT
    lut = get_mxfp6_e2m3_lut().to("cpu")

    # Pack to MXFP6 and unpack back to get fake-quantized tensors (on CPU)
    a_packed, a_scales = f32_to_mxfp6(a.clone().to(torch.float32).cpu(), lut)
    b_packed, b_scales = f32_to_mxfp6(b.clone().to(torch.float32).cpu(), lut)

    # Dequantize back to f32 tensors
    a_dequant = mxfp6_to_f32(a_packed, a_scales, lut).to("cuda")
    b_dequant = mxfp6_to_f32(b_packed, b_scales, lut).to("cuda")

    def get_size(t):
        return t.element_size() * t.numel()

    print("=" * 60)
    print("MXFP6 Quantization Test Results")
    print("=" * 60)

    print("\n--- Memory Usage ---")
    print(
        f"A size f16: {get_size(a)} bytes, A packed size: {get_size(a_packed)} bytes, A dequant size: {get_size(a_dequant)} bytes"
    )
    print(
        f"B size f16: {get_size(b)} bytes, B packed size: {get_size(b_packed)} bytes, B dequant size: {get_size(b_dequant)} bytes"
    )

    # Compare original vs dequantized tensors
    print("\n--- Tensor A: Original vs Dequantized ---")
    a_cpu = a.to(torch.float32).cpu()
    a_dequant_cpu = a_dequant.cpu()
    a_abs_error = (a_cpu - a_dequant_cpu).abs()
    a_rel_error = a_abs_error / (a_cpu.abs() + 1e-12)
    print(f"Max absolute error: {a_abs_error.max().item():.6f}")
    print(f"Mean absolute error: {a_abs_error.mean().item():.6f}")
    print(f"Max relative error: {a_rel_error.max().item():.4f}")
    print(f"Mean relative error: {a_rel_error.mean().item():.4f}")

    print("\n--- Tensor B: Original vs Dequantized ---")
    b_cpu = b.to(torch.float32).cpu()
    b_dequant_cpu = b_dequant.cpu()
    b_abs_error = (b_cpu - b_dequant_cpu).abs()
    b_rel_error = b_abs_error / (b_cpu.abs() + 1e-12)
    print(f"Max absolute error: {b_abs_error.max().item():.6f}")
    print(f"Mean absolute error: {b_abs_error.mean().item():.6f}")
    print(f"Max relative error: {b_rel_error.max().item():.4f}")
    print(f"Mean relative error: {b_rel_error.mean().item():.4f}")

    # Compare LUT-based quantization against analytical MXFP reference
    # Must reshape to block_size=32 to match f32_to_mxfp6's block structure
    quantizer = MXFP("e2m3")
    a_ref = quantizer.quantize(a.to(torch.float32).cpu().view(-1, 32)).view(m, k)
    b_ref = quantizer.quantize(b.to(torch.float32).cpu().view(-1, 32)).view(n, k)

    a_ref_diff = (a_dequant_cpu - a_ref).abs()
    b_ref_diff = (b_dequant_cpu - b_ref).abs()

    print("\n--- LUT vs Analytical MXFP Reference ---")
    print(
        f"Tensor A: max diff = {a_ref_diff.max().item():.8f}, "
        f"mean diff = {a_ref_diff.mean().item():.8f}"
    )
    print(
        f"Tensor B: max diff = {b_ref_diff.max().item():.8f}, "
        f"mean diff = {b_ref_diff.mean().item():.8f}"
    )
    a_match = torch.allclose(a_dequant_cpu, a_ref, atol=1e-6)
    b_match = torch.allclose(b_dequant_cpu, b_ref, atol=1e-6)
    print(f"All close (atol=1e-6): A={a_match}, B={b_match}")

    # Matmul comparison
    expected = torch.matmul(a.to(torch.float32), b.to(torch.float32).transpose(0, 1))
    fake_quantize = torch.matmul(a_dequant, b_dequant.transpose(0, 1))

    print("\n--- Matmul: Reference vs Quantized ---")
    print(
        "--- (Can have large differences due to errors when expected is near zero) ---"
    )
    matmul_abs_error = (fake_quantize - expected).abs()
    matmul_rel_error = matmul_abs_error / (expected.abs() + 1e-12)
    print(f"Max absolute error: {matmul_abs_error.max().item():.6f}")
    print(f"Mean absolute error: {matmul_abs_error.mean().item():.6f}")
    print(f"Max relative error: {matmul_rel_error.max().item():.4f}")
    print(f"Mean relative error: {matmul_rel_error.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("GEMM test with quantization ran successfully!")
    print("=" * 60)


if __name__ == "__main__":
    simple_quantize_matmul_test()
