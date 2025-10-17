"""
Reduction Operations Examples

Demonstrates various reduction patterns including basic sum reductions,
broadcast-then-reduce operations, and complex 3D weighted sums (MOE-style).
"""

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
import torch
import sympy

from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from utils import parse_args, list_tests, run_test


M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def test_reduce_sum(is_debug=False):
    """Basic sum reduction along the last dimension of a 2D tensor."""
    shape = (64, 128)
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        res = tkw.sum(res, dim=N)
        tkw.write(res, c)

    torch.manual_seed(1)
    a = torch.randn(shape, dtype=torch.float16, device="cuda")
    c = torch.zeros((shape[0],), dtype=torch.float16, device="cuda")
    ref = torch.sum(a, dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )
    test = wave_compile(options, test)

    if is_debug:
        print(test.asm)

    test(a, c)
    torch.testing.assert_close(ref, c, atol=0.1, rtol=1e-05)
    print("Test passed")


def test_broadcast_reduce_sum(is_debug=False):
    """Broadcast a 1D tensor to 2D, multiply element-wise, then reduce along last dimension."""
    shape = (64, 128)
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
        c_temp: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a)
        rhs = tkw.read(b)
        rhs = tkw.broadcast(rhs, (M, N))
        res = lhs * rhs
        tkw.write(res, c_temp)
        res = tkw.sum(res, dim=N)
        tkw.write(res, c)

    a = torch.randn(shape, dtype=torch.float16, device="cuda")
    b = torch.randn(shape[0], dtype=torch.float16, device="cuda")
    c = torch.zeros((shape[0],), dtype=torch.float16, device="cuda")
    c_temp = torch.zeros(shape, dtype=torch.float16, device="cuda")

    ref_temp = a * b.view(-1, 1)
    ref = torch.sum(ref_temp, dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )
    test = wave_compile(options, test)

    if is_debug:
        print(test.asm)

    test(a, b, c, c_temp)
    torch.testing.assert_close(ref_temp, c_temp, atol=0.1, rtol=1e-05)
    torch.testing.assert_close(ref, c, atol=0.1, rtol=1e-05)
    print("Test passed")


def test_moe_weighted_sum(is_debug=False):
    """3D weighted sum reduction matching MOE pattern: broadcast 2D weights to 3D, multiply, then sum along middle dimension."""
    shape = (64, 64, 128)  # (B, K, D)
    B = tkl.sym.B
    K = tkl.sym.K
    D = tkl.sym.D
    wave_size = 64
    BLOCK_B = 1
    BLOCK_K = sympy.ceiling(shape[1] / wave_size) * wave_size
    BLOCK_D = 1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={B: BLOCK_B, K: BLOCK_K, D: BLOCK_D},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 0)]
    constraints += [tkw.WorkgroupConstraint(D, BLOCK_D, 1)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]

    @tkw.wave(constraints)
    def test(
        out: tkl.Memory[B, K, D, ADDRESS_SPACE, tkl.f16],
        topk_weights: tkl.Memory[B, K, ADDRESS_SPACE, tkl.f16],
        result: tkl.Memory[B, D, ADDRESS_SPACE, tkl.f16],
        temp_output: tkl.Memory[B, K, D, ADDRESS_SPACE, tkl.f16],
    ):
        # Read 3D tensor: (B, K, D)
        out_vals = tkw.read(out)
        # Read 2D weights: (B, K)
        weights = tkw.read(topk_weights)
        # Broadcast weights to (B, K, D)
        weights_broadcast = tkw.broadcast(weights, [B, K, D])
        # Multiply element-wise: (B, K, D) * (B, K, D)
        weighted = out_vals * weights_broadcast
        # Write intermediate result for verification
        tkw.write(weighted, temp_output)
        # Sum along K dimension: (B, K, D) -> (B, D)
        res = tkw.sum(weighted, dim=K)
        # Write final result
        tkw.write(res, result)

    # Create input tensors
    out = torch.randn(shape, dtype=torch.float16, device="cuda")
    topk_weights = torch.randn(shape[0], shape[1], dtype=torch.float16, device="cuda")
    result = torch.zeros((shape[0], shape[2]), dtype=torch.float16, device="cuda")
    temp_output = torch.zeros(shape, dtype=torch.float16, device="cuda")

    # Reference computation matching MOE pattern
    ref_temp = out * topk_weights.view(shape[0], shape[1], 1)
    ref = torch.sum(ref_temp, dim=1)

    options = WaveCompileOptions(
        subs={
            B: shape[0],
            K: shape[1],
            D: shape[2],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        print_ir_after="all" if is_debug else [],
    )
    test = wave_compile(options, test)

    if is_debug:
        print(test.asm)

    test(out, topk_weights, result, temp_output)
    torch.testing.assert_close(ref_temp, temp_output, atol=0.1, rtol=1e-05)
    torch.testing.assert_close(ref, result, atol=0.1, rtol=1e-05)
    print("Test passed")


if __name__ == "__main__":
    args = parse_args()
    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
