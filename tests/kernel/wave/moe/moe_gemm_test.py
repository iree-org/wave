# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from .torch_kernels import moe_align_block_size_pytorch
import torch.nn.functional as F
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
)
from ..common.utils import (
    require_e2e,
    require_cdna_3_or_4,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_zeros,
)
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.lang.global_symbols import *
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

torch.manual_seed(0)


num_tokens_values = [1, 10, 33, 256, 2048]
topk_values = [2]
block_size_values = [16, 32, 64]
num_experts_values = [8, 64]
k_values = [32, 64, 128, 511, 4096]
n_values = [64, 128, 256, 512, 1024, 14336]
dtype_values = [torch.float16, torch.bfloat16]


def get_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    dtype: torch.dtype = torch.float16,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    dtype = torch_dtype_to_wave(dtype)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, M, 0)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, dtype],
        b: tkl.Memory[N, K, ADDRESS_SPACE, dtype],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, dtype]
            a_reg = tkw.read(a)
            # b_reg: tkw.Register[N, K, dtype]
            b_reg = tkw.read(b)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        M: shape[0],
        N: shape[1],
        K: shape[2],
        BLOCK_K: 32,
    }
    hyperparams.update(get_default_scheduling_params())

    return gemm, hyperparams


def get_moe_gemm_kernel(
    num_tokens: int,
    topk: int,
    block_size: int,
    num_experts: int,
    k: int,
    n: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    assert datatype in [tkl.f16, tkl.bf16], f"Unsupported datatype: {datatype}"

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = tkl.sym.E
    EM = tkl.sym.EM
    TOPK = tkl.sym.TOPK
    MAX_M_BLOCKS = tkl.sym.MAX_M_BLOCKS
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    NUM_TOKENS_BUF_SIZE = tkl.sym.NUM_TOKENS_BUF_SIZE

    print("INSIDE ", n, block_size)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(EM * N, BLOCK_M * BLOCK_N, 0)]
 #  constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            vector_shapes={M: 0, TOPK: 0, N: 32},
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def moe_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, datatype],
        b: tkl.Memory[E, N, K, ADDRESS_SPACE, datatype],
        sorted_token_ids: tkl.Memory[EM, ADDRESS_SPACE, tkl.i32],
        expert_ids: tkl.Memory[MAX_M_BLOCKS, ADDRESS_SPACE, tkl.i32],
        num_tokens_post_padded: tkl.Memory[NUM_TOKENS_BUF_SIZE, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, TOPK, N, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        c_reg = tkl.Register[M, TOPK, N, datatype](0.0)

        tkw.write(c_reg, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_size,
        BLOCK_N: block_size,
        BLOCK_K: 32,
        M: num_tokens,
        N: n,
        K: k,
        E: num_experts,
        EM: max_num_tokens_padded,
        TOPK: topk,
        MAX_M_BLOCKS: max_num_m_blocks,
        NUM_TOKENS_BUF_SIZE: 1,
    }

    return moe_gemm, hyperparams


def moe_gemm_pytorch(
    a,  # Input tokens: [M, K]
    b,  # Expert weights: [E, N, K]
    c,  # Output: [M, topk, N]
    sorted_token_ids,  # Sorted token-expert pair indices: [EM] (padded)
    expert_ids,  # Expert ID for each block: [num_blocks]
    num_tokens_post_padded,  # Total padded length: [1]
    top_k,  # Number of experts per token
    block_size_m=64,
    block_size_n=64,
    block_size_k=64,
):
    """
    PyTorch equivalent of the Triton fused MoE kernel.

    Args:
        a: Input token embeddings [M, K]
        b: Expert weight matrices [E, N, K]
        sorted_token_ids: Token-expert pair indices sorted by expert [EM] (padded)
        expert_ids: Expert ID for each block [num_blocks]
        num_tokens_post_padded: Total padded length [1]
        top_k: Number of experts each token is routed to

    Returns:
        c: Output tensor [M, topk, N]
    """
    M, K = a.shape
    E, N, _ = b.shape
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = M * top_k

    # Process tokens in blocks
    num_blocks = (EM + block_size_m - 1) // block_size_m

    for i, idx in enumerate(sorted_token_ids.tolist()):
        if i % block_size_m == 0:
            block_token_ids = sorted_token_ids[i : i + block_size_m]
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            valid_mask = block_token_ids < num_valid_tokens
    for block_idx in range(num_blocks):
        # Determine token range for this block
        start_idx = block_idx * block_size_m
        end_idx = min(start_idx + block_size_m, EM)

        # Skip if we're past valid tokens
        if start_idx >= num_tokens_post_padded.item():
            continue

        # Get token-expert pair indices for this block
        block_token_ids = sorted_token_ids[start_idx:end_idx]

        # Create mask for valid token-expert pairs
        valid_mask = block_token_ids < num_valid_tokens

        if not valid_mask.any():
            continue

        # Get the expert ID for this block
        expert_id = expert_ids[block_idx].item()

        # Initialize accumulator for this block
        accumulator = torch.zeros(block_size_m, N, dtype=a.dtype, device=a.device)

        # Process K dimension in chunks (simulating the K-loop in Triton)
        for k_start in range(0, K, block_size_k):
            k_end = min(k_start + block_size_k, K)
            actual_k_size = k_end - k_start

            # Load block from A
            # Map token-expert pair indices to original token indices
            orig_token_ids = torch.clamp(block_token_ids // top_k, 0, M - 1)
            block_a = a[orig_token_ids, k_start:k_end]  # [block_size_m, actual_k_size]

            # Apply valid mask to A
            block_a = block_a * valid_mask.to(a.dtype).unsqueeze(1)

            # Load block from B (expert weights)
            # Process N dimension in chunks
            for n_start in range(0, N, block_size_n):
                n_end = min(n_start + block_size_n, N)
                actual_n_size = n_end - n_start

                # Get expert weights: B[expert_id, n_start:n_end, k_start:k_end]
                # Need to transpose to [k, n] for matrix multiplication
                block_b = b[
                    expert_id, n_start:n_end, k_start:k_end
                ].t()  # [actual_k_size, actual_n_size]

                # Compute matrix multiplication: [block_size_m, k] @ [k, n] = [block_size_m, n]
                partial_result = torch.matmul(
                    block_a, block_b
                )  # [block_size_m, actual_n_size]

                # Accumulate in the correct position
                accumulator[:, n_start:n_end] += partial_result

        # Write back to output tensor using the stride-based mapping
        # This is the key: sorted_token_ids encodes the mapping to 3D positions
        for i, token_id in enumerate(block_token_ids):
            if token_id >= num_valid_tokens:
                continue

            # Decode the 3D position from the flat index
            # Which original token (0 to M-1)
            orig_token = token_id // top_k
            # Which expert slot for that token (0 to top_k-1)
            expert_slot = token_id % top_k

            c[orig_token, expert_slot] = accumulator[i]


@require_e2e
@require_cdna_3_or_4
@pytest.mark.parametrize("shape", [[64, 64, 128]])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize("datatype", [torch.float16])
def testPureGemm(
    shape: tuple[int],
    mfma_variant: MMAType,
    datatype: torch.dtype,
):
    gemm, hyperparams = get_gemm_kernel(
        shape, mfma_variant, datatype
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        print_mlir=True,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    a = device_randn(shape[0], shape[2], dtype=datatype)
  # a = torch.arange(shape[0] * shape[2], dtype=datatype, device='cuda').view((shape[0], shape[2]))
    b = device_randn(shape[1], shape[2], dtype=datatype)
  # b = torch.reshape(a, (shape[1], shape[2]))
    c = device_zeros(shape[0], shape[1], dtype=torch.float32)
    gemm(a, b, c)

  # ref = torch.matmul(a, b.T).to(torch.float32)
    ref = torch.matmul(a.float(), b.T.float())
    for i in range(64):
        print(f"C[{i}] {c[i]}")
        print(f"REF[{i}] {ref[i]}")
    torch.testing.assert_close(c, ref, check_device=False)
    abort()


@pytest.mark.parametrize("num_tokens", num_tokens_values)
@pytest.mark.parametrize("topk", topk_values)
@pytest.mark.parametrize("block_size", block_size_values)
@pytest.mark.parametrize("num_experts", num_experts_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("dtype", dtype_values)
def test_moe_gemm(
    num_tokens: int,
    topk: int,
    block_size: int,
    num_experts: int,
    k: int,
    n: int,
    dtype: DataType,
):
    """
    Tests the fused_moe_kernel function using Pytest parameterization.
    """
    scores = torch.rand(num_tokens, num_experts, device='cuda')

    # Get topk expert indices for each token
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())

    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )

    a = torch.rand((num_tokens, k), dtype=dtype, device='cuda')
    b = torch.rand((num_experts, n, k), dtype=dtype, device='cuda')
    c = torch.zeros(num_tokens, topk, n, dtype=dtype, device='cuda')

    moe_gemm_pytorch(
        a, b, c, sorted_ids, expert_ids, num_tokens_post_pad, topk, block_size
    )

    ref_c = torch.zeros(num_tokens, topk, n, dtype=dtype, device='cuda')
    for expert_idx in range(num_experts):
        mask = topk_ids == expert_idx
        token_indices, topk_indices = torch.where(mask)

        if len(token_indices) > 0:
            tokens = a[token_indices]
            expert_out = torch.matmul(tokens, b[expert_idx].T)

            ref_c[token_indices, topk_indices] = expert_out

    rtol, atol = 1e-1, 1e-2

    torch.testing.assert_close(
        c,
        ref_c,
        rtol=rtol,
        atol=atol,
    )

    mlir_c = torch.zeros(num_tokens, topk, n, dtype=dtype, device='cuda')
    asm_dtype0_64_32_8_64_2_10 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [1, 1, 1] subgroup_size = 32>
module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel {
    stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
      %c40 = arith.constant 40 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c40, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel(
          // Input memrefs
       // %a_ptr: memref<10x32xf16>,
       // %b_ptr: memref<8x64x32xf16>,
       // %c_ptr: memref<10x2x64xf16>,
       // %sorted_token_ids_ptr: memref<587xi32>,
       // %expert_ids_ptr: memref<10xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 64
        // K = 32
        // EM = 587
        // top_k = 2
        // num_valid_tokens = 20
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 64 : index
        %K = arith.constant 32 : index
        %EM = arith.constant 587 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 20 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c2048 = arith.constant 2048 : index
        %c10 = arith.constant 10 : index
        %c30 = arith.constant 30 : index
        %c31 = arith.constant 31 : index
        %c62 = arith.constant 62 : index
        %c63 = arith.constant 63 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c5 = arith.constant 5 : index
        %c6 = arith.constant 6 : index
        %c7 = arith.constant 7 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %passthru_a = vector.broadcast %f0_f16 : f16 to vector<64x32xf16>
        %true_mask = arith.constant dense<true> : vector<64xi1>
        %zeroes_64 = arith.constant dense<0> : vector<64xi32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<10x32xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x64x32xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<10x2x64xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<587xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Program ID mapping
        %pid = gpu.block_id x
        %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
        %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
        %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
        %group_id = arith.divui %pid, %num_pid_in_group : index
        %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
        %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
        %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
        %0 = arith.remsi %pid, %num_pid_in_group : index
        %1 = arith.remsi %0, %group_size_m : index
        %pid_m = arith.addi %first_pid_m, %1 : index
        %pid_n = arith.divui %0, %group_size_m : index

        %print = arith.cmpi eq, %pid, %c1 : index

        // Early exit check
        %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
        %num_tokens_post_padded = arith.index_cast %2 : i32 to index
        %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
        %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
        scf.if %should_exit {
          scf.yield
        } else {
          // Compute token mask
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
          %range_m = vector.step : vector<64xindex>
          %3 = vector.broadcast %offs_token_id_base : index to vector<64xindex>
          %offs_token_id = arith.addi %3, %range_m : vector<64xindex>

          // Load token IDs
          %4 = vector.gather %sorted_token_ids_ptr[%c0] [%offs_token_id],
          	%true_mask, %zeroes_64 : memref<587xi32>, vector<64xindex>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %offs_token = arith.index_cast %4 : vector<64xi32> to vector<64xindex>

          // Create token mask
          %5 = vector.broadcast %num_valid_tokens : index to vector<64xindex>
          %token_mask = arith.cmpi slt, %offs_token, %5 : vector<64xindex>

          // Load expert ID
          %6 = memref.load %expert_ids_ptr[%pid_m] : memref<10xi32>
          %off_experts = arith.index_cast %6 : i32 to index

          // Setup B matrix pointers
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %range_n = vector.step : vector<64xindex>
          %7 = vector.broadcast %offs_bn_base : index to vector<64xindex>
          %offs_cn = arith.addi %7, %range_n : vector<64xindex>
          %N_splat = vector.broadcast %N : index to vector<64xindex>
          %offs_bn = arith.remsi %offs_cn, %N_splat : vector<64xindex>
          %off_bn = arith.remsi %offs_bn_base, %N : index

scf.if %print { gpu.printf "pid %d\\n", %pid : index }
scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
scf.if %print { gpu.printf "off_experts %d\\n", %off_experts : index }
scf.if %print { gpu.printf "off_bn %d\\n", %off_bn : index }

scf.for %i = %c0 to %c10 step %c1 {
  %token_val = vector.extract %offs_token[%i] : index from vector<64xindex>
  %mask_val = vector.extract %token_mask[%i] : i1 from vector<64xi1>
  %token_i32 = arith.index_cast %token_val : index to i32
  %mask_i32 = arith.select %mask_val, %c1_i32, %c0_i32 : i32
  scf.if %print { gpu.printf "offs_token[%d] %d token_mask[%d] %d\\n", %i, %token_i32, %i, %mask_i32 : index, i32, index, i32 }
}

          // Setup K range
          %offs_k = vector.step : vector<32xindex>

          // -----------------------------------------------------------
          // Compute A matrix indices (equivalent to a_ptrs computation)
          // Expand dims: offs_token[:, None] -> 64x1
          %offs_token_2d = vector.shape_cast %offs_token : vector<64xindex> to vector<64x1xindex>

          // Broadcast top_k to 64x1
          %top_k_2d = vector.broadcast %top_k : index to vector<64x1xindex>

          // Compute offs_token[:, None] // top_k
          %token_div_topk = arith.divsi %offs_token_2d, %top_k_2d : vector<64x1xindex>

          // Broadcast K to 64x1
          %K_splat = vector.broadcast %K : index to vector<64x1xindex>

          // Compute first term: (offs_token // top_k) * K
          %a_row_offsets = arith.muli %token_div_topk, %K_splat : vector<64x1xindex>

          // Broadcast both terms to 64x32 and add
          %first_broadcast_a = vector.broadcast %a_row_offsets : vector<64x1xindex> to vector<64x32xindex>
          %second_broadcast_a = vector.broadcast %offs_k : vector<32xindex> to vector<64x32xindex>
          %indices_a = arith.addi %first_broadcast_a, %second_broadcast_a : vector<64x32xindex>

          // -----------------------------------------------------------
          // Initialize accumulator
          %accumulator = vector.broadcast %f0 : f32 to vector<64x64xf32>

          // Main K loop
          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %8 = vector.shape_cast %token_mask : vector<64xi1> to vector<64x1xi1>
          %a_mask_base = vector.broadcast %8 : vector<64x1xi1> to vector<64x32xi1>

          %a_flat = memref.collapse_shape %a_ptr [[0, 1]] : memref<10x32xf16> into memref<320xf16>
          %b_view =  memref.subview %b_ptr[%off_experts, 0, 0] [1, 64, 32] [1, 1, 1] :
        	memref<8x64x32xf16> to memref<1x64x32xf16, strided<[2048, 32, 1], offset: ?>>
       // %b_block = memref.collapse_shape %b_view [[0, 1], [2]] : memref<1x64x32xf16, strided<[2048, 32, 1], offset: ?>> into memref<64x32xf16, strided<[32, 1], offset: ?>>
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<10x2x64xf16> into memref<1280xf16>

//%b_block0.0 = memref.load %b_block[%c0, %c0] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[0][0] %f\\n", %b_block0.0 : f16 }
//
//%b_block0.1 = memref.load %b_block[%c0, %c1] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[0][1] %f\\n", %b_block0.1 : f16 }
//
//%b_block0.30 = memref.load %b_block[%c0, %c30] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[0][30] %f\\n", %b_block0.30 : f16 }
//
//%b_block0.31 = memref.load %b_block[%c0, %c31] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[0][31] %f\\n", %b_block0.31 : f16 }
//
//%b_block1.0 = memref.load %b_block[%c1, %c0] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[1][0] %f\\n", %b_block1.0 : f16 }
//
//%b_block1.1 = memref.load %b_block[%c1, %c1] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[1][1] %f\\n", %b_block1.1 : f16 }
//
//%b_block1.30 = memref.load %b_block[%c1, %c30] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[1][30] %f\\n", %b_block1.30 : f16 }
//
//%b_block1.31 = memref.load %b_block[%c1, %c31] : memref<64x32xf16, strided<[32, 1], offset: ?>>
//scf.if %print { gpu.printf "b_block[1][31] %f\\n", %b_block1.31 : f16 }

          // 2048 = 32 * 64
          %b_linear_offset = arith.muli %off_experts, %c2048 : index
          %b_block_transposed = memref.reinterpret_cast %b_ptr to offset: [%b_linear_offset], sizes: [32, 64], strides: [1, 32] : memref<8x64x32xf16> to memref<32x64xf16, strided<[1, 32], offset: ?>>


%b_block_transposed0.0 = memref.load %b_block_transposed[%c0, %c0] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][0] %f\\n", %b_block_transposed0.0 : f16 }

%b_block_transposed0.1 = memref.load %b_block_transposed[%c0, %c1] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][1] %f\\n", %b_block_transposed0.1 : f16 }

%b_block_transposed0.30 = memref.load %b_block_transposed[%c0, %c30] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][30] %f\\n", %b_block_transposed0.30 : f16 }

%b_block_transposed0.31 = memref.load %b_block_transposed[%c0, %c31] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][31] %f\\n", %b_block_transposed0.31 : f16 }

%b_block_transposed0.62 = memref.load %b_block_transposed[%c0, %c62] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][62] %f\\n", %b_block_transposed0.62 : f16 }

%b_block_transposed0.63 = memref.load %b_block_transposed[%c0, %c63] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[0][63] %f\\n", %b_block_transposed0.63 : f16 }

%b_block_transposed1.0 = memref.load %b_block_transposed[%c1, %c0] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][0] %f\\n", %b_block_transposed1.0 : f16 }

%b_block_transposed1.1 = memref.load %b_block_transposed[%c1, %c1] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][1] %f\\n", %b_block_transposed1.1 : f16 }

%b_block_transposed1.30 = memref.load %b_block_transposed[%c1, %c30] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][30] %f\\n", %b_block_transposed1.30 : f16 }

%b_block_transposed1.31 = memref.load %b_block_transposed[%c1, %c31] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][31] %f\\n", %b_block_transposed1.31 : f16 }

%b_block_transposed1.62 = memref.load %b_block_transposed[%c1, %c62] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][62] %f\\n", %b_block_transposed1.62 : f16 }

%b_block_transposed1.63 = memref.load %b_block_transposed[%c1, %c63] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[1][63] %f\\n", %b_block_transposed1.63 : f16 }

%b_block_transposed30.0 = memref.load %b_block_transposed[%c30, %c0] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[30][0] %f\\n", %b_block_transposed30.0 : f16 }

%b_block_transposed30.1 = memref.load %b_block_transposed[%c30, %c1] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[30][1] %f\\n", %b_block_transposed30.1 : f16 }

%b_block_transposed31.0 = memref.load %b_block_transposed[%c31, %c0] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[31][0] %f\\n", %b_block_transposed31.0 : f16 }

%b_block_transposed31.1 = memref.load %b_block_transposed[%c31, %c1] : memref<32x64xf16, strided<[1, 32], offset: ?>>
scf.if %print { gpu.printf "b_block_transposed[31][1] %f\\n", %b_block_transposed31.1 : f16 }


//%index_a0 = vector.extract %a_row_offsets[0, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[0][0] %d\\n", %index_a0 : index }
//
//%index_a1 = vector.extract %a_row_offsets[1, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[1][0] %d\\n", %index_a1 : index }
//
//%index_a2 = vector.extract %a_row_offsets[2, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[2][0] %d\\n", %index_a2 : index }
//
//%index_a3 = vector.extract %a_row_offsets[3, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[3][0] %d\\n", %index_a3 : index }
//
//%index_a4 = vector.extract %a_row_offsets[4, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[4][0] %d\\n", %index_a4 : index }
//
//%index_a5 = vector.extract %a_row_offsets[5, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[5][0] %d\\n", %index_a5 : index }
//
//%index_a6 = vector.extract %a_row_offsets[6, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[6][0] %d\\n", %index_a6 : index }
//
//%index_a7 = vector.extract %a_row_offsets[7, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[7][0] %d\\n", %index_a7 : index }
//
//%index_a8 = vector.extract %a_row_offsets[7, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[8][0] %d\\n", %index_a7 : index }
//
//%index_a9 = vector.extract %a_row_offsets[9, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_a[9][0] %d\\n", %index_a9 : index }

          %result = scf.for %k_block = %c0 to %num_blocks step %c1 iter_args(%acc = %accumulator) -> vector<64x64xf32> {
            // Compute current K offset
            %k_offset = arith.muli %k_block, %BLOCK_SIZE_K : index

//%k_offset_i32 = arith.index_cast %k_offset : index to i32
//scf.if %print { gpu.printf "k_offset [%d] %d\\n", %k_block, %k_offset_i32 : index, i32 }

            %k_offset_broadcast = vector.broadcast %k_offset : index to vector<64x32xindex>
            %indices_a_with_k = arith.addi %indices_a, %k_offset_broadcast : vector<64x32xindex>

            %a = vector.gather %a_flat[%c0][%indices_a_with_k], %a_mask_base, %passthru_a :
                memref<320xf16>, vector<64x32xindex>, vector<64x32xi1>, vector<64x32xf16> into vector<64x32xf16>

%a0.0 = vector.extract %a[0, 0] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }

%a0.1 = vector.extract %a[0, 1] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }

%a0.30 = vector.extract %a[0, 30] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[0][30] %f\\n", %a0.30 : f16 }

%a0.31 = vector.extract %a[0, 31] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[0][31] %f\\n", %a0.31 : f16 }

%a1.0 = vector.extract %a[1, 0] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }

%a1.1 = vector.extract %a[1, 1] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }

%a1.30 = vector.extract %a[1, 30] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[1][30] %f\\n", %a1.30 : f16 }

%a1.31 = vector.extract %a[1, 31] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[1][31] %f\\n", %a1.31 : f16 }

%a63.0 = vector.extract %a[63, 0] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }

%a63.1 = vector.extract %a[63, 1] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }

%a63.30 = vector.extract %a[63, 30] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[63][30] %f\\n", %a63.30 : f16 }

%a63.31 = vector.extract %a[63, 31] : f16 from vector<64x32xf16>
scf.if %print { gpu.printf "a[63][31] %f\\n", %a63.31 : f16 }

           %b = vector.transfer_read %b_block_transposed[%k_offset, %off_bn], %f0_f16 : memref<32x64xf16, strided<[1, 32], offset: ?>>, vector<32x64xf16>

      //   %b = vector.transfer_read %b_block[%off_bn, %k_offset], %f0_f16
      // {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<64x32xf16, strided<[32, 1], offset: ?>>, vector<32x64xf16>

%b0.0 = vector.extract %b[0, 0] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }

%b0.1 = vector.extract %b[0, 1] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }

%b0.62 = vector.extract %b[0, 62] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[0][62] %f\\n", %b0.62 : f16 }

%b0.63 = vector.extract %b[0, 63] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[0][63] %f\\n", %b0.63 : f16 }

%b1.0 = vector.extract %b[1, 0] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }

%b1.1 = vector.extract %b[1, 1] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }

%b1.62 = vector.extract %b[1, 62] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[1][62] %f\\n", %b1.62 : f16 }

%b1.63 = vector.extract %b[1, 63] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[1][63] %f\\n", %b1.63 : f16 }

%b30.0 = vector.extract %b[30, 0] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[30][0] %f\\n", %b30.0 : f16 }

%b30.1 = vector.extract %b[30, 1] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[30][1] %f\\n", %b30.1 : f16 }

%b30.62 = vector.extract %b[30, 62] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[30][62] %f\\n", %b30.62 : f16 }

%b30.63 = vector.extract %b[30, 63] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[30][63] %f\\n", %b30.63 : f16 }

%b31.0 = vector.extract %b[31, 0] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[31][0] %f\\n", %b31.0 : f16 }

%b31.1 = vector.extract %b[31, 1] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[31][1] %f\\n", %b31.1 : f16 }

%b31.62 = vector.extract %b[31, 62] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[31][62] %f\\n", %b31.62 : f16 }

%b31.63 = vector.extract %b[31, 63] : f16 from vector<32x64xf16>
scf.if %print { gpu.printf "b[31][63] %f\\n", %b31.63 : f16 }

            // Matrix multiplication: f16 x f16 -> f32 accumulation
            %dot_result = vector.contract {
                indexing_maps = [affine_map<(m,n,k) -> (m,k)>,
                               affine_map<(m,n,k) -> (k,n)>,
                               affine_map<(m,n,k) -> (m,n)>],
                iterator_types = ["parallel", "parallel", "reduction"]
            } %a, %b, %acc : vector<64x32xf16>, vector<32x64xf16> into vector<64x64xf32>

            scf.yield %dot_result : vector<64x64xf32>
          }

          // Convert accumulator to output precision
          %result_f16 = arith.truncf %result : vector<64x64xf32> to vector<64x64xf16>

	      // TODO: use vector.scatter once we have
	      // vector::populateVectorScatterLoweringPatterns to handle multi
	      // dimensional scatter lowering.

          // Compute output indices
          %offs_cn_2d = vector.shape_cast %offs_cn : vector<64xindex> to vector<1x64xindex>

          %N_2d = vector.broadcast %N : index to vector<64x1xindex>
          %c_row_offsets = arith.muli %offs_token_2d, %N_2d : vector<64x1xindex>

          %first_broadcast_c = vector.broadcast %c_row_offsets : vector<64x1xindex> to vector<64x64xindex>
          %second_broadcast_c = vector.broadcast %offs_cn_2d : vector<1x64xindex> to vector<64x64xindex>
          %c_indices = arith.addi %first_broadcast_c, %second_broadcast_c : vector<64x64xindex>

          // Create output mask
          %offs_cn_mask = arith.cmpi slt, %offs_cn, %N_splat : vector<64xindex>

          %c_mask_base = vector.broadcast %8 : vector<64x1xi1> to vector<64x64xi1>
          %cn_mask_broadcast = vector.broadcast %offs_cn_mask : vector<64xi1> to vector<64x64xi1>
          %output_mask = arith.andi %c_mask_base, %cn_mask_broadcast : vector<64x64xi1>

//%c_mask_base_0 = vector.extract %8[0, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_0 = vector.extract %offs_cn_mask[0] : i1 from vector<64xi1>
//%c_mask_base0_i32 = arith.select %c_mask_base_0, %c1_i32, %c0_i32 : i32
//%offs_cn_mask0_i32 = arith.select %offs_cn_mask_0, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[0] %d offs_cn_mask[0] %d\\n", %c_mask_base0_i32, %offs_cn_mask0_i32 : i32, i32 }
//
//%c_mask_base_1 = vector.extract %8[1, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_1 = vector.extract %offs_cn_mask[1] : i1 from vector<64xi1>
//%c_mask_base1_i32 = arith.select %c_mask_base_1, %c1_i32, %c0_i32 : i32
//%offs_cn_mask1_i32 = arith.select %offs_cn_mask_1, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[1] %d offs_cn_mask[1] %d\\n", %c_mask_base1_i32, %offs_cn_mask1_i32 : i32, i32 }
//
//%c_mask_base_2 = vector.extract %8[2, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_2 = vector.extract %offs_cn_mask[2] : i1 from vector<64xi1>
//%c_mask_base2_i32 = arith.select %c_mask_base_2, %c1_i32, %c0_i32 : i32
//%offs_cn_mask2_i32 = arith.select %offs_cn_mask_2, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[2] %d offs_cn_mask[2] %d\\n", %c_mask_base2_i32, %offs_cn_mask2_i32 : i32, i32 }
//
//%c_mask_base_3 = vector.extract %8[3, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_3 = vector.extract %offs_cn_mask[3] : i1 from vector<64xi1>
//%c_mask_base3_i32 = arith.select %c_mask_base_3, %c1_i32, %c0_i32 : i32
//%offs_cn_mask3_i32 = arith.select %offs_cn_mask_3, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[3] %d offs_cn_mask[3] %d\\n", %c_mask_base3_i32, %offs_cn_mask3_i32 : i32, i32 }
//
//%c_mask_base_4 = vector.extract %8[4, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_4 = vector.extract %offs_cn_mask[4] : i1 from vector<64xi1>
//%c_mask_base4_i32 = arith.select %c_mask_base_4, %c1_i32, %c0_i32 : i32
//%offs_cn_mask4_i32 = arith.select %offs_cn_mask_4, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[4] %d offs_cn_mask[4] %d\\n", %c_mask_base4_i32, %offs_cn_mask4_i32 : i32, i32 }
//
//%c_mask_base_63 = vector.extract %8[63, 0] : i1 from vector<64x1xi1>
//%offs_cn_mask_63 = vector.extract %offs_cn_mask[63] : i1 from vector<64xi1>
//%c_mask_base63_i32 = arith.select %c_mask_base_63, %c1_i32, %c0_i32 : i32
//%offs_cn_mask63_i32 = arith.select %offs_cn_mask_63, %c1_i32, %c0_i32 : i32
//scf.if %print { gpu.printf "c_mask_base[63] %d offs_cn_mask[63] %d\\n", %c_mask_base63_i32, %offs_cn_mask63_i32 : i32, i32 }
//
//%index_c0 = vector.extract %c_row_offsets[0, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[0][0] %d\\n", %index_c0 : index }
//
//%index_c1 = vector.extract %c_row_offsets[1, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[1][0] %d\\n", %index_c1 : index }
//
//%index_c2 = vector.extract %c_row_offsets[2, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[2][0] %d\\n", %index_c2 : index }
//
//%index_c3 = vector.extract %c_row_offsets[3, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[3][0] %d\\n", %index_c3 : index }
//
//%index_c4 = vector.extract %c_row_offsets[4, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[4][0] %d\\n", %index_c4 : index }
//
//%index_c5 = vector.extract %c_row_offsets[5, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[5][0] %d\\n", %index_c5 : index }
//
//%index_c6 = vector.extract %c_row_offsets[6, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[6][0] %d\\n", %index_c6 : index }
//
//%index_c7 = vector.extract %c_row_offsets[7, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[7][0] %d\\n", %index_c7 : index }
//
//%index_c8 = vector.extract %c_row_offsets[8, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[8][0] %d\\n", %index_c8 : index }
//
//%index_c9 = vector.extract %c_row_offsets[9, 0] : index from vector<64x1xindex>
//scf.if %print { gpu.printf "indices_c[9][0] %d\\n", %index_c9 : index }

          // Store results
          vector.scatter %c_flat[%c0][%c_indices], %output_mask, %result_f16 :
              memref<1280xf16>, vector<64x64xindex>, vector<64x64xi1>, vector<64x64xf16>

%c1.0.0 = memref.load %c_ptr[%c1, %c0, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][0][0] %f\\n", %c1.0.0 : f16 }

%c1.0.1 = memref.load %c_ptr[%c1, %c0, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][0][1] %f\\n", %c1.0.1 : f16 }

%c1.0.63 = memref.load %c_ptr[%c1, %c0, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][0][63] %f\\n", %c1.0.63 : f16 }

%c1.1.0 = memref.load %c_ptr[%c1, %c1, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][1][0] %f\\n", %c1.1.0 : f16 }

%c1.1.1 = memref.load %c_ptr[%c1, %c1, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][1][1] %f\\n", %c1.1.1 : f16 }

%c1.1.63 = memref.load %c_ptr[%c1, %c1, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[1][1][63] %f\\n", %c1.1.63 : f16 }

%c2.0.0 = memref.load %c_ptr[%c2, %c0, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[2][0][0] %f\\n", %c2.0.0 : f16 }

%c2.0.1 = memref.load %c_ptr[%c2, %c0, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[2][0][1] %f\\n", %c2.0.1 : f16 }

%c2.0.63 = memref.load %c_ptr[%c2, %c0, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[2][0][63] %f\\n", %c2.0.63 : f16 }

%c5.0.0 = memref.load %c_ptr[%c5, %c0, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][0][0] %f\\n", %c5.0.0 : f16 }

%c5.0.1 = memref.load %c_ptr[%c5, %c0, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][0][1] %f\\n", %c5.0.1 : f16 }

%c5.0.63 = memref.load %c_ptr[%c5, %c0, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][0][63] %f\\n", %c5.0.63 : f16 }

%c5.1.0 = memref.load %c_ptr[%c5, %c1, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][1][0] %f\\n", %c5.1.0 : f16 }

%c5.1.1 = memref.load %c_ptr[%c5, %c1, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][1][1] %f\\n", %c5.1.1 : f16 }

%c5.1.63 = memref.load %c_ptr[%c5, %c1, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[5][1][63] %f\\n", %c5.1.63 : f16 }

%c7.0.0 = memref.load %c_ptr[%c7, %c0, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][0][0] %f\\n", %c7.0.0 : f16 }

%c7.0.1 = memref.load %c_ptr[%c7, %c0, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][0][1] %f\\n", %c7.0.1 : f16 }

%c7.0.63 = memref.load %c_ptr[%c7, %c0, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][0][63] %f\\n", %c7.0.63 : f16 }

%c7.1.0 = memref.load %c_ptr[%c7, %c1, %c0] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][1][0] %f\\n", %c7.1.0 : f16 }

%c7.1.1 = memref.load %c_ptr[%c7, %c1, %c1] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][1][1] %f\\n", %c7.1.1 : f16 }

%c7.1.63 = memref.load %c_ptr[%c7, %c1, %c63] : memref<10x2x64xf16>
scf.if %print { gpu.printf "c[7][1][63] %f\\n", %c7.1.63 : f16 }

          // TODO: use this when vector.extract for 2d vector can be lowered to LLVM.
          // Compute output indices

     //   %top_k_splat = vector.broadcast %top_k : index to vector<64xindex>
     //   %orig_token_ids = arith.divsi %offs_token, %top_k_splat : vector<64xindex>
     //   %expert_slots = arith.remsi %offs_token, %top_k_splat : vector<64xindex>

     //   // Create output mask
     //   %offs_cn_mask = arith.cmpi slt, %offs_cn, %N_splat : vector<64xindex>
     //   %output_mask = arith.andi %offs_cn_mask, %token_mask : vector<64xi1>

     //   // Store results
     //   scf.for %row = %c0 to %BLOCK_SIZE_M step %c1 {
	 //     %row_token_mask = vector.extract %token_mask[%row] : i1 from vector<64xi1>
     //     scf.if %row_token_mask {
     //       %vec = vector.extract %result_f16[%row] : vector<64xf16> from vector<64x64xf16>
     //       %orig_token_id = vector.extract %orig_token_ids[%row] : index from vector<64xindex>
     //       %expert_slot = vector.extract %expert_slots[%row] : index from vector<64xindex>
     //       vector.transfer_write %vec, %c_ptr[%orig_token_id, %expert_slot, %offs_bn_base], %output_mask : vector<64xf16>, memref<10x2x64xf16>
     //       scf.yield
     //     }
     //   }
        }

        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<10x32xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x64x32xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<587xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<10xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<10x2x64xf16>
    %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<10x32xf16>, tensor<8x64x32xf16>, tensor<587xi32>, tensor<10xi32>, tensor<1xi32>, tensor<10x2x64xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<10x2x64xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<10x2x64xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)
    asm_dtype0_256_128_8_64_2_33 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
builtin.module {
  func.func @fused_moe_kernel(
      // Input memrefs
      %a_ptr: memref<33x128xf16>,
      %b_ptr: memref<8x256x128xf16>,
      %sorted_token_ids_ptr: memref<633xi32>,
      %expert_ids_ptr: memref<10xi32>,
      %num_tokens_post_padded_ptr: memref<1xi32>,
      %c_ptr: memref<33x2x256xf16>
  ) attributes {translation_info = #translation} {
    // N = 256
    // K = 128
    // EM = 633
    // top_k = 2
    // num_valid_tokens = 66
    // GROUP_SIZE_M = 8
    // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
    // BLOCK_SIZE_K = 32
    %N = arith.constant 256 : index
    %K = arith.constant 128 : index
    %EM = arith.constant 633 : index
    %top_k = arith.constant 2 : index
    %num_valid_tokens = arith.constant 66 : index
    %GROUP_SIZE_M = arith.constant 8 : index
    %BLOCK_SIZE_M = arith.constant 64 : index
    %BLOCK_SIZE_N = arith.constant 64 : index
    %BLOCK_SIZE_K = arith.constant 32 : index

    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f32
    %f0_f16 = arith.constant 0.0 : f16
    %passthru_a = vector.broadcast %f0_f16 : f16 to vector<64x32xf16>
    %true_mask = arith.constant dense<true> : vector<64xi1>
    %zeroes_64 = arith.constant dense<0> : vector<64xi32>

    // Program ID mapping
    %pid = gpu.block_id x
    %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
    %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
    %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
    %group_id = arith.divui %pid, %num_pid_in_group : index
    %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
    %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
    %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
    %0 = arith.remsi %pid, %num_pid_in_group : index
    %1 = arith.remsi %0, %group_size_m : index
    %pid_m = arith.addi %first_pid_m, %1 : index
    %pid_n = arith.divui %0, %group_size_m : index

    // Early exit check
    %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
    %num_tokens_post_padded = arith.index_cast %2 : i32 to index
    %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
    %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
    scf.if %should_exit {
      scf.yield
    } else {
      // Compute token mask
      %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
      %range_m = vector.step : vector<64xindex>
      %3 = vector.broadcast %offs_token_id_base : index to vector<64xindex>
      %offs_token_id = arith.addi %3, %range_m : vector<64xindex>

      // Load token IDs
      %4 = vector.gather %sorted_token_ids_ptr[%c0] [%offs_token_id],
      	%true_mask, %zeroes_64 : memref<633xi32>, vector<64xindex>, vector<64xi1>, vector<64xi32> into vector<64xi32>
      %offs_token = arith.index_cast %4 : vector<64xi32> to vector<64xindex>

      // Create token mask
      %5 = vector.broadcast %num_valid_tokens : index to vector<64xindex>
      %token_mask = arith.cmpi slt, %offs_token, %5 : vector<64xindex>

      // Load expert ID
      %6 = memref.load %expert_ids_ptr[%pid_m] : memref<10xi32>
      %off_experts = arith.index_cast %6 : i32 to index

      // Setup B matrix pointers
      %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
      %range_n = vector.step : vector<64xindex>
      %7 = vector.broadcast %offs_bn_base : index to vector<64xindex>
      %offs_cn = arith.addi %7, %range_n : vector<64xindex>
      %N_splat = vector.broadcast %N : index to vector<64xindex>
      %offs_bn = arith.remsi %offs_cn, %N_splat : vector<64xindex>
      %off_bn = arith.remsi %offs_bn_base, %N : index

      // Setup K range
      %offs_k = vector.step : vector<32xindex>

      // -----------------------------------------------------------
      // Compute A matrix indices (equivalent to a_ptrs computation)
      // Expand dims: offs_token[:, None] -> 64x1
      %offs_token_2d = vector.shape_cast %offs_token : vector<64xindex> to vector<64x1xindex>

      // Broadcast top_k to 64x1
      %top_k_2d = vector.broadcast %top_k : index to vector<64x1xindex>

      // Compute offs_token[:, None] // top_k
      %token_div_topk = arith.divsi %offs_token_2d, %top_k_2d : vector<64x1xindex>

      // Broadcast K to 64x1
      %K_splat = vector.broadcast %K : index to vector<64x1xindex>

      // Compute first term: (offs_token // top_k) * K
      %a_row_offsets = arith.muli %token_div_topk, %K_splat : vector<64x1xindex>

      // Broadcast both terms to 64x32 and add
      %first_broadcast_a = vector.broadcast %a_row_offsets : vector<64x1xindex> to vector<64x32xindex>
      %second_broadcast_a = vector.broadcast %offs_k : vector<32xindex> to vector<64x32xindex>
      %indices_a = arith.addi %first_broadcast_a, %second_broadcast_a : vector<64x32xindex>

      // -----------------------------------------------------------
      // Initialize accumulator
      %accumulator = vector.broadcast %f0 : f32 to vector<64x64xf32>

      // Main K loop
      %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
      %8 = vector.shape_cast %token_mask : vector<64xi1> to vector<64x1xi1>
      %a_mask_base = vector.broadcast %8 : vector<64x1xi1> to vector<64x32xi1>

      %a_flat = memref.collapse_shape %a_ptr [[0, 1]] : memref<33x128xf16> into memref<4224xf16>
      %b_linear_offset = arith.muli %off_experts, %c32768 : index
      %b_block_transposed = memref.reinterpret_cast %b_ptr to offset: [%b_linear_offset], sizes: [128, 256], strides: [1, 128] : memref<8x256x128xf16> to memref<128x256xf16, strided<[1, 128], offset: ?>>
      %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<33x2x256xf16> into memref<16896xf16>

      %result = scf.for %k_block = %c0 to %num_blocks step %c1 iter_args(%acc = %accumulator) -> vector<64x64xf32> {
        // Compute current K offset
        %k_offset = arith.muli %k_block, %BLOCK_SIZE_K : index

        %k_offset_broadcast = vector.broadcast %k_offset : index to vector<64x32xindex>
        %indices_a_with_k = arith.addi %indices_a, %k_offset_broadcast : vector<64x32xindex>

        %a = vector.gather %a_flat[%c0][%indices_a_with_k], %a_mask_base, %passthru_a :
            memref<4224xf16>, vector<64x32xindex>, vector<64x32xi1>, vector<64x32xf16> into vector<64x32xf16>

        %b = vector.transfer_read %b_block_transposed[%k_offset, %off_bn], %f0_f16 : memref<128x256xf16, strided<[1, 128], offset: ?>>, vector<32x64xf16>

        // Matrix multiplication: f16 x f16 -> f32 accumulation
        %dot_result = vector.contract {
            indexing_maps = [affine_map<(m,n,k) -> (m,k)>,
                           affine_map<(m,n,k) -> (k,n)>,
                           affine_map<(m,n,k) -> (m,n)>],
            iterator_types = ["parallel", "parallel", "reduction"]
        } %a, %b, %acc : vector<64x32xf16>, vector<32x64xf16> into vector<64x64xf32>

        scf.yield %dot_result : vector<64x64xf32>
      }

      // Convert accumulator to output precision
      %result_f16 = arith.truncf %result : vector<64x64xf32> to vector<64x64xf16>

      // Compute output indices
      %offs_cn_2d = vector.shape_cast %offs_cn : vector<64xindex> to vector<1x64xindex>

      %N_2d = vector.broadcast %N : index to vector<64x1xindex>
      %c_row_offsets = arith.muli %offs_token_2d, %N_2d : vector<64x1xindex>

      %first_broadcast_c = vector.broadcast %c_row_offsets : vector<64x1xindex> to vector<64x64xindex>
      %second_broadcast_c = vector.broadcast %offs_cn_2d : vector<1x64xindex> to vector<64x64xindex>
      %c_indices = arith.addi %first_broadcast_c, %second_broadcast_c : vector<64x64xindex>

      // Create output mask
      %offs_cn_mask = arith.cmpi slt, %offs_cn, %N_splat : vector<64xindex>

      %c_mask_base = vector.broadcast %8 : vector<64x1xi1> to vector<64x64xi1>
      %cn_mask_broadcast = vector.broadcast %offs_cn_mask : vector<64xi1> to vector<64x64xi1>
      %output_mask = arith.andi %c_mask_base, %cn_mask_broadcast : vector<64x64xi1>

      // Store results
      vector.scatter %c_flat[%c0][%c_indices], %output_mask, %result_f16 :
          memref<16896xf16>, vector<64x64xindex>, vector<64x64xi1>, vector<64x64xf16>
    }

    return
  }
}
    """
)

    asm_dtype0_256_128_8_64_2_33_mfma = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel {
    stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
      %c40 = arith.constant 40 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c40, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel(
          // Input memrefs
       // %a_ptr: memref<33x128xf16>,
       // %b_ptr: memref<8x256x128xf16>,
       // %sorted_token_ids_ptr: memref<633xi32>,
       // %expert_ids_ptr: memref<10xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<33x2x256xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 256
        // K = 128
        // EM = 633
        // top_k = 2
        // num_valid_tokens = 66
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 256 : index
        %K = arith.constant 128 : index
        %EM = arith.constant 633 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 66 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c16384 = arith.constant 16384 : index
        %c32768 = arith.constant 32768 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c48 = arith.constant 48 : index
        %c63 = arith.constant 63 : index
        %c127 = arith.constant 127 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<33x128xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x256x128xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<33x2x256xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<633xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Program ID mapping
        %pid = gpu.block_id x
        %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
        %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
        %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
        %group_id = arith.divui %pid, %num_pid_in_group : index
        %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
        %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
        %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
        %0 = arith.remsi %pid, %num_pid_in_group : index
        %1 = arith.remsi %0, %group_size_m : index
        %pid_m = arith.addi %first_pid_m, %1 : index
        %pid_n = arith.divui %0, %group_size_m : index

        %thread_id = gpu.thread_id x upper_bound 64

        // Early exit check
        %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
        %num_tokens_post_padded = arith.index_cast %2 : i32 to index
        %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
        %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
        scf.if %should_exit {
          scf.yield
        } else {
          // Compute token mask
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
          %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

          // Load token ID for this row
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<633xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<128xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<10xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64×128 for A, 64×128 for B
          %alloc = memref.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<32768xi8, #gpu.address_space<workgroup>>
            to memref<64x128xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c16384][] : memref<32768xi8, #gpu.address_space<workgroup>>
            to memref<64x128xf16, #gpu.address_space<workgroup>>
//%alloc_c = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
//%shared_c = memref.view %alloc_c[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
//  to memref<64x64xf16, #gpu.address_space<workgroup>>

          // Each thread loads its full row from A (128 f16)
          %a_row_vec = vector.transfer_read %a_ptr[%a_row, %c0], %f0_f16, %token_mask :
            memref<33x128xf16>, vector<128xf16>
          // Store to shared memory
          vector.store %a_row_vec, %shared_a[%thread_id, %c0] :
            memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

          // Each thread loads its row from B (128 f16)
          // B is [8, 256, 128], we need [expert_id, b_row, :]
          // Note: b_row is always < 256 since pid_n * 64 + thread_id_x < 256
          %b_row_vec = vector.transfer_read %b_ptr[%expert_id, %b_row, %c0], %f0_f16 :
            memref<8x256x128xf16>, vector<128xf16>
          // Store to shared memory
          vector.store %b_row_vec, %shared_b[%thread_id, %c0] :
            memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

          amdgpu.lds_barrier

//%print = arith.cmpi eq, %thread_id, %c1 : index
//
//scf.if %print { gpu.printf "pid %d\\n", %pid : index }
//scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
//scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
//
//%a0.0 = memref.load %shared_a[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }
//
//%a0.1 = memref.load %shared_a[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }
//
//%a0.127 = memref.load %shared_a[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][127] %f\\n", %a0.127 : f16 }
//
//%a1.0 = memref.load %shared_a[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }
//
//%a1.1 = memref.load %shared_a[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }
//
//%a1.127 = memref.load %shared_a[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][127] %f\\n", %a1.127 : f16 }
//
//%a63.0 = memref.load %shared_a[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }
//
//%a63.1 = memref.load %shared_a[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }
//
//%a63.127 = memref.load %shared_a[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][127] %f\\n", %a63.127 : f16 }
//
//%b0.0 = memref.load %shared_b[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }
//
//%b0.1 = memref.load %shared_b[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }
//
//%b0.127 = memref.load %shared_b[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][127] %f\\n", %b0.127 : f16 }
//
//%b1.0 = memref.load %shared_b[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }
//
//%b1.1 = memref.load %shared_b[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }
//
//%b1.127 = memref.load %shared_b[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][127] %f\\n", %b1.127 : f16 }
//
//%b63.0 = memref.load %shared_b[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][0] %f\\n", %b63.0 : f16 }
//
//%b63.1 = memref.load %shared_b[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][1] %f\\n", %b63.1 : f16 }
//
//%b63.127 = memref.load %shared_b[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][127] %f\\n", %b63.127 : f16 }
//
//amdgpu.lds_barrier
          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index
//gpu.printf "T%d load_col %d load_row %d\\n", %thread_id, %load_col, %load_row : index, index, index

          // =========================================================================
          // MFMA COMPUTATION
          // =========================================================================
          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index

          %result:16 = scf.for %k_block = %c0 to %num_blocks step %c1
              iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                        %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                        %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                        %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
              -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

            // Compute K offset for this iteration
            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
            %k_col = arith.addi %k_start, %load_col : index
            %k_col_k = arith.addi %k_col, %c16 : index

            // Load A vectors: 4 M tiles × 2 K slices (columns k_col and k_col+16)
            %a0 = vector.load %shared_a[%load_row, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a0k = vector.load %shared_a[%load_row, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors: 4 N tiles × 2 K slices
            // Note: B is stored as [64, 128] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[row, k_col]
            %b0 = vector.load %shared_b[%load_row, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k = vector.load %shared_b[%load_row, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // MFMA operations: 4×4 tile grid
            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %result#0 : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %result#1 : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %result#2 : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %result#3 : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %result#4 : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %result#5 : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %result#6 : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %result#7 : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %result#8 : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %result#9 : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %result#10 : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %result#11 : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %result#12 : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %result#13 : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %result#14 : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %result#15 : vector<4xf32> to vector<4xf16>

          %store_col_0 = affine.apply #map_store_col()[%thread_id]
          %store_col_1 = arith.addi %store_col_0, %c16 : index
          %store_col_2 = arith.addi %store_col_0, %c32 : index
          %store_col_3 = arith.addi %store_col_0, %c48 : index
          %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
          %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
          %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
          %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
          %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
          %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
          %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
          %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
          %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
          %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
          %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
          %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
          %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
          %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
          %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
          %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

          %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

//vector.store %r00_0_f16, %shared_c[%store_row_0_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_1_f16, %shared_c[%store_row_0_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_2_f16, %shared_c[%store_row_0_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_3_f16, %shared_c[%store_row_0_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_0_f16, %shared_c[%store_row_0_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_1_f16, %shared_c[%store_row_0_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_2_f16, %shared_c[%store_row_0_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_3_f16, %shared_c[%store_row_0_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_0_f16, %shared_c[%store_row_0_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_1_f16, %shared_c[%store_row_0_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_2_f16, %shared_c[%store_row_0_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_3_f16, %shared_c[%store_row_0_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_0_f16, %shared_c[%store_row_0_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_1_f16, %shared_c[%store_row_0_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_2_f16, %shared_c[%store_row_0_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_3_f16, %shared_c[%store_row_0_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r10_0_f16, %shared_c[%store_row_16_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_1_f16, %shared_c[%store_row_16_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_2_f16, %shared_c[%store_row_16_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_3_f16, %shared_c[%store_row_16_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_0_f16, %shared_c[%store_row_16_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_1_f16, %shared_c[%store_row_16_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_2_f16, %shared_c[%store_row_16_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_3_f16, %shared_c[%store_row_16_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_0_f16, %shared_c[%store_row_16_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_1_f16, %shared_c[%store_row_16_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_2_f16, %shared_c[%store_row_16_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_3_f16, %shared_c[%store_row_16_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_0_f16, %shared_c[%store_row_16_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_1_f16, %shared_c[%store_row_16_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_2_f16, %shared_c[%store_row_16_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_3_f16, %shared_c[%store_row_16_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r20_0_f16, %shared_c[%store_row_32_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_1_f16, %shared_c[%store_row_32_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_2_f16, %shared_c[%store_row_32_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_3_f16, %shared_c[%store_row_32_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_0_f16, %shared_c[%store_row_32_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_1_f16, %shared_c[%store_row_32_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_2_f16, %shared_c[%store_row_32_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_3_f16, %shared_c[%store_row_32_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_0_f16, %shared_c[%store_row_32_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_1_f16, %shared_c[%store_row_32_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_2_f16, %shared_c[%store_row_32_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_3_f16, %shared_c[%store_row_32_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_0_f16, %shared_c[%store_row_32_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_1_f16, %shared_c[%store_row_32_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_2_f16, %shared_c[%store_row_32_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_3_f16, %shared_c[%store_row_32_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r30_0_f16, %shared_c[%store_row_48_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_1_f16, %shared_c[%store_row_48_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_2_f16, %shared_c[%store_row_48_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_3_f16, %shared_c[%store_row_48_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_0_f16, %shared_c[%store_row_48_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_1_f16, %shared_c[%store_row_48_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_2_f16, %shared_c[%store_row_48_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_3_f16, %shared_c[%store_row_48_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_0_f16, %shared_c[%store_row_48_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_1_f16, %shared_c[%store_row_48_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_2_f16, %shared_c[%store_row_48_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_3_f16, %shared_c[%store_row_48_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_0_f16, %shared_c[%store_row_48_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_1_f16, %shared_c[%store_row_48_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_2_f16, %shared_c[%store_row_48_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_3_f16, %shared_c[%store_row_48_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//%c0.0 = memref.load %shared_c[%c0, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][0] %f\\n", %c0.0 : f16 }
//
//%c0.1 = memref.load %shared_c[%c0, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][1] %f\\n", %c0.1 : f16 }
//
//%c0.63 = memref.load %shared_c[%c0, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][63] %f\\n", %c0.63 : f16 }
//
//%c1.0 = memref.load %shared_c[%c1, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][0] %f\\n", %c1.0 : f16 }
//
//%c1.1 = memref.load %shared_c[%c1, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][1] %f\\n", %c1.1 : f16 }
//
//%c1.63 = memref.load %shared_c[%c1, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][63] %f\\n", %c1.63 : f16 }
//
//%c2.0 = memref.load %shared_c[%c2, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][0] %f\\n", %c2.0 : f16 }
//
//%c2.1 = memref.load %shared_c[%c2, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][1] %f\\n", %c2.1 : f16 }
//
//%c2.63 = memref.load %shared_c[%c2, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][63] %f\\n", %c2.63 : f16 }
//
//%c63.0 = memref.load %shared_c[%c63, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][0] %f\\n", %c63.0 : f16 }
//
//%c63.1 = memref.load %shared_c[%c63, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][1] %f\\n", %c63.1 : f16 }
//
//%c63.63 = memref.load %shared_c[%c63, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][63] %f\\n", %c63.63 : f16 }
//
//amdgpu.lds_barrier

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<33x2x256xf16> into memref<16896xf16>

          // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
          // across 4 column groups (base, base+16, base+32, base+48)

          // Get token indices for output rows
          %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
          %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
          %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
          %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
          %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
          %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
          %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
          %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
          %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
          %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
          %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
          %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
          %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
          %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
          %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
          %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

          %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<633xi32>
          %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
          %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
          %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<633xi32>
          %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
          %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
          %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<633xi32>
          %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
          %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
          %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<633xi32>
          %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
          %out_base_0_3 = arith.muli %tok_id_0_3, %N : index

          %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<633xi32>
          %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
          %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
          %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<633xi32>
          %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
          %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
          %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<633xi32>
          %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
          %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
          %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<633xi32>
          %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
          %out_base_16_3 = arith.muli %tok_id_16_3, %N : index

          %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<633xi32>
          %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
          %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
          %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<633xi32>
          %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
          %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
          %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<633xi32>
          %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
          %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
          %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<633xi32>
          %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
          %out_base_32_3 = arith.muli %tok_id_32_3, %N : index

          %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<633xi32>
          %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
          %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
          %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<633xi32>
          %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
          %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
          %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<633xi32>
          %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
          %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
          %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<633xi32>
          %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
          %out_base_48_3 = arith.muli %tok_id_48_3, %N : index

          // pid_n determines which 64-neuron block we're computing
          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

          // Column offsets for the 4 column tiles
          %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
          %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
          %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
          %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

          // Write all 16 tiles using vector.store
          // Tile (0,0)
          %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
          vector.store %r00_0_f16, %c_flat[%idx_00_0] : memref<16896xf16>, vector<1xf16>
          %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
          vector.store %r00_1_f16, %c_flat[%idx_00_1] : memref<16896xf16>, vector<1xf16>
          %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
          vector.store %r00_2_f16, %c_flat[%idx_00_2] : memref<16896xf16>, vector<1xf16>
          %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
          vector.store %r00_3_f16, %c_flat[%idx_00_3] : memref<16896xf16>, vector<1xf16>

          // Tile (0,1)
          %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
          vector.store %r01_0_f16, %c_flat[%idx_01_0] : memref<16896xf16>, vector<1xf16>
          %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
          vector.store %r01_1_f16, %c_flat[%idx_01_1] : memref<16896xf16>, vector<1xf16>
          %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
          vector.store %r01_2_f16, %c_flat[%idx_01_2] : memref<16896xf16>, vector<1xf16>
          %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
          vector.store %r01_3_f16, %c_flat[%idx_01_3] : memref<16896xf16>, vector<1xf16>

          // Tile (0,2)
          %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
          vector.store %r02_0_f16, %c_flat[%idx_02_0] : memref<16896xf16>, vector<1xf16>
          %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
          vector.store %r02_1_f16, %c_flat[%idx_02_1] : memref<16896xf16>, vector<1xf16>
          %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
          vector.store %r02_2_f16, %c_flat[%idx_02_2] : memref<16896xf16>, vector<1xf16>
          %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
          vector.store %r02_3_f16, %c_flat[%idx_02_3] : memref<16896xf16>, vector<1xf16>

          // Tile (0,3)
          %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
          vector.store %r03_0_f16, %c_flat[%idx_03_0] : memref<16896xf16>, vector<1xf16>
          %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
          vector.store %r03_1_f16, %c_flat[%idx_03_1] : memref<16896xf16>, vector<1xf16>
          %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
          vector.store %r03_2_f16, %c_flat[%idx_03_2] : memref<16896xf16>, vector<1xf16>
          %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
          vector.store %r03_3_f16, %c_flat[%idx_03_3] : memref<16896xf16>, vector<1xf16>

          // Tile (1,0)
          %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
          vector.store %r10_0_f16, %c_flat[%idx_10_0] : memref<16896xf16>, vector<1xf16>
          %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
          vector.store %r10_1_f16, %c_flat[%idx_10_1] : memref<16896xf16>, vector<1xf16>
          %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
          vector.store %r10_2_f16, %c_flat[%idx_10_2] : memref<16896xf16>, vector<1xf16>
          %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
          vector.store %r10_3_f16, %c_flat[%idx_10_3] : memref<16896xf16>, vector<1xf16>

          // Tile (1,1)
          %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
          vector.store %r11_0_f16, %c_flat[%idx_11_0] : memref<16896xf16>, vector<1xf16>
          %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
          vector.store %r11_1_f16, %c_flat[%idx_11_1] : memref<16896xf16>, vector<1xf16>
          %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
          vector.store %r11_2_f16, %c_flat[%idx_11_2] : memref<16896xf16>, vector<1xf16>
          %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
          vector.store %r11_3_f16, %c_flat[%idx_11_3] : memref<16896xf16>, vector<1xf16>

          // Tile (1,2)
          %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
          vector.store %r12_0_f16, %c_flat[%idx_12_0] : memref<16896xf16>, vector<1xf16>
          %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
          vector.store %r12_1_f16, %c_flat[%idx_12_1] : memref<16896xf16>, vector<1xf16>
          %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
          vector.store %r12_2_f16, %c_flat[%idx_12_2] : memref<16896xf16>, vector<1xf16>
          %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
          vector.store %r12_3_f16, %c_flat[%idx_12_3] : memref<16896xf16>, vector<1xf16>

          // Tile (1,3)
          %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
          vector.store %r13_0_f16, %c_flat[%idx_13_0] : memref<16896xf16>, vector<1xf16>
          %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
          vector.store %r13_1_f16, %c_flat[%idx_13_1] : memref<16896xf16>, vector<1xf16>
          %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
          vector.store %r13_2_f16, %c_flat[%idx_13_2] : memref<16896xf16>, vector<1xf16>
          %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
          vector.store %r13_3_f16, %c_flat[%idx_13_3] : memref<16896xf16>, vector<1xf16>

          // Tile (2,0)
          %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
          vector.store %r20_0_f16, %c_flat[%idx_20_0] : memref<16896xf16>, vector<1xf16>
          %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
          vector.store %r20_1_f16, %c_flat[%idx_20_1] : memref<16896xf16>, vector<1xf16>
          %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
          vector.store %r20_2_f16, %c_flat[%idx_20_2] : memref<16896xf16>, vector<1xf16>
          %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
          vector.store %r20_3_f16, %c_flat[%idx_20_3] : memref<16896xf16>, vector<1xf16>

          // Tile (2,1)
          %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
          vector.store %r21_0_f16, %c_flat[%idx_21_0] : memref<16896xf16>, vector<1xf16>
          %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
          vector.store %r21_1_f16, %c_flat[%idx_21_1] : memref<16896xf16>, vector<1xf16>
          %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
          vector.store %r21_2_f16, %c_flat[%idx_21_2] : memref<16896xf16>, vector<1xf16>
          %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
          vector.store %r21_3_f16, %c_flat[%idx_21_3] : memref<16896xf16>, vector<1xf16>

          // Tile (2,2)
          %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
          vector.store %r22_0_f16, %c_flat[%idx_22_0] : memref<16896xf16>, vector<1xf16>
          %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
          vector.store %r22_1_f16, %c_flat[%idx_22_1] : memref<16896xf16>, vector<1xf16>
          %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
          vector.store %r22_2_f16, %c_flat[%idx_22_2] : memref<16896xf16>, vector<1xf16>
          %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
          vector.store %r22_3_f16, %c_flat[%idx_22_3] : memref<16896xf16>, vector<1xf16>

          // Tile (2,3)
          %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
          vector.store %r23_0_f16, %c_flat[%idx_23_0] : memref<16896xf16>, vector<1xf16>
          %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
          vector.store %r23_1_f16, %c_flat[%idx_23_1] : memref<16896xf16>, vector<1xf16>
          %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
          vector.store %r23_2_f16, %c_flat[%idx_23_2] : memref<16896xf16>, vector<1xf16>
          %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
          vector.store %r23_3_f16, %c_flat[%idx_23_3] : memref<16896xf16>, vector<1xf16>

          // Tile (3,0)
          %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
          vector.store %r30_0_f16, %c_flat[%idx_30_0] : memref<16896xf16>, vector<1xf16>
          %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
          vector.store %r30_1_f16, %c_flat[%idx_30_1] : memref<16896xf16>, vector<1xf16>
          %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
          vector.store %r30_2_f16, %c_flat[%idx_30_2] : memref<16896xf16>, vector<1xf16>
          %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
          vector.store %r30_3_f16, %c_flat[%idx_30_3] : memref<16896xf16>, vector<1xf16>

          // Tile (3,1)
          %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
          vector.store %r31_0_f16, %c_flat[%idx_31_0] : memref<16896xf16>, vector<1xf16>
          %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
          vector.store %r31_1_f16, %c_flat[%idx_31_1] : memref<16896xf16>, vector<1xf16>
          %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
          vector.store %r31_2_f16, %c_flat[%idx_31_2] : memref<16896xf16>, vector<1xf16>
          %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
          vector.store %r31_3_f16, %c_flat[%idx_31_3] : memref<16896xf16>, vector<1xf16>

          // Tile (3,2)
          %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
          vector.store %r32_0_f16, %c_flat[%idx_32_0] : memref<16896xf16>, vector<1xf16>
          %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
          vector.store %r32_1_f16, %c_flat[%idx_32_1] : memref<16896xf16>, vector<1xf16>
          %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
          vector.store %r32_2_f16, %c_flat[%idx_32_2] : memref<16896xf16>, vector<1xf16>
          %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
          vector.store %r32_3_f16, %c_flat[%idx_32_3] : memref<16896xf16>, vector<1xf16>

          // Tile (3,3)
          %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
          vector.store %r33_0_f16, %c_flat[%idx_33_0] : memref<16896xf16>, vector<1xf16>
          %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
          vector.store %r33_1_f16, %c_flat[%idx_33_1] : memref<16896xf16>, vector<1xf16>
          %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
          vector.store %r33_2_f16, %c_flat[%idx_33_2] : memref<16896xf16>, vector<1xf16>
          %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
          vector.store %r33_3_f16, %c_flat[%idx_33_3] : memref<16896xf16>, vector<1xf16>
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<33x128xf16>,
       // %b_ptr: memref<8x256x128xf16>,
       // %sorted_token_ids_ptr: memref<633xi32>,
       // %expert_ids_ptr: memref<10xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<33x2x256xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<33x128xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x256x128xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<633xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<10xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<33x2x256xf16>
    %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<33x128xf16>, tensor<8x256x128xf16>, tensor<633xi32>, tensor<10xi32>, tensor<1xi32>, tensor<33x2x256xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<33x2x256xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<33x2x256xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

    asm_dtype0_1024_128_8_64_2_2048_mfma = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel {
    stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
      %c1168 = arith.constant 1168 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c1168, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel(
          // Input memrefs
       // %a_ptr: memref<2048x128xf16>,
       // %b_ptr: memref<8x1024x128xf16>,
       // %sorted_token_ids_ptr: memref<4663xi32>,
       // %expert_ids_ptr: memref<10xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<2048x2x1024xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 1024
        // K = 128
        // EM = 4663
        // top_k = 2
        // num_valid_tokens = 4096
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 1024 : index
        %K = arith.constant 128 : index
        %EM = arith.constant 4663 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 4096 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c16384 = arith.constant 16384 : index
        %c32768 = arith.constant 32768 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %c48 = arith.constant 48 : index
        %c63 = arith.constant 63 : index
        %c127 = arith.constant 127 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2048x128xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x1024x128xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<2048x2x1024xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<4663xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<73xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Program ID mapping
        %pid = gpu.block_id x
        %num_pid_m = arith.ceildivui %EM, %BLOCK_SIZE_M : index
        %num_pid_n = arith.ceildivui %N, %BLOCK_SIZE_N : index
        %num_pid_in_group = arith.muli %GROUP_SIZE_M, %num_pid_n : index
        %group_id = arith.divui %pid, %num_pid_in_group : index
        %first_pid_m = arith.muli %group_id, %GROUP_SIZE_M : index
        %min_group_size_m = arith.subi %num_pid_m, %first_pid_m : index
        %group_size_m = arith.minui %GROUP_SIZE_M, %min_group_size_m : index
        %0 = arith.remsi %pid, %num_pid_in_group : index
        %1 = arith.remsi %0, %group_size_m : index
        %pid_m = arith.addi %first_pid_m, %1 : index
        %pid_n = arith.divui %0, %group_size_m : index

        %thread_id = gpu.thread_id x upper_bound 64

        // Early exit check
        %2 = memref.load %num_tokens_post_padded_ptr[%c0] : memref<1xi32>
        %num_tokens_post_padded = arith.index_cast %2 : i32 to index
        %pid_m_offset = arith.muli %pid_m, %BLOCK_SIZE_M : index
        %should_exit = arith.cmpi sge, %pid_m_offset, %num_tokens_post_padded : index
        scf.if %should_exit {
          scf.yield
        } else {
          // Compute token mask
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
          %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

          // Load token ID for this row
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<4663xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<128xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<73xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64×128 for A, 64×128 for B
          %alloc = memref.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<32768xi8, #gpu.address_space<workgroup>>
            to memref<64x128xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c16384][] : memref<32768xi8, #gpu.address_space<workgroup>>
            to memref<64x128xf16, #gpu.address_space<workgroup>>
//%alloc_c = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
//%shared_c = memref.view %alloc_c[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
//  to memref<64x64xf16, #gpu.address_space<workgroup>>

          // Each thread loads its full row from A (128 f16)
          %a_row_vec = vector.transfer_read %a_ptr[%a_row, %c0], %f0_f16, %token_mask :
            memref<2048x128xf16>, vector<128xf16>
          // Store to shared memory
          vector.store %a_row_vec, %shared_a[%thread_id, %c0] :
            memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

          // Each thread loads its row from B (128 f16)
          // B is [8, 1024, 128], we need [expert_id, b_row, :]
          // Note: b_row is always < 1024 since pid_n * 64 + thread_id_x < 1024
          %b_row_vec = vector.transfer_read %b_ptr[%expert_id, %b_row, %c0], %f0_f16 :
            memref<8x1024x128xf16>, vector<128xf16>
          // Store to shared memory
          vector.store %b_row_vec, %shared_b[%thread_id, %c0] :
            memref<64x128xf16, #gpu.address_space<workgroup>>, vector<128xf16>

          amdgpu.lds_barrier

//%print = arith.cmpi eq, %thread_id, %c1 : index
//
//scf.if %print { gpu.printf "pid %d\\n", %pid : index }
//scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
//scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
//
//%a0.0 = memref.load %shared_a[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }
//
//%a0.1 = memref.load %shared_a[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }
//
//%a0.127 = memref.load %shared_a[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][127] %f\\n", %a0.127 : f16 }
//
//%a1.0 = memref.load %shared_a[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }
//
//%a1.1 = memref.load %shared_a[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }
//
//%a1.127 = memref.load %shared_a[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][127] %f\\n", %a1.127 : f16 }
//
//%a63.0 = memref.load %shared_a[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }
//
//%a63.1 = memref.load %shared_a[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }
//
//%a63.127 = memref.load %shared_a[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][127] %f\\n", %a63.127 : f16 }
//
//%b0.0 = memref.load %shared_b[%c0, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }
//
//%b0.1 = memref.load %shared_b[%c0, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }
//
//%b0.127 = memref.load %shared_b[%c0, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][127] %f\\n", %b0.127 : f16 }
//
//%b1.0 = memref.load %shared_b[%c1, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }
//
//%b1.1 = memref.load %shared_b[%c1, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }
//
//%b1.127 = memref.load %shared_b[%c1, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][127] %f\\n", %b1.127 : f16 }
//
//%b63.0 = memref.load %shared_b[%c63, %c0] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][0] %f\\n", %b63.0 : f16 }
//
//%b63.1 = memref.load %shared_b[%c63, %c1] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][1] %f\\n", %b63.1 : f16 }
//
//%b63.127 = memref.load %shared_b[%c63, %c127] : memref<64x128xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][127] %f\\n", %b63.127 : f16 }
//
//amdgpu.lds_barrier
          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index
//gpu.printf "T%d load_col %d load_row %d\\n", %thread_id, %load_col, %load_row : index, index, index

          // =========================================================================
          // MFMA COMPUTATION
          // =========================================================================
          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index

          %result:16 = scf.for %k_block = %c0 to %num_blocks step %c1
              iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                        %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                        %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                        %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
              -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

            // Compute K offset for this iteration
            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
            %k_col = arith.addi %k_start, %load_col : index
            %k_col_k = arith.addi %k_col, %c16 : index

            // Load A vectors: 4 M tiles × 2 K slices (columns k_col and k_col+16)
            %a0 = vector.load %shared_a[%load_row, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a0k = vector.load %shared_a[%load_row, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors: 4 N tiles × 2 K slices
            // Note: B is stored as [64, 128] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[row, k_col]
            %b0 = vector.load %shared_b[%load_row, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %k_col] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k = vector.load %shared_b[%load_row, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %k_col_k] :
                memref<64x128xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // MFMA operations: 4×4 tile grid
            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %result#0 : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %result#1 : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %result#2 : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %result#3 : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %result#4 : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %result#5 : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %result#6 : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %result#7 : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %result#8 : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %result#9 : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %result#10 : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %result#11 : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %result#12 : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %result#13 : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %result#14 : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %result#15 : vector<4xf32> to vector<4xf16>

          %store_col_0 = affine.apply #map_store_col()[%thread_id]
          %store_col_1 = arith.addi %store_col_0, %c16 : index
          %store_col_2 = arith.addi %store_col_0, %c32 : index
          %store_col_3 = arith.addi %store_col_0, %c48 : index
          %store_row_0_0 = affine.apply #map_store_row()[%thread_id]
          %store_row_0_1 = arith.addi %store_row_0_0, %c1 : index
          %store_row_0_2 = arith.addi %store_row_0_0, %c2 : index
          %store_row_0_3 = arith.addi %store_row_0_0, %c3 : index
          %store_row_16_0 = arith.addi %store_row_0_0, %c16 : index
          %store_row_16_1 = arith.addi %store_row_16_0, %c1 : index
          %store_row_16_2 = arith.addi %store_row_16_0, %c2 : index
          %store_row_16_3 = arith.addi %store_row_16_0, %c3 : index
          %store_row_32_0 = arith.addi %store_row_0_0, %c32 : index
          %store_row_32_1 = arith.addi %store_row_32_0, %c1 : index
          %store_row_32_2 = arith.addi %store_row_32_0, %c2 : index
          %store_row_32_3 = arith.addi %store_row_32_0, %c3 : index
          %store_row_48_0 = arith.addi %store_row_0_0, %c48 : index
          %store_row_48_1 = arith.addi %store_row_48_0, %c1 : index
          %store_row_48_2 = arith.addi %store_row_48_0, %c2 : index
          %store_row_48_3 = arith.addi %store_row_48_0, %c3 : index

          %r00_0_f16 = vector.extract_strided_slice %r00_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_1_f16 = vector.extract_strided_slice %r00_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_2_f16 = vector.extract_strided_slice %r00_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r00_3_f16 = vector.extract_strided_slice %r00_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_0_f16 = vector.extract_strided_slice %r01_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_1_f16 = vector.extract_strided_slice %r01_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_2_f16 = vector.extract_strided_slice %r01_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r01_3_f16 = vector.extract_strided_slice %r01_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_0_f16 = vector.extract_strided_slice %r02_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_1_f16 = vector.extract_strided_slice %r02_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_2_f16 = vector.extract_strided_slice %r02_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r02_3_f16 = vector.extract_strided_slice %r02_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_0_f16 = vector.extract_strided_slice %r03_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_1_f16 = vector.extract_strided_slice %r03_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_2_f16 = vector.extract_strided_slice %r03_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r03_3_f16 = vector.extract_strided_slice %r03_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r10_0_f16 = vector.extract_strided_slice %r10_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_1_f16 = vector.extract_strided_slice %r10_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_2_f16 = vector.extract_strided_slice %r10_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r10_3_f16 = vector.extract_strided_slice %r10_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_0_f16 = vector.extract_strided_slice %r11_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_1_f16 = vector.extract_strided_slice %r11_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_2_f16 = vector.extract_strided_slice %r11_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r11_3_f16 = vector.extract_strided_slice %r11_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_0_f16 = vector.extract_strided_slice %r12_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_1_f16 = vector.extract_strided_slice %r12_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_2_f16 = vector.extract_strided_slice %r12_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r12_3_f16 = vector.extract_strided_slice %r12_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_0_f16 = vector.extract_strided_slice %r13_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_1_f16 = vector.extract_strided_slice %r13_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_2_f16 = vector.extract_strided_slice %r13_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r13_3_f16 = vector.extract_strided_slice %r13_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r20_0_f16 = vector.extract_strided_slice %r20_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_1_f16 = vector.extract_strided_slice %r20_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_2_f16 = vector.extract_strided_slice %r20_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r20_3_f16 = vector.extract_strided_slice %r20_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_0_f16 = vector.extract_strided_slice %r21_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_1_f16 = vector.extract_strided_slice %r21_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_2_f16 = vector.extract_strided_slice %r21_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r21_3_f16 = vector.extract_strided_slice %r21_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_0_f16 = vector.extract_strided_slice %r22_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_1_f16 = vector.extract_strided_slice %r22_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_2_f16 = vector.extract_strided_slice %r22_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r22_3_f16 = vector.extract_strided_slice %r22_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_0_f16 = vector.extract_strided_slice %r23_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_1_f16 = vector.extract_strided_slice %r23_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_2_f16 = vector.extract_strided_slice %r23_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r23_3_f16 = vector.extract_strided_slice %r23_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

          %r30_0_f16 = vector.extract_strided_slice %r30_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_1_f16 = vector.extract_strided_slice %r30_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_2_f16 = vector.extract_strided_slice %r30_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r30_3_f16 = vector.extract_strided_slice %r30_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_0_f16 = vector.extract_strided_slice %r31_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_1_f16 = vector.extract_strided_slice %r31_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_2_f16 = vector.extract_strided_slice %r31_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r31_3_f16 = vector.extract_strided_slice %r31_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_0_f16 = vector.extract_strided_slice %r32_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_1_f16 = vector.extract_strided_slice %r32_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_2_f16 = vector.extract_strided_slice %r32_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r32_3_f16 = vector.extract_strided_slice %r32_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_0_f16 = vector.extract_strided_slice %r33_f16 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_1_f16 = vector.extract_strided_slice %r33_f16 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_2_f16 = vector.extract_strided_slice %r33_f16 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %r33_3_f16 = vector.extract_strided_slice %r33_f16 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>

//vector.store %r00_0_f16, %shared_c[%store_row_0_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_1_f16, %shared_c[%store_row_0_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_2_f16, %shared_c[%store_row_0_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r00_3_f16, %shared_c[%store_row_0_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_0_f16, %shared_c[%store_row_0_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_1_f16, %shared_c[%store_row_0_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_2_f16, %shared_c[%store_row_0_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r01_3_f16, %shared_c[%store_row_0_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_0_f16, %shared_c[%store_row_0_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_1_f16, %shared_c[%store_row_0_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_2_f16, %shared_c[%store_row_0_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r02_3_f16, %shared_c[%store_row_0_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_0_f16, %shared_c[%store_row_0_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_1_f16, %shared_c[%store_row_0_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_2_f16, %shared_c[%store_row_0_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r03_3_f16, %shared_c[%store_row_0_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r10_0_f16, %shared_c[%store_row_16_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_1_f16, %shared_c[%store_row_16_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_2_f16, %shared_c[%store_row_16_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r10_3_f16, %shared_c[%store_row_16_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_0_f16, %shared_c[%store_row_16_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_1_f16, %shared_c[%store_row_16_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_2_f16, %shared_c[%store_row_16_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r11_3_f16, %shared_c[%store_row_16_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_0_f16, %shared_c[%store_row_16_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_1_f16, %shared_c[%store_row_16_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_2_f16, %shared_c[%store_row_16_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r12_3_f16, %shared_c[%store_row_16_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_0_f16, %shared_c[%store_row_16_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_1_f16, %shared_c[%store_row_16_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_2_f16, %shared_c[%store_row_16_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r13_3_f16, %shared_c[%store_row_16_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r20_0_f16, %shared_c[%store_row_32_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_1_f16, %shared_c[%store_row_32_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_2_f16, %shared_c[%store_row_32_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r20_3_f16, %shared_c[%store_row_32_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_0_f16, %shared_c[%store_row_32_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_1_f16, %shared_c[%store_row_32_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_2_f16, %shared_c[%store_row_32_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r21_3_f16, %shared_c[%store_row_32_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_0_f16, %shared_c[%store_row_32_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_1_f16, %shared_c[%store_row_32_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_2_f16, %shared_c[%store_row_32_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r22_3_f16, %shared_c[%store_row_32_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_0_f16, %shared_c[%store_row_32_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_1_f16, %shared_c[%store_row_32_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_2_f16, %shared_c[%store_row_32_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r23_3_f16, %shared_c[%store_row_32_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//vector.store %r30_0_f16, %shared_c[%store_row_48_0, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_1_f16, %shared_c[%store_row_48_1, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_2_f16, %shared_c[%store_row_48_2, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r30_3_f16, %shared_c[%store_row_48_3, %store_col_0] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_0_f16, %shared_c[%store_row_48_0, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_1_f16, %shared_c[%store_row_48_1, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_2_f16, %shared_c[%store_row_48_2, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r31_3_f16, %shared_c[%store_row_48_3, %store_col_1] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_0_f16, %shared_c[%store_row_48_0, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_1_f16, %shared_c[%store_row_48_1, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_2_f16, %shared_c[%store_row_48_2, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r32_3_f16, %shared_c[%store_row_48_3, %store_col_2] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_0_f16, %shared_c[%store_row_48_0, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_1_f16, %shared_c[%store_row_48_1, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_2_f16, %shared_c[%store_row_48_2, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//vector.store %r33_3_f16, %shared_c[%store_row_48_3, %store_col_3] : memref<64x64xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//
//%c0.0 = memref.load %shared_c[%c0, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][0] %f\\n", %c0.0 : f16 }
//
//%c0.1 = memref.load %shared_c[%c0, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][1] %f\\n", %c0.1 : f16 }
//
//%c0.63 = memref.load %shared_c[%c0, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[0][63] %f\\n", %c0.63 : f16 }
//
//%c1.0 = memref.load %shared_c[%c1, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][0] %f\\n", %c1.0 : f16 }
//
//%c1.1 = memref.load %shared_c[%c1, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][1] %f\\n", %c1.1 : f16 }
//
//%c1.63 = memref.load %shared_c[%c1, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[1][63] %f\\n", %c1.63 : f16 }
//
//%c2.0 = memref.load %shared_c[%c2, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][0] %f\\n", %c2.0 : f16 }
//
//%c2.1 = memref.load %shared_c[%c2, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][1] %f\\n", %c2.1 : f16 }
//
//%c2.63 = memref.load %shared_c[%c2, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[2][63] %f\\n", %c2.63 : f16 }
//
//%c63.0 = memref.load %shared_c[%c63, %c0] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][0] %f\\n", %c63.0 : f16 }
//
//%c63.1 = memref.load %shared_c[%c63, %c1] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][1] %f\\n", %c63.1 : f16 }
//
//%c63.63 = memref.load %shared_c[%c63, %c63] : memref<64x64xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "c[63][63] %f\\n", %c63.63 : f16 }
//
//amdgpu.lds_barrier

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<2048x2x1024xf16> into memref<4194304xf16>

          // Each thread writes to 4 different rows (from load_row, load_row+16, load_row+32, load_row+48)
          // across 4 column groups (base, base+16, base+32, base+48)

          // Get token indices for output rows
          %out_token_0_0 = arith.addi %offs_token_id_base, %store_row_0_0 : index
          %out_token_0_1 = arith.addi %offs_token_id_base, %store_row_0_1 : index
          %out_token_0_2 = arith.addi %offs_token_id_base, %store_row_0_2 : index
          %out_token_0_3 = arith.addi %offs_token_id_base, %store_row_0_3 : index
          %out_token_16_0 = arith.addi %offs_token_id_base, %store_row_16_0 : index
          %out_token_16_1 = arith.addi %offs_token_id_base, %store_row_16_1 : index
          %out_token_16_2 = arith.addi %offs_token_id_base, %store_row_16_2 : index
          %out_token_16_3 = arith.addi %offs_token_id_base, %store_row_16_3 : index
          %out_token_32_0 = arith.addi %offs_token_id_base, %store_row_32_0 : index
          %out_token_32_1 = arith.addi %offs_token_id_base, %store_row_32_1 : index
          %out_token_32_2 = arith.addi %offs_token_id_base, %store_row_32_2 : index
          %out_token_32_3 = arith.addi %offs_token_id_base, %store_row_32_3 : index
          %out_token_48_0 = arith.addi %offs_token_id_base, %store_row_48_0 : index
          %out_token_48_1 = arith.addi %offs_token_id_base, %store_row_48_1 : index
          %out_token_48_2 = arith.addi %offs_token_id_base, %store_row_48_2 : index
          %out_token_48_3 = arith.addi %offs_token_id_base, %store_row_48_3 : index

          %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<4663xi32>
          %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
          %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
          %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<4663xi32>
          %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
          %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
          %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<4663xi32>
          %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
          %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
          %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<4663xi32>
          %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
          %out_base_0_3 = arith.muli %tok_id_0_3, %N : index

          %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<4663xi32>
          %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
          %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
          %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<4663xi32>
          %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
          %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
          %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<4663xi32>
          %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
          %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
          %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<4663xi32>
          %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
          %out_base_16_3 = arith.muli %tok_id_16_3, %N : index

          %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<4663xi32>
          %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
          %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
          %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<4663xi32>
          %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
          %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
          %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<4663xi32>
          %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
          %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
          %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<4663xi32>
          %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
          %out_base_32_3 = arith.muli %tok_id_32_3, %N : index

          %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<4663xi32>
          %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
          %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
          %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<4663xi32>
          %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
          %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
          %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<4663xi32>
          %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
          %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
          %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<4663xi32>
          %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
          %out_base_48_3 = arith.muli %tok_id_48_3, %N : index

          // pid_n determines which 64-neuron block we're computing
          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

          // Column offsets for the 4 column tiles
          %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
          %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
          %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
          %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

          // Write all 16 tiles using vector.store
          // Tile (0,0)
          %idx_00_0 = arith.addi %out_base_0_0, %out_col_0 : index
          vector.store %r00_0_f16, %c_flat[%idx_00_0] : memref<4194304xf16>, vector<1xf16>
          %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
          vector.store %r00_1_f16, %c_flat[%idx_00_1] : memref<4194304xf16>, vector<1xf16>
          %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
          vector.store %r00_2_f16, %c_flat[%idx_00_2] : memref<4194304xf16>, vector<1xf16>
          %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
          vector.store %r00_3_f16, %c_flat[%idx_00_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (0,1)
          %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
          vector.store %r01_0_f16, %c_flat[%idx_01_0] : memref<4194304xf16>, vector<1xf16>
          %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
          vector.store %r01_1_f16, %c_flat[%idx_01_1] : memref<4194304xf16>, vector<1xf16>
          %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
          vector.store %r01_2_f16, %c_flat[%idx_01_2] : memref<4194304xf16>, vector<1xf16>
          %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
          vector.store %r01_3_f16, %c_flat[%idx_01_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (0,2)
          %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
          vector.store %r02_0_f16, %c_flat[%idx_02_0] : memref<4194304xf16>, vector<1xf16>
          %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
          vector.store %r02_1_f16, %c_flat[%idx_02_1] : memref<4194304xf16>, vector<1xf16>
          %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
          vector.store %r02_2_f16, %c_flat[%idx_02_2] : memref<4194304xf16>, vector<1xf16>
          %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
          vector.store %r02_3_f16, %c_flat[%idx_02_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (0,3)
          %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
          vector.store %r03_0_f16, %c_flat[%idx_03_0] : memref<4194304xf16>, vector<1xf16>
          %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
          vector.store %r03_1_f16, %c_flat[%idx_03_1] : memref<4194304xf16>, vector<1xf16>
          %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
          vector.store %r03_2_f16, %c_flat[%idx_03_2] : memref<4194304xf16>, vector<1xf16>
          %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
          vector.store %r03_3_f16, %c_flat[%idx_03_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (1,0)
          %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
          vector.store %r10_0_f16, %c_flat[%idx_10_0] : memref<4194304xf16>, vector<1xf16>
          %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
          vector.store %r10_1_f16, %c_flat[%idx_10_1] : memref<4194304xf16>, vector<1xf16>
          %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
          vector.store %r10_2_f16, %c_flat[%idx_10_2] : memref<4194304xf16>, vector<1xf16>
          %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
          vector.store %r10_3_f16, %c_flat[%idx_10_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (1,1)
          %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
          vector.store %r11_0_f16, %c_flat[%idx_11_0] : memref<4194304xf16>, vector<1xf16>
          %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
          vector.store %r11_1_f16, %c_flat[%idx_11_1] : memref<4194304xf16>, vector<1xf16>
          %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
          vector.store %r11_2_f16, %c_flat[%idx_11_2] : memref<4194304xf16>, vector<1xf16>
          %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
          vector.store %r11_3_f16, %c_flat[%idx_11_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (1,2)
          %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
          vector.store %r12_0_f16, %c_flat[%idx_12_0] : memref<4194304xf16>, vector<1xf16>
          %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
          vector.store %r12_1_f16, %c_flat[%idx_12_1] : memref<4194304xf16>, vector<1xf16>
          %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
          vector.store %r12_2_f16, %c_flat[%idx_12_2] : memref<4194304xf16>, vector<1xf16>
          %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
          vector.store %r12_3_f16, %c_flat[%idx_12_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (1,3)
          %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
          vector.store %r13_0_f16, %c_flat[%idx_13_0] : memref<4194304xf16>, vector<1xf16>
          %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
          vector.store %r13_1_f16, %c_flat[%idx_13_1] : memref<4194304xf16>, vector<1xf16>
          %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
          vector.store %r13_2_f16, %c_flat[%idx_13_2] : memref<4194304xf16>, vector<1xf16>
          %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
          vector.store %r13_3_f16, %c_flat[%idx_13_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (2,0)
          %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
          vector.store %r20_0_f16, %c_flat[%idx_20_0] : memref<4194304xf16>, vector<1xf16>
          %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
          vector.store %r20_1_f16, %c_flat[%idx_20_1] : memref<4194304xf16>, vector<1xf16>
          %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
          vector.store %r20_2_f16, %c_flat[%idx_20_2] : memref<4194304xf16>, vector<1xf16>
          %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
          vector.store %r20_3_f16, %c_flat[%idx_20_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (2,1)
          %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
          vector.store %r21_0_f16, %c_flat[%idx_21_0] : memref<4194304xf16>, vector<1xf16>
          %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
          vector.store %r21_1_f16, %c_flat[%idx_21_1] : memref<4194304xf16>, vector<1xf16>
          %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
          vector.store %r21_2_f16, %c_flat[%idx_21_2] : memref<4194304xf16>, vector<1xf16>
          %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
          vector.store %r21_3_f16, %c_flat[%idx_21_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (2,2)
          %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
          vector.store %r22_0_f16, %c_flat[%idx_22_0] : memref<4194304xf16>, vector<1xf16>
          %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
          vector.store %r22_1_f16, %c_flat[%idx_22_1] : memref<4194304xf16>, vector<1xf16>
          %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
          vector.store %r22_2_f16, %c_flat[%idx_22_2] : memref<4194304xf16>, vector<1xf16>
          %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
          vector.store %r22_3_f16, %c_flat[%idx_22_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (2,3)
          %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
          vector.store %r23_0_f16, %c_flat[%idx_23_0] : memref<4194304xf16>, vector<1xf16>
          %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
          vector.store %r23_1_f16, %c_flat[%idx_23_1] : memref<4194304xf16>, vector<1xf16>
          %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
          vector.store %r23_2_f16, %c_flat[%idx_23_2] : memref<4194304xf16>, vector<1xf16>
          %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
          vector.store %r23_3_f16, %c_flat[%idx_23_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (3,0)
          %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
          vector.store %r30_0_f16, %c_flat[%idx_30_0] : memref<4194304xf16>, vector<1xf16>
          %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
          vector.store %r30_1_f16, %c_flat[%idx_30_1] : memref<4194304xf16>, vector<1xf16>
          %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
          vector.store %r30_2_f16, %c_flat[%idx_30_2] : memref<4194304xf16>, vector<1xf16>
          %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
          vector.store %r30_3_f16, %c_flat[%idx_30_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (3,1)
          %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
          vector.store %r31_0_f16, %c_flat[%idx_31_0] : memref<4194304xf16>, vector<1xf16>
          %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
          vector.store %r31_1_f16, %c_flat[%idx_31_1] : memref<4194304xf16>, vector<1xf16>
          %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
          vector.store %r31_2_f16, %c_flat[%idx_31_2] : memref<4194304xf16>, vector<1xf16>
          %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
          vector.store %r31_3_f16, %c_flat[%idx_31_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (3,2)
          %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
          vector.store %r32_0_f16, %c_flat[%idx_32_0] : memref<4194304xf16>, vector<1xf16>
          %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
          vector.store %r32_1_f16, %c_flat[%idx_32_1] : memref<4194304xf16>, vector<1xf16>
          %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
          vector.store %r32_2_f16, %c_flat[%idx_32_2] : memref<4194304xf16>, vector<1xf16>
          %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
          vector.store %r32_3_f16, %c_flat[%idx_32_3] : memref<4194304xf16>, vector<1xf16>

          // Tile (3,3)
          %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
          vector.store %r33_0_f16, %c_flat[%idx_33_0] : memref<4194304xf16>, vector<1xf16>
          %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
          vector.store %r33_1_f16, %c_flat[%idx_33_1] : memref<4194304xf16>, vector<1xf16>
          %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
          vector.store %r33_2_f16, %c_flat[%idx_33_2] : memref<4194304xf16>, vector<1xf16>
          %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
          vector.store %r33_3_f16, %c_flat[%idx_33_3] : memref<4194304xf16>, vector<1xf16>
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<2048x128xf16>,
       // %b_ptr: memref<8x1024x128xf16>,
       // %sorted_token_ids_ptr: memref<4663xi32>,
       // %expert_ids_ptr: memref<73xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<2048x2x1024xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<2048x128xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x1024x128xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<4663xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<73xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<2048x2x1024xf16>
    %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<2048x128xf16>, tensor<8x1024x128xf16>, tensor<4663xi32>, tensor<73xi32>, tensor<1xi32>, tensor<2048x2x1024xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<2048x2x1024xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<2048x2x1024xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)
    asm = asm_dtype0_256_128_8_64_2_33
 #  asm = asm_dtype0_256_128_8_64_2_33_mfma
    asm = asm_dtype0_1024_128_8_64_2_2048_mfma
    gemm_kernel, symbols = get_moe_gemm_kernel(
        num_tokens,
        topk,
        block_size,
        num_experts,
        k,
        n,
        max_num_tokens_padded,
        max_num_m_blocks,
        MMAType.F32_16x16x16_F16,
        tkl.f16,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        print_mlir=True,
        override_mlir=asm,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm_kernel)
  # print(gemm.asm)
    print("MAX_NUM_TOKENS_PADDED ", max_num_tokens_padded, topk_ids.numel(), num_experts, block_size)
    print("SORTED_IDS ", sorted_ids)
    print("EXPERT_IDS ", expert_ids)
    print("NUM_TOKENS_POST_PAD ", num_tokens_post_pad)
    gemm(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  print("A[1] ", a[1])
 #  print("A[4] ", a[4])
 #  print("A[7] ", a[7])
   #print("A[2] ", a[2])
   #print("A[5] ", a[5])
   #print(f"B[0][0] {b[0][0]}")
 #  print("A[5] ", a[5])
 #  print("A[6] ", a[6])
 #  print("A[8] ", a[8])
 #  for i in range(64):
 #      print(f"B[0][{i}] {b[0][i]}")
 #      print(f"B[1][{i}] {b[1][i]}")
 #  for i in range(10):
 #      print(f"REF_C[{i}][0] {c[i][0]}")
 #      print(f"C[{i}][0] {mlir_c[i][0]}")
 #      print(f"REF_C[{i}][1] {c[i][1]}")
 #      print(f"C[{i}][1] {mlir_c[i][1]}")
   #print(f"REF_C[2][0] {c[2][0]}")
   #print(f"C[2][0] {mlir_c[2][0]}")
   #print(f"REF_C[5][1] {c[5][1]}")
   #print(f"REF_C[6][1] {c[6][1]}")
   #print(f"REF_C[6][1] {c[6][1]}")
    torch.testing.assert_close(
        c,
        mlir_c,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        ref_c,
        mlir_c,
        rtol=rtol,
        atol=atol,
    )
