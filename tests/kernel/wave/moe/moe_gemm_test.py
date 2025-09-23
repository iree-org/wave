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
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile

torch.manual_seed(0)


num_tokens_values = [1, 33, 256]
topk_values = [2]
block_size_values = [16, 32, 64]
num_experts_values = [8, 64]
k_values = [128, 511]
n_values = [128, 256, 512, 1024]
dtype_values = [torch.float16, torch.bfloat16]


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
    scores = torch.rand(num_tokens, num_experts)

    # Get topk expert indices for each token
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
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

    a = torch.rand((num_tokens, k), dtype=dtype)
    b = torch.rand((num_experts, n, k), dtype=dtype)
    c = torch.zeros(num_tokens, topk, n, dtype=dtype)

    moe_gemm_pytorch(
        a, b, c, sorted_ids, expert_ids, num_tokens_post_pad, topk, block_size
    )

    ref_c = torch.zeros(num_tokens, topk, n, dtype=dtype)
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

    mlir_c = torch.zeros(num_tokens, topk, n, dtype=dtype)
    asm = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel {
    stream.executable.export public @fused_moe_kernel workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel(
          // Input memrefs
       // %a_ptr: memref<33x128xf16>,
       // %b_ptr: memref<8x256x128xf16>,
       // %c_ptr: memref<33x2x256xf16>,
       // %sorted_token_ids_ptr: memref<633xi32>,
       // %expert_ids_ptr: memref<10xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 256
        // K = 128
        // EM = 8
        // top_k = 2
        // num_valid_tokens = 66
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 256 : index
        %K = arith.constant 128 : index
        %EM = arith.constant 8 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 66 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %passthru_a = vector.broadcast %f0_f16 : f16 to vector<64x32xf16>
        %true_mask = arith.constant dense<true> : vector<64xi1>
        %zeroes_64 = arith.constant dense<0> : vector<64xi32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<33x128xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x256x128xf16>
        %c_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33x2x256xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<633xi32>
        %expert_ids_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<10xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<1xi32>

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
          %b_view =  memref.subview %b_ptr[%off_experts, 0, 0] [1, 256, 128] [1, 1, 1] :
        	memref<8x256x128xf16> to memref<1x256x128xf16, strided<[32768, 128, 1], offset: ?>>
          %b_block = memref.collapse_shape %b_view [[0, 1], [2]] : memref<1x256x128xf16, strided<[32768, 128, 1], offset: ?>> into memref<256x128xf16, strided<[128, 1], offset: ?>>
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<33x2x256xf16> into memref<16896xf16>

          %result = scf.for %k_block = %c0 to %num_blocks step %c1 iter_args(%acc = %accumulator) -> vector<64x64xf32> {
            // Compute current K offset
            %k_offset = arith.muli %k_block, %BLOCK_SIZE_K : index

            %a = vector.gather %a_flat[%k_offset][%indices_a], %a_mask_base, %passthru_a :
                memref<4224xf16>, vector<64x32xindex>, vector<64x32xi1>, vector<64x32xf16> into vector<64x32xf16>

            %b = vector.transfer_read %b_block[%off_bn, %k_offset], %f0_f16
      	  {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<32x64xf16>

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

     //   // Compute output indices
     //   %offs_cn_2d = vector.shape_cast %offs_cn : vector<64xindex> to vector<1x64xindex>

     //   %N_2d = vector.broadcast %N : index to vector<64x1xindex>
     //   %c_row_offsets = arith.muli %offs_token_2d, %N_2d : vector<64x1xindex>

     //   %first_broadcast_c = vector.broadcast %c_row_offsets : vector<64x1xindex> to vector<64x64xindex>
     //   %second_broadcast_c = vector.broadcast %offs_cn_2d : vector<1x64xindex> to vector<64x64xindex>
     //   %c_indices = arith.addi %first_broadcast_c, %second_broadcast_c : vector<64x64xindex>

     //   // Create output mask
     //   %offs_cn_mask = arith.cmpi slt, %offs_cn, %N_splat : vector<64xindex>

     //   %c_mask_base = vector.broadcast %8 : vector<64x1xi1> to vector<64x64xi1>
     //   %cn_mask_broadcast = vector.broadcast %offs_cn_mask : vector<64xi1> to vector<64x64xi1>
     //   %output_mask = arith.andi %c_mask_base, %cn_mask_broadcast : vector<64x64xi1>

     //   // Store results
     //   vector.scatter %c_flat[%c0][%c_indices], %output_mask, %result_f16 :
     //       memref<16896xf16>, vector<64x64xindex>, vector<64x64xi1>, vector<64x64xf16>

          // TODO: use this when vector.extract for 2d vector can be lowered to LLVM.
          // Compute output indices

          %top_k_splat = vector.broadcast %top_k : index to vector<64xindex>
          %orig_token_ids = arith.divsi %offs_token, %top_k_splat : vector<64xindex>
          %expert_slots = arith.remsi %offs_token, %top_k_splat : vector<64xindex>

          // Create output mask
          %offs_cn_mask = arith.cmpi slt, %offs_cn, %N_splat : vector<64xindex>
          %output_mask = arith.andi %offs_cn_mask, %token_mask : vector<64xi1>

          // Store results
          scf.for %row = %c0 to %BLOCK_SIZE_M step %c1 {
	        %row_token_mask = vector.extract %token_mask[%row] : i1 from vector<64xi1>
            scf.if %row_token_mask {
              %vec = vector.extract %result_f16[%row] : vector<64xf16> from vector<64x64xf16>
              %orig_token_id = vector.extract %orig_token_ids[%row] : index from vector<64xindex>
              %expert_slot = vector.extract %expert_slots[%row] : index from vector<64xindex>
              vector.transfer_write %vec, %c_ptr[%orig_token_id, %expert_slot, %offs_bn_base], %output_mask : vector<64xf16>, memref<33x2x256xf16>
              scf.yield
            }
          }
        }

        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<33x128xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x256x128xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33x2x256xf16>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<633xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<10xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<1xi32>
    %6 = flow.dispatch @fused_moe_kernel::@fused_moe_kernel(%0, %1, %2, %3, %4, %5) : (tensor<33x128xf16>, tensor<8x256x128xf16>, tensor<33x2x256xf16>, tensor<633xi32>, tensor<10xi32>, tensor<1xi32>) -> %2
    %7 = hal.tensor.barrier join(%6 : tensor<33x2x256xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<33x2x256xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)
    from wave_lang.kernel.wave.templates.moe import (
        get_gemm_kernel,
        get_silu_and_mul_kernel,
    )
    from wave_lang.kernel.wave.constraints import MMAType
    import wave_lang.kernel.lang as tkl
    from wave_lang.kernel.wave.utils.general_utils import (
        get_default_scheduling_params,
    )
#    asm = (
#    """
##map = affine_map<()[s0, s1] -> ((s1 * 32 + s0 floordiv 4) mod 64)>
##map1 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>
##map2 = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 4 - ((s1 * 32 + s0 floordiv 4) floordiv 64) * 64)>
##map3 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16)>
##map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
##map5 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
##map6 = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + 16)>
##map7 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32)>
##map8 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 32 + 16)>
##map9 = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 8 - (s1 floordiv 4) * 32)>
##map10 = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 16) * 4)>
##map11 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16)>
##map12 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 64 + s2 * 32 - (s0 floordiv 16) * 16 + 16)>
##translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
#module attributes {transform.with_named_sequence} {
#  stream.executable private @gemm {
#    stream.executable.export public @gemm workgroups() -> (index, index, index) {
#      %c1 = arith.constant 1 : index
#      %c2 = arith.constant 2 : index
#      stream.return %c1, %c2, %c1 : index, index, index
#    }
#    builtin.module {
#      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
#        %cst = arith.constant dense<0.000000e+00> : vector<8xf16>
#        %cst_0 = arith.constant dense<511> : vector<8xindex>
#        %cst_1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
#        %c1 = arith.constant 1 : index
#        %c16 = arith.constant 16 : index
#        %c4608 = arith.constant 4608 : index
#        %c0 = arith.constant 0 : index
#        %cst_2 = arith.constant dense<0.000000e+00> : vector<4xf32>
#        %block_id_y = gpu.block_id  y upper_bound 2
#        %thread_id_x = gpu.thread_id  x upper_bound 128
#        %thread_id_y = gpu.thread_id  y upper_bound 2
#        %alloc = memref.alloc() : memref<9216xi8, #gpu.address_space<workgroup>>
#        %view = memref.view %alloc[%c0][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xf16, #gpu.address_space<workgroup>>
#        %view_3 = memref.view %alloc[%c4608][] : memref<9216xi8, #gpu.address_space<workgroup>> to memref<64x36xf16, #gpu.address_space<workgroup>>
#        %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<128x511xf16, strided<[511, 1], offset: ?>>
#        %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<1x511xf16, strided<[511, 1], offset: ?>>
#        %2 = affine.apply #map()[%thread_id_x, %thread_id_y]
#        %3 = arith.cmpi slt, %2, %c1 : index
#        %4 = vector.broadcast %3 : i1 to vector<8xi1>
#        %5 = affine.apply #map1()[%thread_id_x]
#        %6 = affine.apply #map2()[%thread_id_x, %thread_id_y, %block_id_y]
#        %7 = affine.apply #map3()[%thread_id_x, %thread_id_y]
#        %8 = affine.apply #map4()[%thread_id_x]
#        %9 = affine.apply #map5()[%thread_id_x]
#        %10 = affine.apply #map6()[%thread_id_x, %thread_id_y]
#        %11 = affine.apply #map7()[%thread_id_x]
#        %12 = affine.apply #map8()[%thread_id_x]
#        %13:4 = scf.for %arg3 = %c0 to %c16 step %c1 iter_args(%arg4 = %cst_2, %arg5 = %cst_2, %arg6 = %cst_2, %arg7 = %cst_2) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
#          %22 = affine.apply #map9()[%arg3, %thread_id_x]
#          %23 = vector.broadcast %22 : index to vector<8xindex>
#          %24 = arith.addi %23, %cst_1 overflow<nsw, nuw> : vector<8xindex>
#          %25 = arith.cmpi slt, %24, %cst_0 : vector<8xindex>
#          %26 = arith.andi %25, %4 : vector<8xi1>
#          %27 = vector.maskedload %1[%2, %22], %26, %cst : memref<1x511xf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
#          amdgpu.lds_barrier
#          vector.store %27, %view_3[%2, %5] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
#          %28 = vector.maskedload %0[%6, %22], %25, %cst : memref<128x511xf16, strided<[511, 1], offset: ?>>, vector<8xi1>, vector<8xf16> into vector<8xf16>
#          vector.store %28, %view[%2, %5] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
#          amdgpu.lds_barrier
#          %29 = vector.load %view[%7, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %30 = vector.load %view[%7, %9] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %31 = vector.load %view[%10, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %32 = vector.load %view[%10, %9] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %33 = vector.load %view_3[%11, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %34 = vector.load %view_3[%11, %9] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %35 = vector.load %view_3[%12, %8] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %36 = vector.load %view_3[%12, %9] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
#          %37 = amdgpu.mfma %33 * %29 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %38 = amdgpu.mfma %34 * %30 + %37 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %39 = amdgpu.mfma %33 * %31 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %40 = amdgpu.mfma %34 * %32 + %39 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %41 = amdgpu.mfma %35 * %29 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %42 = amdgpu.mfma %36 * %30 + %41 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %43 = amdgpu.mfma %35 * %31 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          %44 = amdgpu.mfma %36 * %32 + %43 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
#          scf.yield %38, %40, %42, %44 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
#        }
#        %14 = vector.extract_strided_slice %13#0 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
#        %15 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1x128xf32, strided<[128, 1], offset: ?>>
#        %16 = affine.apply #map10()[%thread_id_x]
#        %17 = affine.apply #map11()[%thread_id_x, %block_id_y, %thread_id_y]
#        %18 = arith.cmpi slt, %16, %c1 : index
#        %19 = vector.broadcast %18 : i1 to vector<1xi1>
#        vector.maskedstore %15[%16, %17], %19, %14 : memref<1x128xf32, strided<[128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
#        %20 = vector.extract_strided_slice %13#1 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
#        %21 = affine.apply #map12()[%thread_id_x, %block_id_y, %thread_id_y]
#        vector.maskedstore %15[%16, %21], %19, %20 : memref<1x128xf32, strided<[128, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
#        return
#      }
#    }
#  }
#  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
#    %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<1x511xf16>
#    %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<128x511xf16>
#    %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<1x128xf32>
#    %3 = flow.dispatch @gemm::@gemm(%0, %1, %2) : (tensor<1x511xf16>, tensor<128x511xf16>, tensor<1x128xf32>) -> %2
#    %4 = hal.tensor.barrier join(%3 : tensor<1x128xf32>) => %arg4 : !hal.fence
#    %5 = hal.tensor.export %4 : tensor<1x128xf32> -> !hal.buffer_view
#    return %5 : !hal.buffer_view
#  }
#}
#    """
#)
    gemm_kernel, symbols = get_gemm_kernel(
        64,
        64,
        64,
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
    #   print_mlir=True,
        override_mlir=asm,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm_kernel)
  # generate_iree_ref("", [a, b, mlir_c, sorted_ids, expert_ids, num_tokens_post_pad], [], options, asm=asm)
  # generate_iree_ref("", [a, b, mlir_c], [], options, asm=asm)
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
