import triton
import triton.language as tl
import torch
import itertools
import math

import triton.compiler as tc
from .torch_kernels import moe_align_block_size_pytorch
from pathlib import Path
import datetime as dt

import torch.nn.functional as F
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
    enable_scheduling_barriers,
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


@triton.jit
def _moe_gemm_kernel_triton(
    a_ptr,              # (M, K)
    b_ptr,              # (E, N, K)
    c2d_ptr,            # (M*topk, N)  -- 2D view of C to match offs_token indexing
    sorted_token_ids_ptr,   # (EM,)
    expert_ids_ptr,         # (num_blocks_m,)
    num_tokens_post_padded_ptr,  # (1,)

    # dimensions
    N, K, EM, num_valid_tokens,

    # strides
    stride_am, stride_ak,     # A
    stride_be, stride_bk, stride_bn,  # B  (E,N,K) with layout E-major
    stride_c2dm, stride_c2dn, # C2D  (M*topk, N)

    # meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,     # tl.float32 (accumulator)
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # early-out if padded rows
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # per-block token slice
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # expert for this M-block
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    # if -1, nothing to compute (not in this expert-parallel shard)
    if off_expert == -1:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
        c_mask = token_mask[:, None] & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers (flattened token pairs // top_k → original token)
    a_ptrs = a_ptr + ( (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak )

    # B pointers: expert-major, NxK per expert
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + (offs_n[None, :] % N) * stride_bn)
    )

    # fp32 accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)

    # K loop
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for kt in range(0, num_k_tiles):
        # for last tile, mask loads if K not a multiple of BLOCK_SIZE_K
        k_left = K - kt * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_left), other=0)
        b = tl.load(b_ptrs,  mask=(offs_k[:, None] < k_left), other=0)
        acc += tl.dot(a, b)  # fp16/bf16→fp32

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # write-back into (M*topk, N) view using offs_token as row index
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# --------- Python launcher that uses your test’s tensors ---------
def moe_gemm_triton(a, b, c, sorted_ids, expert_ids, num_tokens_post_padded, topk,
                    block_m=64, block_n=64, block_k=32, group_m=8):
    """
    a: (M, K)          dtype: fp16/bf16
    b: (E, N, K)       dtype: fp16/bf16 (same as a)
    c: (M, topk, N)    dtype: same as a (or fp16/bf16); accumulator is fp32
    sorted_ids: (EM,)  int32/int64
    expert_ids: (ceil(EM/block_m),) int32/int64  (one expert id per M-block)
    num_tokens_post_padded: (1,) int32/int64
    """
    assert a.device.type == "cuda" and b.device.type == "cuda" and c.device.type == "cuda"
    M, K = a.shape
    E, N, Kb = b.shape
    assert K == Kb, "B last dim must equal K"
    assert c.shape == (M, topk, N)
    EM = sorted_ids.numel()
    compute_type = tl.float32

    # convenience: make 2D view of c as (M*topk, N) to match offs_token addressing
    c2d = c.view(M * topk, N)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    _moe_gemm_kernel_triton[grid](
        a, b, c2d,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, M * topk,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        c2d.stride(0), c2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        top_k=topk,
        compute_type=compute_type,
        num_warps=4,      # good defaults for 64x64x32 tiles on RDNA/CDNA
        num_stages=2,
    )



@torch.no_grad()
def build_sorted_ids_and_expert_blocks(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """
    Given topk_ids: (M, topk) of expert indices, build:
      - sorted_ids: (EM_padded,) int32 — token*topk+slot, grouped by expert, padded per-expert to multiples of block_m
      - expert_ids: (num_m_blocks,) int32 — expert id for each M-block (64 rows), -1 for host-padding blocks
      - num_tokens_post_padded: (1,) int32 — total length of sorted_ids after padding
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    torch.manual_seed(0)

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_m - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_m)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())

    # Populate using the same routine as the test harness
    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_m, sorted_ids, expert_ids, num_tokens_post_pad
    )

    return sorted_ids, expert_ids, num_tokens_post_pad, max_num_tokens_padded, max_num_m_blocks


asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16 {
    stream.executable.export public @fused_moe_kernel_16x16x16 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64×32 for A, 64×32 for B (instead of 64×6144)
          %c4096 = arith.constant 4096 : index
          %alloc = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>

//%tid_cond = arith.cmpi eq, %thread_id, %c0 : index
//%pid_cond = arith.cmpi eq, %pid, %c0 : index
//%print = arith.andi %tid_cond, %pid_cond : i1
//
//scf.if %print { gpu.printf "pid %d\\n", %pid : index }
//scf.if %print { gpu.printf "pid_m %d\\n", %pid_m : index }
//scf.if %print { gpu.printf "pid_n %d\\n",  %pid_n : index }
//
//%a0.0 = memref.load %shared_a[%c0, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][0] %f\\n", %a0.0 : f16 }
//
//%a0.1 = memref.load %shared_a[%c0, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][1] %f\\n", %a0.1 : f16 }
//
//%a0.127 = memref.load %shared_a[%c0, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[0][127] %f\\n", %a0.127 : f16 }
//
//%a1.0 = memref.load %shared_a[%c1, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][0] %f\\n", %a1.0 : f16 }
//
//%a1.1 = memref.load %shared_a[%c1, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][1] %f\\n", %a1.1 : f16 }
//
//%a1.127 = memref.load %shared_a[%c1, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[1][127] %f\\n", %a1.127 : f16 }
//
//%a63.0 = memref.load %shared_a[%c63, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][0] %f\\n", %a63.0 : f16 }
//
//%a63.1 = memref.load %shared_a[%c63, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][1] %f\\n", %a63.1 : f16 }
//
//%a63.127 = memref.load %shared_a[%c63, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "a[63][127] %f\\n", %a63.127 : f16 }
//
//%b0.0 = memref.load %shared_b[%c0, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][0] %f\\n", %b0.0 : f16 }
//
//%b0.1 = memref.load %shared_b[%c0, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][1] %f\\n", %b0.1 : f16 }
//
//%b0.127 = memref.load %shared_b[%c0, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[0][127] %f\\n", %b0.127 : f16 }
//
//%b1.0 = memref.load %shared_b[%c1, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][0] %f\\n", %b1.0 : f16 }
//
//%b1.1 = memref.load %shared_b[%c1, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][1] %f\\n", %b1.1 : f16 }
//
//%b1.127 = memref.load %shared_b[%c1, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[1][127] %f\\n", %b1.127 : f16 }
//
//%b63.0 = memref.load %shared_b[%c63, %c0] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][0] %f\\n", %b63.0 : f16 }
//
//%b63.1 = memref.load %shared_b[%c63, %c1] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][1] %f\\n", %b63.1 : f16 }
//
//%b63.127 = memref.load %shared_b[%c63, %c127] : memref<64x6144xf16, #gpu.address_space<workgroup>>
//scf.if %print { gpu.printf "b[63][127] %f\\n", %b63.127 : f16 }
//
//amdgpu.lds_barrier

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

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

          %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<33335xi32>
          %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
          %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
          %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
          %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
          %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<33335xi32>
          %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
          %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
          %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
          %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
          %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<33335xi32>
          %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
          %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
          %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
          %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
          %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<33335xi32>
          %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
          %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
          %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
          %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

          %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<33335xi32>
          %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
          %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
          %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
          %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
          %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<33335xi32>
          %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
          %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
          %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
          %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
          %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<33335xi32>
          %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
          %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
          %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
          %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
          %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<33335xi32>
          %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
          %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
          %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
          %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

          %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<33335xi32>
          %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
          %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
          %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
          %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
          %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<33335xi32>
          %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
          %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
          %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
          %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
          %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<33335xi32>
          %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
          %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
          %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
          %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
          %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<33335xi32>
          %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
          %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
          %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
          %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

          %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<33335xi32>
          %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
          %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
          %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
          %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
          %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<33335xi32>
          %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
          %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
          %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
          %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
          %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<33335xi32>
          %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
          %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
          %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
          %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
          %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<33335xi32>
          %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
          %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
          %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
          %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

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
          vector.maskedstore %c_flat[%idx_00_0], %out_mask_0_0, %r00_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_1], %out_mask_0_1, %r00_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_2], %out_mask_0_2, %r00_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_3], %out_mask_0_3, %r00_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,1)
          %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_0], %out_mask_0_0, %r01_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_1], %out_mask_0_1, %r01_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_2], %out_mask_0_2, %r01_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_3], %out_mask_0_3, %r01_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,2)
          %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_0], %out_mask_0_0, %r02_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_1], %out_mask_0_1, %r02_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_2], %out_mask_0_2, %r02_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_3], %out_mask_0_3, %r02_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,3)
          %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_0], %out_mask_0_0, %r03_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_1], %out_mask_0_1, %r03_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_2], %out_mask_0_2, %r03_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_3], %out_mask_0_3, %r03_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,0)
          %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_0], %out_mask_16_0, %r10_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_1], %out_mask_16_1, %r10_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_2], %out_mask_16_2, %r10_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_3], %out_mask_16_3, %r10_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,1)
          %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_0], %out_mask_16_0, %r11_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_1], %out_mask_16_1, %r11_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_2], %out_mask_16_2, %r11_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_3], %out_mask_16_3, %r11_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,2)
          %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_0], %out_mask_16_0, %r12_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_1], %out_mask_16_1, %r12_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_2], %out_mask_16_2, %r12_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_3], %out_mask_16_3, %r12_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,3)
          %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_0], %out_mask_16_0, %r13_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_1], %out_mask_16_1, %r13_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_2], %out_mask_16_2, %r13_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_3], %out_mask_16_3, %r13_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,0)
          %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_0], %out_mask_32_0, %r20_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_1], %out_mask_32_1, %r20_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_2], %out_mask_32_2, %r20_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_3], %out_mask_32_3, %r20_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,1)
          %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_0], %out_mask_32_0, %r21_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_1], %out_mask_32_1, %r21_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_2], %out_mask_32_2, %r21_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_3], %out_mask_32_3, %r21_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,2)
          %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_0], %out_mask_32_0, %r22_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_1], %out_mask_32_1, %r22_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_2], %out_mask_32_2, %r22_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_3], %out_mask_32_3, %r22_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,3)
          %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_0], %out_mask_32_0, %r23_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_1], %out_mask_32_1, %r23_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_2], %out_mask_32_2, %r23_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_3], %out_mask_32_3, %r23_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,0)
          %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_0], %out_mask_48_0, %r30_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_1], %out_mask_48_1, %r30_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_2], %out_mask_48_2, %r30_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_3], %out_mask_48_3, %r30_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,1)
          %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_0], %out_mask_48_0, %r31_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_1], %out_mask_48_1, %r31_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_2], %out_mask_48_2, %r31_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_3], %out_mask_48_3, %r31_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,2)
          %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_0], %out_mask_48_0, %r32_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_1], %out_mask_48_1, %r32_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_2], %out_mask_48_2, %r32_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_3], %out_mask_48_3, %r32_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,3)
          %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_0], %out_mask_48_0, %r33_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_1], %out_mask_48_1, %r33_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_2], %out_mask_48_2, %r33_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_3], %out_mask_48_3, %r33_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16::@fused_moe_kernel_16x16x16(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_masked_store {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_masked_store workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_masked_store(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64x48 for A, 64x48 for B (instead of 64×6144)
          %c6144 = arith.constant 6144 : index
          %alloc = memref.alloc() : memref<12288xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<12288xi8, #gpu.address_space<workgroup>>
            to memref<64x48xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c6144][] : memref<12288xi8, #gpu.address_space<workgroup>>
            to memref<64x48xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

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

          %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<33335xi32>
          %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
          %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
          %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
          %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
          %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<33335xi32>
          %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
          %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
          %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
          %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
          %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<33335xi32>
          %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
          %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
          %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
          %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
          %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<33335xi32>
          %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
          %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
          %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
          %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

          %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<33335xi32>
          %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
          %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
          %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
          %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
          %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<33335xi32>
          %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
          %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
          %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
          %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
          %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<33335xi32>
          %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
          %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
          %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
          %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
          %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<33335xi32>
          %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
          %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
          %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
          %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

          %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<33335xi32>
          %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
          %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
          %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
          %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
          %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<33335xi32>
          %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
          %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
          %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
          %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
          %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<33335xi32>
          %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
          %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
          %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
          %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
          %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<33335xi32>
          %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
          %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
          %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
          %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

          %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<33335xi32>
          %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
          %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
          %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
          %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
          %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<33335xi32>
          %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
          %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
          %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
          %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
          %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<33335xi32>
          %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
          %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
          %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
          %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
          %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<33335xi32>
          %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
          %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
          %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
          %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

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
          vector.maskedstore %c_flat[%idx_00_0], %out_mask_0_0, %r00_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_1 = arith.addi %out_base_0_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_1], %out_mask_0_1, %r00_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_2 = arith.addi %out_base_0_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_2], %out_mask_0_2, %r00_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_00_3 = arith.addi %out_base_0_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_00_3], %out_mask_0_3, %r00_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,1)
          %idx_01_0 = arith.addi %out_base_0_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_0], %out_mask_0_0, %r01_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_1 = arith.addi %out_base_0_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_1], %out_mask_0_1, %r01_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_2 = arith.addi %out_base_0_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_2], %out_mask_0_2, %r01_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_01_3 = arith.addi %out_base_0_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_01_3], %out_mask_0_3, %r01_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,2)
          %idx_02_0 = arith.addi %out_base_0_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_0], %out_mask_0_0, %r02_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_1 = arith.addi %out_base_0_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_1], %out_mask_0_1, %r02_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_2 = arith.addi %out_base_0_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_2], %out_mask_0_2, %r02_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_02_3 = arith.addi %out_base_0_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_02_3], %out_mask_0_3, %r02_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (0,3)
          %idx_03_0 = arith.addi %out_base_0_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_0], %out_mask_0_0, %r03_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_1 = arith.addi %out_base_0_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_1], %out_mask_0_1, %r03_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_2 = arith.addi %out_base_0_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_2], %out_mask_0_2, %r03_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_03_3 = arith.addi %out_base_0_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_03_3], %out_mask_0_3, %r03_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,0)
          %idx_10_0 = arith.addi %out_base_16_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_0], %out_mask_16_0, %r10_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_1 = arith.addi %out_base_16_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_1], %out_mask_16_1, %r10_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_2 = arith.addi %out_base_16_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_2], %out_mask_16_2, %r10_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_10_3 = arith.addi %out_base_16_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_10_3], %out_mask_16_3, %r10_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,1)
          %idx_11_0 = arith.addi %out_base_16_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_0], %out_mask_16_0, %r11_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_1 = arith.addi %out_base_16_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_1], %out_mask_16_1, %r11_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_2 = arith.addi %out_base_16_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_2], %out_mask_16_2, %r11_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_11_3 = arith.addi %out_base_16_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_11_3], %out_mask_16_3, %r11_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,2)
          %idx_12_0 = arith.addi %out_base_16_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_0], %out_mask_16_0, %r12_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_1 = arith.addi %out_base_16_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_1], %out_mask_16_1, %r12_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_2 = arith.addi %out_base_16_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_2], %out_mask_16_2, %r12_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_12_3 = arith.addi %out_base_16_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_12_3], %out_mask_16_3, %r12_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (1,3)
          %idx_13_0 = arith.addi %out_base_16_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_0], %out_mask_16_0, %r13_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_1 = arith.addi %out_base_16_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_1], %out_mask_16_1, %r13_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_2 = arith.addi %out_base_16_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_2], %out_mask_16_2, %r13_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_13_3 = arith.addi %out_base_16_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_13_3], %out_mask_16_3, %r13_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,0)
          %idx_20_0 = arith.addi %out_base_32_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_0], %out_mask_32_0, %r20_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_1 = arith.addi %out_base_32_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_1], %out_mask_32_1, %r20_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_2 = arith.addi %out_base_32_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_2], %out_mask_32_2, %r20_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_20_3 = arith.addi %out_base_32_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_20_3], %out_mask_32_3, %r20_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,1)
          %idx_21_0 = arith.addi %out_base_32_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_0], %out_mask_32_0, %r21_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_1 = arith.addi %out_base_32_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_1], %out_mask_32_1, %r21_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_2 = arith.addi %out_base_32_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_2], %out_mask_32_2, %r21_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_21_3 = arith.addi %out_base_32_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_21_3], %out_mask_32_3, %r21_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,2)
          %idx_22_0 = arith.addi %out_base_32_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_0], %out_mask_32_0, %r22_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_1 = arith.addi %out_base_32_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_1], %out_mask_32_1, %r22_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_2 = arith.addi %out_base_32_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_2], %out_mask_32_2, %r22_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_22_3 = arith.addi %out_base_32_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_22_3], %out_mask_32_3, %r22_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (2,3)
          %idx_23_0 = arith.addi %out_base_32_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_0], %out_mask_32_0, %r23_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_1 = arith.addi %out_base_32_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_1], %out_mask_32_1, %r23_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_2 = arith.addi %out_base_32_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_2], %out_mask_32_2, %r23_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_23_3 = arith.addi %out_base_32_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_23_3], %out_mask_32_3, %r23_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,0)
          %idx_30_0 = arith.addi %out_base_48_0, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_0], %out_mask_48_0, %r30_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_1 = arith.addi %out_base_48_1, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_1], %out_mask_48_1, %r30_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_2 = arith.addi %out_base_48_2, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_2], %out_mask_48_2, %r30_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_30_3 = arith.addi %out_base_48_3, %out_col_0 : index
          vector.maskedstore %c_flat[%idx_30_3], %out_mask_48_3, %r30_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,1)
          %idx_31_0 = arith.addi %out_base_48_0, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_0], %out_mask_48_0, %r31_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_1 = arith.addi %out_base_48_1, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_1], %out_mask_48_1, %r31_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_2 = arith.addi %out_base_48_2, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_2], %out_mask_48_2, %r31_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_31_3 = arith.addi %out_base_48_3, %out_col_1 : index
          vector.maskedstore %c_flat[%idx_31_3], %out_mask_48_3, %r31_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,2)
          %idx_32_0 = arith.addi %out_base_48_0, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_0], %out_mask_48_0, %r32_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_1 = arith.addi %out_base_48_1, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_1], %out_mask_48_1, %r32_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_2 = arith.addi %out_base_48_2, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_2], %out_mask_48_2, %r32_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_32_3 = arith.addi %out_base_48_3, %out_col_2 : index
          vector.maskedstore %c_flat[%idx_32_3], %out_mask_48_3, %r32_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>

          // Tile (3,3)
          %idx_33_0 = arith.addi %out_base_48_0, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_0], %out_mask_48_0, %r33_0_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_1 = arith.addi %out_base_48_1, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_1], %out_mask_48_1, %r33_1_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_2 = arith.addi %out_base_48_2, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_2], %out_mask_48_2, %r33_2_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
          %idx_33_3 = arith.addi %out_base_48_3, %out_col_3 : index
          vector.maskedstore %c_flat[%idx_33_3], %out_mask_48_3, %r33_3_f16 : memref<1073741824xf16>, vector<1xi1>, vector<1xf16>
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_masked_store::@fused_moe_kernel_16x16x16_padding_masked_store(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_96_db {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_96_db workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_96_db(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c16384 = arith.constant 16384 : index
        %c32768 = arith.constant 32768 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c1_i32 = arith.constant 1 : i32
        %c3_i32 = arith.constant 3 : i32
        %c12_i32 = arith.constant 12 : i32
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

// Allocate 2x LDS for A and B (double buffering)
%c6144 = arith.constant 6144 : index
%c12288 = arith.constant 12288 : index
%c18432 = arith.constant 18432 : index
%c24576 = arith.constant 24576 : index

%alloc = memref.alloc() : memref<37120xi8, #gpu.address_space<workgroup>>  // 24576 + 12544 (output)

// Create views for both buffers
%shared_a_0 = memref.view %alloc[%c0][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b_0 = memref.view %alloc[%c6144][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_a_1 = memref.view %alloc[%c12288][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b_1 = memref.view %alloc[%c18432][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_output = memref.view %alloc[%c24576][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x96xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
vector.store %a_row_vec_0_first, %shared_a_0[%thread_row_base, %thread_col_offset] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %a_row_vec_0_second, %shared_a_0[%thread_row_second, %thread_col_offset] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
vector.store %b_row_vec_0_first, %shared_b_0[%thread_row_base, %thread_col_offset] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %b_row_vec_0_second, %shared_b_0[%thread_row_second, %thread_col_offset] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

  // Compute buffer offsets using XOR (toggles between 0 and 12288)
  // k_block % 2 == 0 -> offset = 0 (buffer 0)
  // k_block % 2 == 1 -> offset = 12288 (buffer 1)
  %k_block_i32 = arith.index_cast %k_block : index to i32
  %buffer_bit = arith.andi %k_block_i32, %c1_i32 : i32
  %buffer_offset_i32 = arith.shli %buffer_bit, %c12_i32 : i32  // multiply by 4096, then adjust
  %buffer_offset_bytes = arith.muli %buffer_offset_i32, %c3_i32 : i32  // 4096 * 3 = 12288
  %buffer_offset = arith.index_cast %buffer_offset_bytes : i32 to index

  // Create read buffer views (current iteration)
  %shared_a_read = memref.view %alloc[%buffer_offset][] : memref<37120xi8, #gpu.address_space<workgroup>>
    to memref<64x48xf16, #gpu.address_space<workgroup>>
  %shared_b_read_offset = arith.addi %buffer_offset, %c6144 : index
  %shared_b_read = memref.view %alloc[%shared_b_read_offset][] : memref<37120xi8, #gpu.address_space<workgroup>>
    to memref<64x48xf16, #gpu.address_space<workgroup>>

  // Create write buffer views (next iteration) - XOR with 12288 to toggle
  %buffer_offset_write = arith.xori %buffer_offset, %c12288 : index
  %shared_a_write = memref.view %alloc[%buffer_offset_write][] : memref<37120xi8, #gpu.address_space<workgroup>>
    to memref<64x48xf16, #gpu.address_space<workgroup>>
  %shared_b_write_offset = arith.addi %buffer_offset_write, %c6144 : index
  %shared_b_write = memref.view %alloc[%shared_b_write_offset][] : memref<37120xi8, #gpu.address_space<workgroup>>
    to memref<64x48xf16, #gpu.address_space<workgroup>>

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
  %a0 = vector.load %shared_a_read[%load_row, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a1 = vector.load %shared_a_read[%load_row_1, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a2 = vector.load %shared_a_read[%load_row_2, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a3 = vector.load %shared_a_read[%load_row_3, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
  %b0 = vector.load %shared_b_read[%load_row, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b1 = vector.load %shared_b_read[%load_row_1, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b2 = vector.load %shared_b_read[%load_row_2, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b3 = vector.load %shared_b_read[%load_row_3, %load_col] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
  %a0k = vector.load %shared_a_read[%load_row, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a1k = vector.load %shared_a_read[%load_row_1, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a2k = vector.load %shared_a_read[%load_row_2, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %a3k = vector.load %shared_a_read[%load_row_3, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
  %b0k = vector.load %shared_b_read[%load_row, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b1k = vector.load %shared_b_read[%load_row_1, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b2k = vector.load %shared_b_read[%load_row_2, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  %b3k = vector.load %shared_b_read[%load_row_3, %load_col_k] :
      memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

  vector.store %a_row_vec_next_first, %shared_a_write[%thread_row_base, %thread_col_offset] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %a_row_vec_next_second, %shared_a_write[%thread_row_second, %thread_col_offset] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %b_row_vec_next_first, %shared_b_write[%thread_row_base, %thread_col_offset] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %b_row_vec_next_second, %shared_b_write[%thread_row_second, %thread_col_offset] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
// Determine which buffer to read from for the last iteration
%last_k_block_i32 = arith.index_cast %num_blocks_minus_1 : index to i32
%last_buffer_bit = arith.andi %last_k_block_i32, %c1_i32 : i32
%last_buffer_offset_i32 = arith.shli %last_buffer_bit, %c12_i32 : i32
%last_buffer_offset_bytes = arith.muli %last_buffer_offset_i32, %c3_i32 : i32
%last_buffer_offset = arith.index_cast %last_buffer_offset_bytes : i32 to index

// Create read buffer view for last iteration
%shared_a_last = memref.view %alloc[%last_buffer_offset][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b_last_offset = arith.addi %last_buffer_offset, %c6144 : index
%shared_b_last = memref.view %alloc[%shared_b_last_offset][] : memref<37120xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>

// Load first half from shared memory
%a0_last = vector.load %shared_a_last[%load_row, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a1_last = vector.load %shared_a_last[%load_row_1, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a2_last = vector.load %shared_a_last[%load_row_2, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a3_last = vector.load %shared_a_last[%load_row_3, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

%b0_last = vector.load %shared_b_last[%load_row, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b1_last = vector.load %shared_b_last[%load_row_1, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b2_last = vector.load %shared_b_last[%load_row_2, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b3_last = vector.load %shared_b_last[%load_row_3, %load_col] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

// Load second half from shared memory
%a0_k_last = vector.load %shared_a_last[%load_row, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a1_k_last = vector.load %shared_a_last[%load_row_1, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a2_k_last = vector.load %shared_a_last[%load_row_2, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%a3_k_last = vector.load %shared_a_last[%load_row_3, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

%b0_k_last = vector.load %shared_b_last[%load_row, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b1_k_last = vector.load %shared_b_last[%load_row_1, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b2_k_last = vector.load %shared_b_last[%load_row_2, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
%b3_k_last = vector.load %shared_b_last[%load_row_3, %load_col_k] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
        %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
        %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
        %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

        %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
        %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
        %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
        %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

        %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
        %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
        %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
        %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

        %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
        %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
        %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
        %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

        %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
        %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
        %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
        %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

        %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
        %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
        %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
        %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

        %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
        %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
        %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
        %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

        %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
        %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
        %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
        %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

        %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
        %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
        %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
        %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

        %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
        %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
        %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
        %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

        %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
        %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
        %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
        %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

        %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
        %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
        %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
        %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

        %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
        %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
        %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
        %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

        %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
        %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
        %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
        %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

        %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
        %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
        %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
        %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

        %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
        %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
        %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
        %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

// Write all 64 elements to LDS (M-tile 0)
memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 1
memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 2
memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 3
memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
%out_token = arith.addi %offs_token_id_base, %thread_id : index
%tok_id_i32 = memref.load %sorted_token_ids_ptr[%out_token] : memref<33335xi32>
%tok_id = arith.index_cast %tok_id_i32 : i32 to index
%out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

scf.if %out_valid {
  // Read 64 elements from LDS (full row, excluding padding)
  %row_data = vector.load %shared_output[%thread_id, %c0] :
    memref<64x96xf16, #gpu.address_space<workgroup>>, vector<64xf16>

  // Write to global memory - fully coalesced!
  %out_base = arith.muli %tok_id, %N : index
  %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
  %out_col = arith.addi %out_base, %out_col_base_global : index

  vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
}
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_96_db::@fused_moe_kernel_16x16x16_padding_lds_96_db(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_global_32 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_96_global_32 {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_96_global_32 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_96_global_32(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64x48 for A, 64x48 for B (instead of 64×6144)
%c6144 = arith.constant 6144 : index
%c12288 = arith.constant 12288 : index

%alloc = memref.alloc() : memref<24576xi8, #gpu.address_space<workgroup>>  // 6144 + 6144 + 10240

%shared_a = memref.view %alloc[%c0][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b = memref.view %alloc[%c6144][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_output = memref.view %alloc[%c12288][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x96xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<32xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<32xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %c0 : index

// Load A - first row (32 elements instead of 16)
%a_row_vec_0_first_wide = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
  memref<16384x6144xf16>, vector<32xf16>

// Load A - second row (32 elements)
%a_row_vec_0_second_wide = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
  memref<16384x6144xf16>, vector<32xf16>

// Split into two 16-element chunks for storing to LDS
%a_row_vec_0_first_chunk1 = vector.extract_strided_slice %a_row_vec_0_first_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
%a_row_vec_0_first_chunk2 = vector.extract_strided_slice %a_row_vec_0_first_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

%a_row_vec_0_second_chunk1 = vector.extract_strided_slice %a_row_vec_0_second_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
%a_row_vec_0_second_chunk2 = vector.extract_strided_slice %a_row_vec_0_second_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

// Store first chunk (columns 0-15) to LDS
vector.store %a_row_vec_0_first_chunk1, %shared_a[%thread_row_base, %c0] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %a_row_vec_0_second_chunk1, %shared_a[%thread_row_second, %c0] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

// Store second chunk (columns 16-31) to LDS
vector.store %a_row_vec_0_first_chunk2, %shared_a[%thread_row_base, %c16] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %a_row_vec_0_second_chunk2, %shared_a[%thread_row_second, %c16] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

// Load B - first row (32 elements)
%b_row_vec_0_first_wide = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
  memref<8x32768x6144xf16>, vector<32xf16>

// Load B - second row (32 elements)
%b_row_vec_0_second_wide = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
  memref<8x32768x6144xf16>, vector<32xf16>

// Split into two 16-element chunks
%b_row_vec_0_first_chunk1 = vector.extract_strided_slice %b_row_vec_0_first_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
%b_row_vec_0_first_chunk2 = vector.extract_strided_slice %b_row_vec_0_first_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

%b_row_vec_0_second_chunk1 = vector.extract_strided_slice %b_row_vec_0_second_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
%b_row_vec_0_second_chunk2 = vector.extract_strided_slice %b_row_vec_0_second_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

// Store to LDS
vector.store %b_row_vec_0_first_chunk1, %shared_b[%thread_row_base, %c0] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %b_row_vec_0_second_chunk1, %shared_b[%thread_row_second, %c0] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

vector.store %b_row_vec_0_first_chunk2, %shared_b[%thread_row_base, %c16] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %b_row_vec_0_second_chunk2, %shared_b[%thread_row_second, %c16] :
  memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %c0 : index

  // Load A - first row (32 elements)
  %a_row_vec_next_first_wide = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
    memref<16384x6144xf16>, vector<32xf16>

  // Load A - second row (32 elements)
  %a_row_vec_next_second_wide = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
    memref<16384x6144xf16>, vector<32xf16>

  // Load B - first row (32 elements)
  %b_row_vec_next_first_wide = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
    memref<8x32768x6144xf16>, vector<32xf16>

  // Load B - second row (32 elements)
  %b_row_vec_next_second_wide = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
    memref<8x32768x6144xf16>, vector<32xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

  // Split A data into chunks
  %a_next_first_chunk1 = vector.extract_strided_slice %a_row_vec_next_first_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  %a_next_first_chunk2 = vector.extract_strided_slice %a_row_vec_next_first_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  
  %a_next_second_chunk1 = vector.extract_strided_slice %a_row_vec_next_second_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  %a_next_second_chunk2 = vector.extract_strided_slice %a_row_vec_next_second_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

  // Split B data into chunks
  %b_next_first_chunk1 = vector.extract_strided_slice %b_row_vec_next_first_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  %b_next_first_chunk2 = vector.extract_strided_slice %b_row_vec_next_first_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  
  %b_next_second_chunk1 = vector.extract_strided_slice %b_row_vec_next_second_wide {offsets = [0], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>
  %b_next_second_chunk2 = vector.extract_strided_slice %b_row_vec_next_second_wide {offsets = [16], sizes = [16], strides = [1]} : vector<32xf16> to vector<16xf16>

  // Store A to LDS
  vector.store %a_next_first_chunk1, %shared_a[%thread_row_base, %c0] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %a_next_second_chunk1, %shared_a[%thread_row_second, %c0] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %a_next_first_chunk2, %shared_a[%thread_row_base, %c16] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %a_next_second_chunk2, %shared_a[%thread_row_second, %c16] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

  // Store B to LDS
  vector.store %b_next_first_chunk1, %shared_b[%thread_row_base, %c0] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %b_next_second_chunk1, %shared_b[%thread_row_second, %c0] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %b_next_first_chunk2, %shared_b[%thread_row_base, %c16] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  vector.store %b_next_second_chunk2, %shared_b[%thread_row_second, %c16] :
    memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
        %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
        %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
        %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

        %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
        %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
        %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
        %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

        %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
        %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
        %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
        %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

        %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
        %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
        %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
        %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

        %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
        %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
        %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
        %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

        %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
        %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
        %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
        %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

        %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
        %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
        %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
        %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

        %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
        %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
        %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
        %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

        %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
        %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
        %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
        %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

        %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
        %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
        %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
        %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

        %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
        %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
        %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
        %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

        %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
        %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
        %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
        %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

        %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
        %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
        %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
        %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

        %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
        %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
        %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
        %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

        %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
        %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
        %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
        %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

        %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
        %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
        %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
        %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

// Write all 64 elements to LDS (M-tile 0)
memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 1
memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 2
memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 3
memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
%out_token = arith.addi %offs_token_id_base, %thread_id : index
%tok_id_i32 = memref.load %sorted_token_ids_ptr[%out_token] : memref<33335xi32>
%tok_id = arith.index_cast %tok_id_i32 : i32 to index
%out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

scf.if %out_valid {
  // Read 64 elements from LDS (full row, excluding padding)
  %row_data = vector.load %shared_output[%thread_id, %c0] :
    memref<64x96xf16, #gpu.address_space<workgroup>>, vector<64xf16>

  // Write to global memory - fully coalesced!
  %out_base = arith.muli %tok_id, %N : index
  %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
  %out_col = arith.addi %out_base, %out_col_base_global : index

  vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
}
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_96_global_32::@fused_moe_kernel_16x16x16_padding_lds_96_global_32(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_96 {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_96 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_96(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64x48 for A, 64x48 for B (instead of 64×6144)
%c6144 = arith.constant 6144 : index
%c12288 = arith.constant 12288 : index

%alloc = memref.alloc() : memref<24576xi8, #gpu.address_space<workgroup>>  // 6144 + 6144 + 10240

%shared_a = memref.view %alloc[%c0][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b = memref.view %alloc[%c6144][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_output = memref.view %alloc[%c12288][] : memref<24576xi8, #gpu.address_space<workgroup>>
  to memref<64x96xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
        %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
        %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
        %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

        %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
        %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
        %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
        %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

        %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
        %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
        %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
        %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

        %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
        %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
        %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
        %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

        %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
        %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
        %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
        %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

        %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
        %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
        %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
        %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

        %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
        %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
        %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
        %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

        %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
        %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
        %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
        %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

        %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
        %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
        %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
        %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

        %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
        %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
        %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
        %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

        %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
        %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
        %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
        %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

        %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
        %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
        %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
        %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

        %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
        %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
        %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
        %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

        %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
        %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
        %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
        %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

        %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
        %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
        %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
        %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

        %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
        %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
        %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
        %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

// Write all 64 elements to LDS (M-tile 0)
memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 1
memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 2
memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

// M-tile 3
memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
%out_token = arith.addi %offs_token_id_base, %thread_id : index
%tok_id_i32 = memref.load %sorted_token_ids_ptr[%out_token] : memref<33335xi32>
%tok_id = arith.index_cast %tok_id_i32 : i32 to index
%out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

scf.if %out_valid {
  // Read 64 elements from LDS (full row, excluding padding)
  %row_data = vector.load %shared_output[%thread_id, %c0] :
    memref<64x96xf16, #gpu.address_space<workgroup>>, vector<64xf16>

  // Write to global memory - fully coalesced!
  %out_base = arith.muli %tok_id, %N : index
  %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
  %out_col = arith.addi %out_base, %out_col_base_global : index

  vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
}
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_96::@fused_moe_kernel_16x16x16_padding_lds_96(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64 {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64(
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 64 : index

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
        %c64 = arith.constant 64 : index
        %c127 = arith.constant 127 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
          %thread_token_id = arith.addi %offs_token_id_base, %thread_id : index

          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          %a_row = arith.divui %token_id, %top_k : index

          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory with padding: 64x68 for A, 64x68 for B, 64x96 for output
          // 64x68 = 4352 elements * 2 bytes = 8704 bytes per buffer
          // Total: 8704 + 8704 + 12288 = 29696 bytes
          %c8704 = arith.constant 8704 : index
          %c17408 = arith.constant 17408 : index

          %alloc = memref.alloc() : memref<29696xi8, #gpu.address_space<workgroup>>

          %shared_a = memref.view %alloc[%c0][] : memref<29696xi8, #gpu.address_space<workgroup>>
            to memref<64x68xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c8704][] : memref<29696xi8, #gpu.address_space<workgroup>>
            to memref<64x68xf16, #gpu.address_space<workgroup>>
          %shared_output = memref.view %alloc[%c17408][] : memref<29696xi8, #gpu.address_space<workgroup>>
            to memref<64x96xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for K dimension (split 64 into 4x16)
          %load_col_k1 = arith.addi %load_col, %c16 : index
          %load_col_k2 = arith.addi %load_col, %c32 : index
          %load_col_k3 = arith.addi %load_col, %c48 : index

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment for loading
          %thread_row_base = arith.divui %thread_id, %c2 : index
          %thread_col_group = arith.remui %thread_id, %c2 : index
          %thread_col_offset = arith.muli %thread_col_group, %c32 : index  // 0 or 32

          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<32xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<32xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (32 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<32xf16>

          // Load A - second row (32 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<32xf16>

          // Store A to shared memory (with padding - stride is 80, not 64)
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (32 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<32xf16>

          // Load B - second row (32 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<32xf16>

          // Store B to shared memory (with padding - stride is 80, not 64)
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
              iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                        %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                        %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                        %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
              -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

            // =========================================================================
            // LOAD K[0:16] from shared memory
            // =========================================================================
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index
            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<32xf16>
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<32xf16>

            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<32xf16>
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<32xf16>

            // =========================================================================
            // LOAD K[16:32] from shared memory
            // =========================================================================
            %a0k1 = vector.load %shared_a[%load_row, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k1 = vector.load %shared_a[%load_row_1, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k1 = vector.load %shared_a[%load_row_2, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k1 = vector.load %shared_a[%load_row_3, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k1 = vector.load %shared_b[%load_row, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k1 = vector.load %shared_b[%load_row_1, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k1 = vector.load %shared_b[%load_row_2, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k1 = vector.load %shared_b[%load_row_3, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[0:16]
            // =========================================================================
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // LOAD K[32:48] from shared memory
            // =========================================================================
            %a0k2 = vector.load %shared_a[%load_row, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k2 = vector.load %shared_a[%load_row_1, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k2 = vector.load %shared_a[%load_row_2, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k2 = vector.load %shared_a[%load_row_3, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k2 = vector.load %shared_b[%load_row, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k2 = vector.load %shared_b[%load_row_1, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k2 = vector.load %shared_b[%load_row_2, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k2 = vector.load %shared_b[%load_row_3, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[16:32]
            // =========================================================================
            %r00_1 = amdgpu.mfma %a0k1 * %b0k1 + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_1 = amdgpu.mfma %a0k1 * %b1k1 + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_1 = amdgpu.mfma %a0k1 * %b2k1 + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_1 = amdgpu.mfma %a0k1 * %b3k1 + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_1 = amdgpu.mfma %a1k1 * %b0k1 + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_1 = amdgpu.mfma %a1k1 * %b1k1 + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_1 = amdgpu.mfma %a1k1 * %b2k1 + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_1 = amdgpu.mfma %a1k1 * %b3k1 + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_1 = amdgpu.mfma %a2k1 * %b0k1 + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_1 = amdgpu.mfma %a2k1 * %b1k1 + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_1 = amdgpu.mfma %a2k1 * %b2k1 + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_1 = amdgpu.mfma %a2k1 * %b3k1 + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_1 = amdgpu.mfma %a3k1 * %b0k1 + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_1 = amdgpu.mfma %a3k1 * %b1k1 + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_1 = amdgpu.mfma %a3k1 * %b2k1 + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_1 = amdgpu.mfma %a3k1 * %b3k1 + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // LOAD K[48:64] from shared memory
            // =========================================================================
            %a0k3 = vector.load %shared_a[%load_row, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k3 = vector.load %shared_a[%load_row_1, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k3 = vector.load %shared_a[%load_row_2, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k3 = vector.load %shared_a[%load_row_3, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k3 = vector.load %shared_b[%load_row, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k3 = vector.load %shared_b[%load_row_1, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k3 = vector.load %shared_b[%load_row_2, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k3 = vector.load %shared_b[%load_row_3, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[32:48]
            // =========================================================================
            %r00_2 = amdgpu.mfma %a0k2 * %b0k2 + %r00_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_2 = amdgpu.mfma %a0k2 * %b1k2 + %r01_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_2 = amdgpu.mfma %a0k2 * %b2k2 + %r02_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_2 = amdgpu.mfma %a0k2 * %b3k2 + %r03_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_2 = amdgpu.mfma %a1k2 * %b0k2 + %r10_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_2 = amdgpu.mfma %a1k2 * %b1k2 + %r11_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_2 = amdgpu.mfma %a1k2 * %b2k2 + %r12_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_2 = amdgpu.mfma %a1k2 * %b3k2 + %r13_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_2 = amdgpu.mfma %a2k2 * %b0k2 + %r20_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_2 = amdgpu.mfma %a2k2 * %b1k2 + %r21_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_2 = amdgpu.mfma %a2k2 * %b2k2 + %r22_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_2 = amdgpu.mfma %a2k2 * %b3k2 + %r23_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_2 = amdgpu.mfma %a3k2 * %b0k2 + %r30_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_2 = amdgpu.mfma %a3k2 * %b1k2 + %r31_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_2 = amdgpu.mfma %a3k2 * %b2k2 + %r32_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_2 = amdgpu.mfma %a3k2 * %b3k2 + %r33_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA K[48:64] - Final accumulation
            // =========================================================================
            %r00 = amdgpu.mfma %a0k3 * %b0k3 + %r00_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01 = amdgpu.mfma %a0k3 * %b1k3 + %r01_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02 = amdgpu.mfma %a0k3 * %b2k3 + %r02_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03 = amdgpu.mfma %a0k3 * %b3k3 + %r03_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10 = amdgpu.mfma %a1k3 * %b0k3 + %r10_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11 = amdgpu.mfma %a1k3 * %b1k3 + %r11_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12 = amdgpu.mfma %a1k3 * %b2k3 + %r12_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13 = amdgpu.mfma %a1k3 * %b3k3 + %r13_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20 = amdgpu.mfma %a2k3 * %b0k3 + %r20_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21 = amdgpu.mfma %a2k3 * %b1k3 + %r21_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22 = amdgpu.mfma %a2k3 * %b2k3 + %r22_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23 = amdgpu.mfma %a2k3 * %b3k3 + %r23_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30 = amdgpu.mfma %a3k3 * %b0k3 + %r30_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31 = amdgpu.mfma %a3k3 * %b1k3 + %r31_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32 = amdgpu.mfma %a3k3 * %b2k3 + %r32_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33 = amdgpu.mfma %a3k3 * %b3k3 + %r33_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration
          // =========================================================================
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k1_last = vector.load %shared_a[%load_row, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k1_last = vector.load %shared_a[%load_row_1, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k1_last = vector.load %shared_a[%load_row_2, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k1_last = vector.load %shared_a[%load_row_3, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k1_last = vector.load %shared_b[%load_row, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k1_last = vector.load %shared_b[%load_row_1, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k1_last = vector.load %shared_b[%load_row_2, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k1_last = vector.load %shared_b[%load_row_3, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k2_last = vector.load %shared_a[%load_row, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k2_last = vector.load %shared_a[%load_row_1, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k2_last = vector.load %shared_a[%load_row_2, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k2_last = vector.load %shared_a[%load_row_3, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k2_last = vector.load %shared_b[%load_row, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k2_last = vector.load %shared_b[%load_row_1, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k2_last = vector.load %shared_b[%load_row_2, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k2_last = vector.load %shared_b[%load_row_3, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k3_last = vector.load %shared_a[%load_row, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k3_last = vector.load %shared_a[%load_row_1, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k3_last = vector.load %shared_a[%load_row_2, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k3_last = vector.load %shared_a[%load_row_3, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k3_last = vector.load %shared_b[%load_row, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k3_last = vector.load %shared_b[%load_row_1, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k3_last = vector.load %shared_b[%load_row_2, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k3_last = vector.load %shared_b[%load_row_3, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute K[0:16]
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[16:32]
          %r00_1_last = amdgpu.mfma %a0_k1_last * %b0_k1_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_1_last = amdgpu.mfma %a0_k1_last * %b1_k1_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_1_last = amdgpu.mfma %a0_k1_last * %b2_k1_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_1_last = amdgpu.mfma %a0_k1_last * %b3_k1_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_1_last = amdgpu.mfma %a1_k1_last * %b0_k1_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_1_last = amdgpu.mfma %a1_k1_last * %b1_k1_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_1_last = amdgpu.mfma %a1_k1_last * %b2_k1_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_1_last = amdgpu.mfma %a1_k1_last * %b3_k1_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_1_last = amdgpu.mfma %a2_k1_last * %b0_k1_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_1_last = amdgpu.mfma %a2_k1_last * %b1_k1_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_1_last = amdgpu.mfma %a2_k1_last * %b2_k1_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_1_last = amdgpu.mfma %a2_k1_last * %b3_k1_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_1_last = amdgpu.mfma %a3_k1_last * %b0_k1_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_1_last = amdgpu.mfma %a3_k1_last * %b1_k1_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_1_last = amdgpu.mfma %a3_k1_last * %b2_k1_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_1_last = amdgpu.mfma %a3_k1_last * %b3_k1_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[32:48]
          %r00_2_last = amdgpu.mfma %a0_k2_last * %b0_k2_last + %r00_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_2_last = amdgpu.mfma %a0_k2_last * %b1_k2_last + %r01_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_2_last = amdgpu.mfma %a0_k2_last * %b2_k2_last + %r02_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_2_last = amdgpu.mfma %a0_k2_last * %b3_k2_last + %r03_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_2_last = amdgpu.mfma %a1_k2_last * %b0_k2_last + %r10_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_2_last = amdgpu.mfma %a1_k2_last * %b1_k2_last + %r11_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_2_last = amdgpu.mfma %a1_k2_last * %b2_k2_last + %r12_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_2_last = amdgpu.mfma %a1_k2_last * %b3_k2_last + %r13_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_2_last = amdgpu.mfma %a2_k2_last * %b0_k2_last + %r20_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_2_last = amdgpu.mfma %a2_k2_last * %b1_k2_last + %r21_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_2_last = amdgpu.mfma %a2_k2_last * %b2_k2_last + %r22_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_2_last = amdgpu.mfma %a2_k2_last * %b3_k2_last + %r23_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_2_last = amdgpu.mfma %a3_k2_last * %b0_k2_last + %r30_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_2_last = amdgpu.mfma %a3_k2_last * %b1_k2_last + %r31_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_2_last = amdgpu.mfma %a3_k2_last * %b2_k2_last + %r32_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_2_last = amdgpu.mfma %a3_k2_last * %b3_k2_last + %r33_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[48:64] - Final results
          %r00_final = amdgpu.mfma %a0_k3_last * %b0_k3_last + %r00_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k3_last * %b1_k3_last + %r01_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k3_last * %b2_k3_last + %r02_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k3_last * %b3_k3_last + %r03_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k3_last * %b0_k3_last + %r10_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k3_last * %b1_k3_last + %r11_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k3_last * %b2_k3_last + %r12_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k3_last * %b3_k3_last + %r13_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k3_last * %b0_k3_last + %r20_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k3_last * %b1_k3_last + %r21_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k3_last * %b2_k3_last + %r22_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k3_last * %b3_k3_last + %r23_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k3_last * %b0_k3_last + %r30_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k3_last * %b1_k3_last + %r31_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k3_last * %b2_k3_last + %r32_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k3_last * %b3_k3_last + %r33_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
          %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
          %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
          %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

          %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
          %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
          %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
          %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

          %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
          %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
          %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
          %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

          %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
          %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
          %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
          %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

          %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
          %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
          %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
          %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

          %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
          %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
          %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
          %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

          %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
          %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
          %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
          %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

          %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
          %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
          %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
          %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

          %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
          %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
          %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
          %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

          %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
          %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
          %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
          %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

          %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
          %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
          %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
          %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

          %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
          %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
          %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
          %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

          %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
          %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
          %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
          %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

          %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
          %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
          %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
          %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

          %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
          %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
          %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
          %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

          %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
          %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
          %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
          %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

          // Write all 64 elements to LDS (M-tile 0)
          memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 1
          memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 2
          memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 3
          memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
          %out_token = arith.addi %offs_token_id_base, %thread_id : index
          %tok_id_i32 = memref.load %sorted_token_ids_ptr[%out_token] : memref<33335xi32>
          %tok_id = arith.index_cast %tok_id_i32 : i32 to index
          %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

          scf.if %out_valid {
            // Read 64 elements from LDS (full row, excluding padding)
            %row_data = vector.load %shared_output[%thread_id, %c0] :
              memref<64x96xf16, #gpu.address_space<workgroup>>, vector<64xf16>

            // Write to global memory - fully coalesced!
            %out_base = arith.muli %tok_id, %N : index
            %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
            %out_col = arith.addi %out_base, %out_col_base_global : index

            vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
          }
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64::@fused_moe_kernel_16x16x16_padding_lds_96_block_k_64(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}

    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64_lds_sorted_tok_ids {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64_lds_sorted_tok_ids workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64_lds_sorted_tok_ids(
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 64 : index

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
        %c64 = arith.constant 64 : index
        %c127 = arith.constant 127 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<4xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          // Allocate shared memory with padding: 64x68 for A, 64x68 for B, 64x96 for output, 384 for sorted_token_ids
          // 64x68 = 4352 elements * 2 bytes = 8704 bytes per buffer
          // Total: 8704 + 8704 + 12288 + 384 = 30080 bytes
          %c8704 = arith.constant 8704 : index
          %c17408 = arith.constant 17408 : index
          %c29696 = arith.constant 29696 : index

          %alloc = memref.alloc() : memref<30080xi8, #gpu.address_space<workgroup>>

          %shared_a = memref.view %alloc[%c0][] : memref<30080xi8, #gpu.address_space<workgroup>>
            to memref<64x68xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c8704][] : memref<30080xi8, #gpu.address_space<workgroup>>
            to memref<64x68xf16, #gpu.address_space<workgroup>>
          %shared_output = memref.view %alloc[%c17408][] : memref<30080xi8, #gpu.address_space<workgroup>>
            to memref<64x96xf16, #gpu.address_space<workgroup>>
          %shared_token_ids = memref.view %alloc[%c29696][] : memref<30080xi8, #gpu.address_space<workgroup>>
            to memref<96xi32, #gpu.address_space<workgroup>>

          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index
	      %thread_token_id_global = arith.addi %offs_token_id_base, %thread_id : index
          %token_id_val_global = memref.load %sorted_token_ids_ptr[%thread_token_id_global] : memref<33335xi32>
          memref.store %token_id_val_global, %shared_token_ids[%thread_id] : memref<96xi32, #gpu.address_space<workgroup>>

          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for K dimension (split 64 into 4x16)
          %load_col_k1 = arith.addi %load_col, %c16 : index
          %load_col_k2 = arith.addi %load_col, %c32 : index
          %load_col_k3 = arith.addi %load_col, %c48 : index

          // Compute thread's row and column assignment for loading
          %thread_row_base = arith.divui %thread_id, %c2 : index
          %thread_col_group = arith.remui %thread_id, %c2 : index
          %thread_col_offset = arith.muli %thread_col_group, %c32 : index  // 0 or 32

          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          amdgpu.lds_barrier

          // Now load from LDS instead of global memory
          %token_id_val = memref.load %shared_token_ids[%thread_id] : memref<96xi32, #gpu.address_space<workgroup>>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          %a_row = arith.divui %token_id, %top_k : index

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Get token IDs for both rows
          %token_id_val_first = memref.load %shared_token_ids[%thread_row_base] : memref<96xi32, #gpu.address_space<workgroup>>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %token_id_val_second = memref.load %shared_token_ids[%thread_row_second] : memref<96xi32, #gpu.address_space<workgroup>>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<32xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<32xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (32 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<32xf16>

          // Load A - second row (32 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<32xf16>

          // Store A to shared memory (with padding - stride is 68, not 64)
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (32 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<32xf16>

          // Load B - second row (32 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<32xf16>

          // Store B to shared memory (with padding - stride is 68, not 64)
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
              iter_args(%a00=%cst_mfma, %a01=%cst_mfma, %a02=%cst_mfma, %a03=%cst_mfma,
                        %a10=%cst_mfma, %a11=%cst_mfma, %a12=%cst_mfma, %a13=%cst_mfma,
                        %a20=%cst_mfma, %a21=%cst_mfma, %a22=%cst_mfma, %a23=%cst_mfma,
                        %a30=%cst_mfma, %a31=%cst_mfma, %a32=%cst_mfma, %a33=%cst_mfma)
              -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                  vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {

            // =========================================================================
            // LOAD K[0:16] from shared memory
            // =========================================================================
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index
            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<32xf16>
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<32xf16>

            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<32xf16>
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<32xf16>

            // =========================================================================
            // LOAD K[16:32] from shared memory
            // =========================================================================
            %a0k1 = vector.load %shared_a[%load_row, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k1 = vector.load %shared_a[%load_row_1, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k1 = vector.load %shared_a[%load_row_2, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k1 = vector.load %shared_a[%load_row_3, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k1 = vector.load %shared_b[%load_row, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k1 = vector.load %shared_b[%load_row_1, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k1 = vector.load %shared_b[%load_row_2, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k1 = vector.load %shared_b[%load_row_3, %load_col_k1] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[0:16]
            // =========================================================================
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // LOAD K[32:48] from shared memory
            // =========================================================================
            %a0k2 = vector.load %shared_a[%load_row, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k2 = vector.load %shared_a[%load_row_1, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k2 = vector.load %shared_a[%load_row_2, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k2 = vector.load %shared_a[%load_row_3, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k2 = vector.load %shared_b[%load_row, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k2 = vector.load %shared_b[%load_row_1, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k2 = vector.load %shared_b[%load_row_2, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k2 = vector.load %shared_b[%load_row_3, %load_col_k2] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[16:32]
            // =========================================================================
            %r00_1 = amdgpu.mfma %a0k1 * %b0k1 + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_1 = amdgpu.mfma %a0k1 * %b1k1 + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_1 = amdgpu.mfma %a0k1 * %b2k1 + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_1 = amdgpu.mfma %a0k1 * %b3k1 + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_1 = amdgpu.mfma %a1k1 * %b0k1 + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_1 = amdgpu.mfma %a1k1 * %b1k1 + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_1 = amdgpu.mfma %a1k1 * %b2k1 + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_1 = amdgpu.mfma %a1k1 * %b3k1 + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_1 = amdgpu.mfma %a2k1 * %b0k1 + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_1 = amdgpu.mfma %a2k1 * %b1k1 + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_1 = amdgpu.mfma %a2k1 * %b2k1 + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_1 = amdgpu.mfma %a2k1 * %b3k1 + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_1 = amdgpu.mfma %a3k1 * %b0k1 + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_1 = amdgpu.mfma %a3k1 * %b1k1 + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_1 = amdgpu.mfma %a3k1 * %b2k1 + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_1 = amdgpu.mfma %a3k1 * %b3k1 + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // LOAD K[48:64] from shared memory
            // =========================================================================
            %a0k3 = vector.load %shared_a[%load_row, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k3 = vector.load %shared_a[%load_row_1, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k3 = vector.load %shared_a[%load_row_2, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k3 = vector.load %shared_a[%load_row_3, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %b0k3 = vector.load %shared_b[%load_row, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k3 = vector.load %shared_b[%load_row_1, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k3 = vector.load %shared_b[%load_row_2, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k3 = vector.load %shared_b[%load_row_3, %load_col_k3] :
                memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA K[32:48]
            // =========================================================================
            %r00_2 = amdgpu.mfma %a0k2 * %b0k2 + %r00_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01_2 = amdgpu.mfma %a0k2 * %b1k2 + %r01_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02_2 = amdgpu.mfma %a0k2 * %b2k2 + %r02_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03_2 = amdgpu.mfma %a0k2 * %b3k2 + %r03_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10_2 = amdgpu.mfma %a1k2 * %b0k2 + %r10_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11_2 = amdgpu.mfma %a1k2 * %b1k2 + %r11_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12_2 = amdgpu.mfma %a1k2 * %b2k2 + %r12_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13_2 = amdgpu.mfma %a1k2 * %b3k2 + %r13_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20_2 = amdgpu.mfma %a2k2 * %b0k2 + %r20_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21_2 = amdgpu.mfma %a2k2 * %b1k2 + %r21_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22_2 = amdgpu.mfma %a2k2 * %b2k2 + %r22_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23_2 = amdgpu.mfma %a2k2 * %b3k2 + %r23_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30_2 = amdgpu.mfma %a3k2 * %b0k2 + %r30_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31_2 = amdgpu.mfma %a3k2 * %b1k2 + %r31_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32_2 = amdgpu.mfma %a3k2 * %b2k2 + %r32_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33_2 = amdgpu.mfma %a3k2 * %b3k2 + %r33_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<32xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA K[48:64] - Final accumulation
            // =========================================================================
            %r00 = amdgpu.mfma %a0k3 * %b0k3 + %r00_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r01 = amdgpu.mfma %a0k3 * %b1k3 + %r01_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r02 = amdgpu.mfma %a0k3 * %b2k3 + %r02_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r03 = amdgpu.mfma %a0k3 * %b3k3 + %r03_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r10 = amdgpu.mfma %a1k3 * %b0k3 + %r10_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r11 = amdgpu.mfma %a1k3 * %b1k3 + %r11_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r12 = amdgpu.mfma %a1k3 * %b2k3 + %r12_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r13 = amdgpu.mfma %a1k3 * %b3k3 + %r13_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r20 = amdgpu.mfma %a2k3 * %b0k3 + %r20_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r21 = amdgpu.mfma %a2k3 * %b1k3 + %r21_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r22 = amdgpu.mfma %a2k3 * %b2k3 + %r22_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r23 = amdgpu.mfma %a2k3 * %b3k3 + %r23_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            %r30 = amdgpu.mfma %a3k3 * %b0k3 + %r30_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r31 = amdgpu.mfma %a3k3 * %b1k3 + %r31_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r32 = amdgpu.mfma %a3k3 * %b2k3 + %r32_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            %r33 = amdgpu.mfma %a3k3 * %b3k3 + %r33_2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration
          // =========================================================================
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k1_last = vector.load %shared_a[%load_row, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k1_last = vector.load %shared_a[%load_row_1, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k1_last = vector.load %shared_a[%load_row_2, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k1_last = vector.load %shared_a[%load_row_3, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k1_last = vector.load %shared_b[%load_row, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k1_last = vector.load %shared_b[%load_row_1, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k1_last = vector.load %shared_b[%load_row_2, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k1_last = vector.load %shared_b[%load_row_3, %load_col_k1] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k2_last = vector.load %shared_a[%load_row, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k2_last = vector.load %shared_a[%load_row_1, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k2_last = vector.load %shared_a[%load_row_2, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k2_last = vector.load %shared_a[%load_row_3, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k2_last = vector.load %shared_b[%load_row, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k2_last = vector.load %shared_b[%load_row_1, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k2_last = vector.load %shared_b[%load_row_2, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k2_last = vector.load %shared_b[%load_row_3, %load_col_k2] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_k3_last = vector.load %shared_a[%load_row, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k3_last = vector.load %shared_a[%load_row_1, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k3_last = vector.load %shared_a[%load_row_2, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k3_last = vector.load %shared_a[%load_row_3, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k3_last = vector.load %shared_b[%load_row, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k3_last = vector.load %shared_b[%load_row_1, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k3_last = vector.load %shared_b[%load_row_2, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k3_last = vector.load %shared_b[%load_row_3, %load_col_k3] :
              memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute K[0:16]
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[16:32]
          %r00_1_last = amdgpu.mfma %a0_k1_last * %b0_k1_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_1_last = amdgpu.mfma %a0_k1_last * %b1_k1_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_1_last = amdgpu.mfma %a0_k1_last * %b2_k1_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_1_last = amdgpu.mfma %a0_k1_last * %b3_k1_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_1_last = amdgpu.mfma %a1_k1_last * %b0_k1_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_1_last = amdgpu.mfma %a1_k1_last * %b1_k1_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_1_last = amdgpu.mfma %a1_k1_last * %b2_k1_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_1_last = amdgpu.mfma %a1_k1_last * %b3_k1_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_1_last = amdgpu.mfma %a2_k1_last * %b0_k1_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_1_last = amdgpu.mfma %a2_k1_last * %b1_k1_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_1_last = amdgpu.mfma %a2_k1_last * %b2_k1_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_1_last = amdgpu.mfma %a2_k1_last * %b3_k1_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_1_last = amdgpu.mfma %a3_k1_last * %b0_k1_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_1_last = amdgpu.mfma %a3_k1_last * %b1_k1_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_1_last = amdgpu.mfma %a3_k1_last * %b2_k1_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_1_last = amdgpu.mfma %a3_k1_last * %b3_k1_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[32:48]
          %r00_2_last = amdgpu.mfma %a0_k2_last * %b0_k2_last + %r00_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_2_last = amdgpu.mfma %a0_k2_last * %b1_k2_last + %r01_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_2_last = amdgpu.mfma %a0_k2_last * %b2_k2_last + %r02_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_2_last = amdgpu.mfma %a0_k2_last * %b3_k2_last + %r03_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_2_last = amdgpu.mfma %a1_k2_last * %b0_k2_last + %r10_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_2_last = amdgpu.mfma %a1_k2_last * %b1_k2_last + %r11_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_2_last = amdgpu.mfma %a1_k2_last * %b2_k2_last + %r12_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_2_last = amdgpu.mfma %a1_k2_last * %b3_k2_last + %r13_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_2_last = amdgpu.mfma %a2_k2_last * %b0_k2_last + %r20_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_2_last = amdgpu.mfma %a2_k2_last * %b1_k2_last + %r21_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_2_last = amdgpu.mfma %a2_k2_last * %b2_k2_last + %r22_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_2_last = amdgpu.mfma %a2_k2_last * %b3_k2_last + %r23_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_2_last = amdgpu.mfma %a3_k2_last * %b0_k2_last + %r30_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_2_last = amdgpu.mfma %a3_k2_last * %b1_k2_last + %r31_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_2_last = amdgpu.mfma %a3_k2_last * %b2_k2_last + %r32_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_2_last = amdgpu.mfma %a3_k2_last * %b3_k2_last + %r33_1_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute K[48:64] - Final results
          %r00_final = amdgpu.mfma %a0_k3_last * %b0_k3_last + %r00_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k3_last * %b1_k3_last + %r01_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k3_last * %b2_k3_last + %r02_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k3_last * %b3_k3_last + %r03_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k3_last * %b0_k3_last + %r10_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k3_last * %b1_k3_last + %r11_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k3_last * %b2_k3_last + %r12_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k3_last * %b3_k3_last + %r13_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k3_last * %b0_k3_last + %r20_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k3_last * %b1_k3_last + %r21_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k3_last * %b2_k3_last + %r22_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k3_last * %b3_k3_last + %r23_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k3_last * %b0_k3_last + %r30_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k3_last * %b1_k3_last + %r31_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k3_last * %b2_k3_last + %r32_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k3_last * %b3_k3_last + %r33_2_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
          %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
          %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
          %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

          %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
          %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
          %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
          %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

          %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
          %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
          %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
          %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

          %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
          %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
          %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
          %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

          %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
          %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
          %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
          %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

          %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
          %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
          %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
          %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

          %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
          %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
          %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
          %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

          %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
          %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
          %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
          %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

          %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
          %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
          %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
          %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

          %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
          %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
          %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
          %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

          %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
          %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
          %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
          %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

          %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
          %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
          %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
          %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

          %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
          %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
          %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
          %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

          %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
          %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
          %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
          %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

          %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
          %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
          %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
          %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

          %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
          %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
          %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
          %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

          // Write all 64 elements to LDS (M-tile 0)
          memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 1
          memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 2
          memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          // M-tile 3
          memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x96xf16, #gpu.address_space<workgroup>>
          memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x96xf16, #gpu.address_space<workgroup>>

          amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
          %tok_id_i32 = memref.load %shared_token_ids[%thread_id] : memref<96xi32, #gpu.address_space<workgroup>>
          %tok_id = arith.index_cast %tok_id_i32 : i32 to index
          %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

          scf.if %out_valid {
            // Read 64 elements from LDS (full row, excluding padding)
            %row_data = vector.load %shared_output[%thread_id, %c0] :
              memref<64x96xf16, #gpu.address_space<workgroup>>, vector<64xf16>

            // Write to global memory - fully coalesced!
            %out_base = arith.muli %tok_id, %N : index
            %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
            %out_col = arith.addi %out_base, %out_col_base_global : index

            vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
          }
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_96_block_k_64_lds_sorted_tok_ids::@fused_moe_kernel_16x16x16_padding_lds_96_block_k_64_lds_sorted_tok_ids(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}

    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding_lds_68 {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding_lds_68 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding_lds_68(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64x48 for A, 64x48 for B (instead of 64×6144)
%c6144 = arith.constant 6144 : index
%c12288 = arith.constant 12288 : index

%alloc = memref.alloc() : memref<21552xi8, #gpu.address_space<workgroup>>  // 6144 + 6144 + 8704 + padding

%shared_a = memref.view %alloc[%c0][] : memref<21552xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_b = memref.view %alloc[%c6144][] : memref<21552xi8, #gpu.address_space<workgroup>>
  to memref<64x48xf16, #gpu.address_space<workgroup>>
%shared_output = memref.view %alloc[%c12288][] : memref<21552xi8, #gpu.address_space<workgroup>>
  to memref<64x68xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
        %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
        %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
        %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

        %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
        %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
        %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
        %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

        %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
        %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
        %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
        %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

        %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
        %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
        %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
        %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

        %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
        %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
        %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
        %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

        %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
        %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
        %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
        %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

        %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
        %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
        %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
        %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

        %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
        %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
        %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
        %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

        %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
        %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
        %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
        %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

        %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
        %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
        %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
        %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

        %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
        %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
        %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
        %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

        %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
        %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
        %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
        %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

        %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
        %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
        %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
        %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

        %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
        %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
        %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
        %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

        %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
        %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
        %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
        %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

        %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
        %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
        %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
        %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

// Write all 64 elements to LDS (M-tile 0)
memref.store %r00_0, %shared_output[%store_row_0_0, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r01_0, %shared_output[%store_row_0_0, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r02_0, %shared_output[%store_row_0_0, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r03_0, %shared_output[%store_row_0_0, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r00_1, %shared_output[%store_row_0_1, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r01_1, %shared_output[%store_row_0_1, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r02_1, %shared_output[%store_row_0_1, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r03_1, %shared_output[%store_row_0_1, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r00_2, %shared_output[%store_row_0_2, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r01_2, %shared_output[%store_row_0_2, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r02_2, %shared_output[%store_row_0_2, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r03_2, %shared_output[%store_row_0_2, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r00_3, %shared_output[%store_row_0_3, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r01_3, %shared_output[%store_row_0_3, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r02_3, %shared_output[%store_row_0_3, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r03_3, %shared_output[%store_row_0_3, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

// M-tile 1
memref.store %r10_0, %shared_output[%store_row_16_0, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r11_0, %shared_output[%store_row_16_0, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r12_0, %shared_output[%store_row_16_0, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r13_0, %shared_output[%store_row_16_0, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r10_1, %shared_output[%store_row_16_1, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r11_1, %shared_output[%store_row_16_1, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r12_1, %shared_output[%store_row_16_1, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r13_1, %shared_output[%store_row_16_1, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r10_2, %shared_output[%store_row_16_2, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r11_2, %shared_output[%store_row_16_2, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r12_2, %shared_output[%store_row_16_2, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r13_2, %shared_output[%store_row_16_2, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r10_3, %shared_output[%store_row_16_3, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r11_3, %shared_output[%store_row_16_3, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r12_3, %shared_output[%store_row_16_3, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r13_3, %shared_output[%store_row_16_3, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

// M-tile 2
memref.store %r20_0, %shared_output[%store_row_32_0, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r21_0, %shared_output[%store_row_32_0, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r22_0, %shared_output[%store_row_32_0, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r23_0, %shared_output[%store_row_32_0, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r20_1, %shared_output[%store_row_32_1, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r21_1, %shared_output[%store_row_32_1, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r22_1, %shared_output[%store_row_32_1, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r23_1, %shared_output[%store_row_32_1, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r20_2, %shared_output[%store_row_32_2, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r21_2, %shared_output[%store_row_32_2, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r22_2, %shared_output[%store_row_32_2, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r23_2, %shared_output[%store_row_32_2, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r20_3, %shared_output[%store_row_32_3, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r21_3, %shared_output[%store_row_32_3, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r22_3, %shared_output[%store_row_32_3, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r23_3, %shared_output[%store_row_32_3, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

// M-tile 3
memref.store %r30_0, %shared_output[%store_row_48_0, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r31_0, %shared_output[%store_row_48_0, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r32_0, %shared_output[%store_row_48_0, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r33_0, %shared_output[%store_row_48_0, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r30_1, %shared_output[%store_row_48_1, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r31_1, %shared_output[%store_row_48_1, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r32_1, %shared_output[%store_row_48_1, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r33_1, %shared_output[%store_row_48_1, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r30_2, %shared_output[%store_row_48_2, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r31_2, %shared_output[%store_row_48_2, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r32_2, %shared_output[%store_row_48_2, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r33_2, %shared_output[%store_row_48_2, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

memref.store %r30_3, %shared_output[%store_row_48_3, %store_col_0] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r31_3, %shared_output[%store_row_48_3, %store_col_1] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r32_3, %shared_output[%store_row_48_3, %store_col_2] : memref<64x68xf16, #gpu.address_space<workgroup>>
memref.store %r33_3, %shared_output[%store_row_48_3, %store_col_3] : memref<64x68xf16, #gpu.address_space<workgroup>>

amdgpu.lds_barrier

          // Each thread reads one row (64 elements) and writes to global memory
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>
%out_token = arith.addi %offs_token_id_base, %thread_id : index
%tok_id_i32 = memref.load %sorted_token_ids_ptr[%out_token] : memref<33335xi32>
%tok_id = arith.index_cast %tok_id_i32 : i32 to index
%out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

scf.if %out_valid {
  // Read 64 elements from LDS (full row, excluding padding)
  %row_data = vector.load %shared_output[%thread_id, %c0] :
    memref<64x68xf16, #gpu.address_space<workgroup>>, vector<64xf16>

  // Write to global memory - fully coalesced!
  %out_base = arith.muli %tok_id, %N : index
  %out_col_base_global = arith.muli %pid_n, %BLOCK_SIZE_N : index
  %out_col = arith.addi %out_base, %out_col_base_global : index

  vector.store %row_data, %c_flat[%out_col] : memref<1073741824xf16>, vector<64xf16>
}
        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding_lds_68::@fused_moe_kernel_16x16x16_padding_lds_68(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 16)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

#map_store_col = affine_map<()[s0] -> (s0 mod 16)>
#map_store_row = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_16x16x16_padding {
    stream.executable.export public @fused_moe_kernel_16x16x16_padding workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_16x16x16_padding(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<32xi1>

          // Compute A row index: token_id // top_k
          %a_row = arith.divui %token_id, %top_k : index

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Compute B row offset for this thread
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %thread_id : index

          // Allocate shared memory: 64x48 for A, 64x48 for B (instead of 64×6144)
          %c6144 = arith.constant 6144 : index
          %alloc = memref.alloc() : memref<12288xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<12288xi8, #gpu.address_space<workgroup>>
            to memref<64x48xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c6144][] : memref<12288xi8, #gpu.address_space<workgroup>>
            to memref<64x48xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0, 4, 8, 12 (first 16 elements of K)
          %load_row = affine.apply #map_load_row()[%thread_id]
          %load_row_1 = arith.addi %load_row, %c16 : index
          %load_row_2 = arith.addi %load_row, %c32 : index
          %load_row_3 = arith.addi %load_row, %c48 : index

          // Compute column indices for first and second half of K (split 32 into 16+16)
          %load_col_k = arith.addi %load_col, %c16 : index  // 16, 20, 24, 28 (second 16 elements of K)

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Compute B rows
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          %result:16 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
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
            %k_col_k = arith.addi %k_start, %load_col_k : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

            // Load A vectors for first half: 4 M tiles
            %a0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1 = vector.load %shared_a[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2 = vector.load %shared_a[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3 = vector.load %shared_a[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for first half: 4 N tiles
            // Note: B is stored as [64, 32] where rows are output features
            // For MFMA, we need B[n, k], which maps to shared_b[load_row, load_col]
            %b0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%load_row_1, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%load_row_2, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%load_row_3, %load_col] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Load A vectors for second half: 4 M tiles
            %a0k = vector.load %shared_a[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1k = vector.load %shared_a[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a2k = vector.load %shared_a[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a3k = vector.load %shared_a[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Load B vectors for second half: 4 N tiles
            %b0k = vector.load %shared_b[%load_row, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1k = vector.load %shared_b[%load_row_1, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2k = vector.load %shared_b[%load_row_2, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3k = vector.load %shared_b[%load_row_3, %load_col_k] :
                memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // Tile (0,0)
            %r00_0 = amdgpu.mfma %a0 * %b0 + %a00 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01_0 = amdgpu.mfma %a0 * %b1 + %a01 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02_0 = amdgpu.mfma %a0 * %b2 + %a02 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03_0 = amdgpu.mfma %a0 * %b3 + %a03 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10_0 = amdgpu.mfma %a1 * %b0 + %a10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11_0 = amdgpu.mfma %a1 * %b1 + %a11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12_0 = amdgpu.mfma %a1 * %b2 + %a12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13_0 = amdgpu.mfma %a1 * %b3 + %a13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20_0 = amdgpu.mfma %a2 * %b0 + %a20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21_0 = amdgpu.mfma %a2 * %b1 + %a21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22_0 = amdgpu.mfma %a2 * %b2 + %a22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23_0 = amdgpu.mfma %a2 * %b3 + %a23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30_0 = amdgpu.mfma %a3 * %b0 + %a30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31_0 = amdgpu.mfma %a3 * %b1 + %a31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32_0 = amdgpu.mfma %a3 * %b2 + %a32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33_0 = amdgpu.mfma %a3 * %b3 + %a33 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // Tile (0,0)
            %r00 = amdgpu.mfma %a0k * %b0k + %r00_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,1)
            %r01 = amdgpu.mfma %a0k * %b1k + %r01_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,2)
            %r02 = amdgpu.mfma %a0k * %b2k + %r02_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (0,3)
            %r03 = amdgpu.mfma %a0k * %b3k + %r03_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (1,0)
            %r10 = amdgpu.mfma %a1k * %b0k + %r10_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,1)
            %r11 = amdgpu.mfma %a1k * %b1k + %r11_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,2)
            %r12 = amdgpu.mfma %a1k * %b2k + %r12_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (1,3)
            %r13 = amdgpu.mfma %a1k * %b3k + %r13_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (2,0)
            %r20 = amdgpu.mfma %a2k * %b0k + %r20_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,1)
            %r21 = amdgpu.mfma %a2k * %b1k + %r21_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,2)
            %r22 = amdgpu.mfma %a2k * %b2k + %r22_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (2,3)
            %r23 = amdgpu.mfma %a2k * %b3k + %r23_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            // Tile (3,0)
            %r30 = amdgpu.mfma %a3k * %b0k + %r30_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,1)
            %r31 = amdgpu.mfma %a3k * %b1k + %r31_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,2)
            %r32 = amdgpu.mfma %a3k * %b2k + %r32_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
            // Tile (3,3)
            %r33 = amdgpu.mfma %a3k * %b3k + %r33_0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

            scf.yield %r00, %r01, %r02, %r03, %r10, %r11, %r12, %r13,
                      %r20, %r21, %r22, %r23, %r30, %r31, %r32, %r33 :
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>,
                vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          // Load first half from shared memory
          %a0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_last = vector.load %shared_a[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_last = vector.load %shared_a[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_last = vector.load %shared_a[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%load_row_1, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%load_row_2, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%load_row_3, %load_col] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Load second half from shared memory
          %a0_k_last = vector.load %shared_a[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_k_last = vector.load %shared_a[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a2_k_last = vector.load %shared_a[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a3_k_last = vector.load %shared_a[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %b0_k_last = vector.load %shared_b[%load_row, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_k_last = vector.load %shared_b[%load_row_1, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_k_last = vector.load %shared_b[%load_row_2, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_k_last = vector.load %shared_b[%load_row_3, %load_col_k] :
              memref<64x48xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          // Compute first half
          %r00_0_last = amdgpu.mfma %a0_last * %b0_last + %result#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_0_last = amdgpu.mfma %a0_last * %b1_last + %result#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_0_last = amdgpu.mfma %a0_last * %b2_last + %result#2 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_0_last = amdgpu.mfma %a0_last * %b3_last + %result#3 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_0_last = amdgpu.mfma %a1_last * %b0_last + %result#4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_0_last = amdgpu.mfma %a1_last * %b1_last + %result#5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_0_last = amdgpu.mfma %a1_last * %b2_last + %result#6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_0_last = amdgpu.mfma %a1_last * %b3_last + %result#7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_0_last = amdgpu.mfma %a2_last * %b0_last + %result#8 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_0_last = amdgpu.mfma %a2_last * %b1_last + %result#9 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_0_last = amdgpu.mfma %a2_last * %b2_last + %result#10 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_0_last = amdgpu.mfma %a2_last * %b3_last + %result#11 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_0_last = amdgpu.mfma %a3_last * %b0_last + %result#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_0_last = amdgpu.mfma %a3_last * %b1_last + %result#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_0_last = amdgpu.mfma %a3_last * %b2_last + %result#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_0_last = amdgpu.mfma %a3_last * %b3_last + %result#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // Compute second half (final results)
          %r00_final = amdgpu.mfma %a0_k_last * %b0_k_last + %r00_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r01_final = amdgpu.mfma %a0_k_last * %b1_k_last + %r01_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r02_final = amdgpu.mfma %a0_k_last * %b2_k_last + %r02_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r03_final = amdgpu.mfma %a0_k_last * %b3_k_last + %r03_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r10_final = amdgpu.mfma %a1_k_last * %b0_k_last + %r10_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r11_final = amdgpu.mfma %a1_k_last * %b1_k_last + %r11_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r12_final = amdgpu.mfma %a1_k_last * %b2_k_last + %r12_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r13_final = amdgpu.mfma %a1_k_last * %b3_k_last + %r13_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r20_final = amdgpu.mfma %a2_k_last * %b0_k_last + %r20_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r21_final = amdgpu.mfma %a2_k_last * %b1_k_last + %r21_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r22_final = amdgpu.mfma %a2_k_last * %b2_k_last + %r22_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r23_final = amdgpu.mfma %a2_k_last * %b3_k_last + %r23_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          %r30_final = amdgpu.mfma %a3_k_last * %b0_k_last + %r30_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r31_final = amdgpu.mfma %a3_k_last * %b1_k_last + %r31_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r32_final = amdgpu.mfma %a3_k_last * %b2_k_last + %r32_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %r33_final = amdgpu.mfma %a3_k_last * %b3_k_last + %r33_0_last {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<4xf32> to vector<4xf16>
          %r01_f16 = arith.truncf %r01_final : vector<4xf32> to vector<4xf16>
          %r02_f16 = arith.truncf %r02_final : vector<4xf32> to vector<4xf16>
          %r03_f16 = arith.truncf %r03_final : vector<4xf32> to vector<4xf16>
          %r10_f16 = arith.truncf %r10_final : vector<4xf32> to vector<4xf16>
          %r11_f16 = arith.truncf %r11_final : vector<4xf32> to vector<4xf16>
          %r12_f16 = arith.truncf %r12_final : vector<4xf32> to vector<4xf16>
          %r13_f16 = arith.truncf %r13_final : vector<4xf32> to vector<4xf16>
          %r20_f16 = arith.truncf %r20_final : vector<4xf32> to vector<4xf16>
          %r21_f16 = arith.truncf %r21_final : vector<4xf32> to vector<4xf16>
          %r22_f16 = arith.truncf %r22_final : vector<4xf32> to vector<4xf16>
          %r23_f16 = arith.truncf %r23_final : vector<4xf32> to vector<4xf16>
          %r30_f16 = arith.truncf %r30_final : vector<4xf32> to vector<4xf16>
          %r31_f16 = arith.truncf %r31_final : vector<4xf32> to vector<4xf16>
          %r32_f16 = arith.truncf %r32_final : vector<4xf32> to vector<4xf16>
          %r33_f16 = arith.truncf %r33_final : vector<4xf32> to vector<4xf16>

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

          %r00_0 = vector.extract %r00_f16[0] : f16 from vector<4xf16>
        %r00_1 = vector.extract %r00_f16[1] : f16 from vector<4xf16>
        %r00_2 = vector.extract %r00_f16[2] : f16 from vector<4xf16>
        %r00_3 = vector.extract %r00_f16[3] : f16 from vector<4xf16>

        %r01_0 = vector.extract %r01_f16[0] : f16 from vector<4xf16>
        %r01_1 = vector.extract %r01_f16[1] : f16 from vector<4xf16>
        %r01_2 = vector.extract %r01_f16[2] : f16 from vector<4xf16>
        %r01_3 = vector.extract %r01_f16[3] : f16 from vector<4xf16>

        %r02_0 = vector.extract %r02_f16[0] : f16 from vector<4xf16>
        %r02_1 = vector.extract %r02_f16[1] : f16 from vector<4xf16>
        %r02_2 = vector.extract %r02_f16[2] : f16 from vector<4xf16>
        %r02_3 = vector.extract %r02_f16[3] : f16 from vector<4xf16>

        %r03_0 = vector.extract %r03_f16[0] : f16 from vector<4xf16>
        %r03_1 = vector.extract %r03_f16[1] : f16 from vector<4xf16>
        %r03_2 = vector.extract %r03_f16[2] : f16 from vector<4xf16>
        %r03_3 = vector.extract %r03_f16[3] : f16 from vector<4xf16>

        %r10_0 = vector.extract %r10_f16[0] : f16 from vector<4xf16>
        %r10_1 = vector.extract %r10_f16[1] : f16 from vector<4xf16>
        %r10_2 = vector.extract %r10_f16[2] : f16 from vector<4xf16>
        %r10_3 = vector.extract %r10_f16[3] : f16 from vector<4xf16>

        %r11_0 = vector.extract %r11_f16[0] : f16 from vector<4xf16>
        %r11_1 = vector.extract %r11_f16[1] : f16 from vector<4xf16>
        %r11_2 = vector.extract %r11_f16[2] : f16 from vector<4xf16>
        %r11_3 = vector.extract %r11_f16[3] : f16 from vector<4xf16>

        %r12_0 = vector.extract %r12_f16[0] : f16 from vector<4xf16>
        %r12_1 = vector.extract %r12_f16[1] : f16 from vector<4xf16>
        %r12_2 = vector.extract %r12_f16[2] : f16 from vector<4xf16>
        %r12_3 = vector.extract %r12_f16[3] : f16 from vector<4xf16>

        %r13_0 = vector.extract %r13_f16[0] : f16 from vector<4xf16>
        %r13_1 = vector.extract %r13_f16[1] : f16 from vector<4xf16>
        %r13_2 = vector.extract %r13_f16[2] : f16 from vector<4xf16>
        %r13_3 = vector.extract %r13_f16[3] : f16 from vector<4xf16>

        %r20_0 = vector.extract %r20_f16[0] : f16 from vector<4xf16>
        %r20_1 = vector.extract %r20_f16[1] : f16 from vector<4xf16>
        %r20_2 = vector.extract %r20_f16[2] : f16 from vector<4xf16>
        %r20_3 = vector.extract %r20_f16[3] : f16 from vector<4xf16>

        %r21_0 = vector.extract %r21_f16[0] : f16 from vector<4xf16>
        %r21_1 = vector.extract %r21_f16[1] : f16 from vector<4xf16>
        %r21_2 = vector.extract %r21_f16[2] : f16 from vector<4xf16>
        %r21_3 = vector.extract %r21_f16[3] : f16 from vector<4xf16>

        %r22_0 = vector.extract %r22_f16[0] : f16 from vector<4xf16>
        %r22_1 = vector.extract %r22_f16[1] : f16 from vector<4xf16>
        %r22_2 = vector.extract %r22_f16[2] : f16 from vector<4xf16>
        %r22_3 = vector.extract %r22_f16[3] : f16 from vector<4xf16>

        %r23_0 = vector.extract %r23_f16[0] : f16 from vector<4xf16>
        %r23_1 = vector.extract %r23_f16[1] : f16 from vector<4xf16>
        %r23_2 = vector.extract %r23_f16[2] : f16 from vector<4xf16>
        %r23_3 = vector.extract %r23_f16[3] : f16 from vector<4xf16>

        %r30_0 = vector.extract %r30_f16[0] : f16 from vector<4xf16>
        %r30_1 = vector.extract %r30_f16[1] : f16 from vector<4xf16>
        %r30_2 = vector.extract %r30_f16[2] : f16 from vector<4xf16>
        %r30_3 = vector.extract %r30_f16[3] : f16 from vector<4xf16>

        %r31_0 = vector.extract %r31_f16[0] : f16 from vector<4xf16>
        %r31_1 = vector.extract %r31_f16[1] : f16 from vector<4xf16>
        %r31_2 = vector.extract %r31_f16[2] : f16 from vector<4xf16>
        %r31_3 = vector.extract %r31_f16[3] : f16 from vector<4xf16>

        %r32_0 = vector.extract %r32_f16[0] : f16 from vector<4xf16>
        %r32_1 = vector.extract %r32_f16[1] : f16 from vector<4xf16>
        %r32_2 = vector.extract %r32_f16[2] : f16 from vector<4xf16>
        %r32_3 = vector.extract %r32_f16[3] : f16 from vector<4xf16>

        %r33_0 = vector.extract %r33_f16[0] : f16 from vector<4xf16>
        %r33_1 = vector.extract %r33_f16[1] : f16 from vector<4xf16>
        %r33_2 = vector.extract %r33_f16[2] : f16 from vector<4xf16>
        %r33_3 = vector.extract %r33_f16[3] : f16 from vector<4xf16>

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

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

          %tok_id_0_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_0] : memref<33335xi32>
          %tok_id_0_0 = arith.index_cast %tok_id_0_0_i32 : i32 to index
          %out_base_0_0 = arith.muli %tok_id_0_0, %N : index
          %out_valid_0_0 = arith.cmpi slt, %tok_id_0_0, %num_valid_tokens : index
          %out_mask_0_0 = vector.broadcast %out_valid_0_0 : i1 to vector<1xi1>
          %tok_id_0_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_1] : memref<33335xi32>
          %tok_id_0_1 = arith.index_cast %tok_id_0_1_i32 : i32 to index
          %out_base_0_1 = arith.muli %tok_id_0_1, %N : index
          %out_valid_0_1 = arith.cmpi slt, %tok_id_0_1, %num_valid_tokens : index
          %out_mask_0_1 = vector.broadcast %out_valid_0_1 : i1 to vector<1xi1>
          %tok_id_0_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_2] : memref<33335xi32>
          %tok_id_0_2 = arith.index_cast %tok_id_0_2_i32 : i32 to index
          %out_base_0_2 = arith.muli %tok_id_0_2, %N : index
          %out_valid_0_2 = arith.cmpi slt, %tok_id_0_2, %num_valid_tokens : index
          %out_mask_0_2 = vector.broadcast %out_valid_0_2 : i1 to vector<1xi1>
          %tok_id_0_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_0_3] : memref<33335xi32>
          %tok_id_0_3 = arith.index_cast %tok_id_0_3_i32 : i32 to index
          %out_base_0_3 = arith.muli %tok_id_0_3, %N : index
          %out_valid_0_3 = arith.cmpi slt, %tok_id_0_3, %num_valid_tokens : index
          %out_mask_0_3 = vector.broadcast %out_valid_0_3 : i1 to vector<1xi1>

          %tok_id_16_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_0] : memref<33335xi32>
          %tok_id_16_0 = arith.index_cast %tok_id_16_0_i32 : i32 to index
          %out_base_16_0 = arith.muli %tok_id_16_0, %N : index
          %out_valid_16_0 = arith.cmpi slt, %tok_id_16_0, %num_valid_tokens : index
          %out_mask_16_0 = vector.broadcast %out_valid_16_0 : i1 to vector<1xi1>
          %tok_id_16_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_1] : memref<33335xi32>
          %tok_id_16_1 = arith.index_cast %tok_id_16_1_i32 : i32 to index
          %out_base_16_1 = arith.muli %tok_id_16_1, %N : index
          %out_valid_16_1 = arith.cmpi slt, %tok_id_16_1, %num_valid_tokens : index
          %out_mask_16_1 = vector.broadcast %out_valid_16_1 : i1 to vector<1xi1>
          %tok_id_16_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_2] : memref<33335xi32>
          %tok_id_16_2 = arith.index_cast %tok_id_16_2_i32 : i32 to index
          %out_base_16_2 = arith.muli %tok_id_16_2, %N : index
          %out_valid_16_2 = arith.cmpi slt, %tok_id_16_2, %num_valid_tokens : index
          %out_mask_16_2 = vector.broadcast %out_valid_16_2 : i1 to vector<1xi1>
          %tok_id_16_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_16_3] : memref<33335xi32>
          %tok_id_16_3 = arith.index_cast %tok_id_16_3_i32 : i32 to index
          %out_base_16_3 = arith.muli %tok_id_16_3, %N : index
          %out_valid_16_3 = arith.cmpi slt, %tok_id_16_3, %num_valid_tokens : index
          %out_mask_16_3 = vector.broadcast %out_valid_16_3 : i1 to vector<1xi1>

          %tok_id_32_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_0] : memref<33335xi32>
          %tok_id_32_0 = arith.index_cast %tok_id_32_0_i32 : i32 to index
          %out_base_32_0 = arith.muli %tok_id_32_0, %N : index
          %out_valid_32_0 = arith.cmpi slt, %tok_id_32_0, %num_valid_tokens : index
          %out_mask_32_0 = vector.broadcast %out_valid_32_0 : i1 to vector<1xi1>
          %tok_id_32_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_1] : memref<33335xi32>
          %tok_id_32_1 = arith.index_cast %tok_id_32_1_i32 : i32 to index
          %out_base_32_1 = arith.muli %tok_id_32_1, %N : index
          %out_valid_32_1 = arith.cmpi slt, %tok_id_32_1, %num_valid_tokens : index
          %out_mask_32_1 = vector.broadcast %out_valid_32_1 : i1 to vector<1xi1>
          %tok_id_32_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_2] : memref<33335xi32>
          %tok_id_32_2 = arith.index_cast %tok_id_32_2_i32 : i32 to index
          %out_base_32_2 = arith.muli %tok_id_32_2, %N : index
          %out_valid_32_2 = arith.cmpi slt, %tok_id_32_2, %num_valid_tokens : index
          %out_mask_32_2 = vector.broadcast %out_valid_32_2 : i1 to vector<1xi1>
          %tok_id_32_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_32_3] : memref<33335xi32>
          %tok_id_32_3 = arith.index_cast %tok_id_32_3_i32 : i32 to index
          %out_base_32_3 = arith.muli %tok_id_32_3, %N : index
          %out_valid_32_3 = arith.cmpi slt, %tok_id_32_3, %num_valid_tokens : index
          %out_mask_32_3 = vector.broadcast %out_valid_32_3 : i1 to vector<1xi1>

          %tok_id_48_0_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_0] : memref<33335xi32>
          %tok_id_48_0 = arith.index_cast %tok_id_48_0_i32 : i32 to index
          %out_base_48_0 = arith.muli %tok_id_48_0, %N : index
          %out_valid_48_0 = arith.cmpi slt, %tok_id_48_0, %num_valid_tokens : index
          %out_mask_48_0 = vector.broadcast %out_valid_48_0 : i1 to vector<1xi1>
          %tok_id_48_1_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_1] : memref<33335xi32>
          %tok_id_48_1 = arith.index_cast %tok_id_48_1_i32 : i32 to index
          %out_base_48_1 = arith.muli %tok_id_48_1, %N : index
          %out_valid_48_1 = arith.cmpi slt, %tok_id_48_1, %num_valid_tokens : index
          %out_mask_48_1 = vector.broadcast %out_valid_48_1 : i1 to vector<1xi1>
          %tok_id_48_2_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_2] : memref<33335xi32>
          %tok_id_48_2 = arith.index_cast %tok_id_48_2_i32 : i32 to index
          %out_base_48_2 = arith.muli %tok_id_48_2, %N : index
          %out_valid_48_2 = arith.cmpi slt, %tok_id_48_2, %num_valid_tokens : index
          %out_mask_48_2 = vector.broadcast %out_valid_48_2 : i1 to vector<1xi1>
          %tok_id_48_3_i32 = memref.load %sorted_token_ids_ptr[%out_token_48_3] : memref<33335xi32>
          %tok_id_48_3 = arith.index_cast %tok_id_48_3_i32 : i32 to index
          %out_base_48_3 = arith.muli %tok_id_48_3, %N : index
          %out_valid_48_3 = arith.cmpi slt, %tok_id_48_3, %num_valid_tokens : index
          %out_mask_48_3 = vector.broadcast %out_valid_48_3 : i1 to vector<1xi1>

          // pid_n determines which 64-neuron block we're computing
          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

          // Column offsets for the 4 column tiles
          %out_col_0 = arith.addi %out_col_base, %store_col_0 : index
          %out_col_1 = arith.addi %out_col_base, %store_col_1 : index
          %out_col_2 = arith.addi %out_col_base, %store_col_2 : index
          %out_col_3 = arith.addi %out_col_base, %store_col_3 : index

          // Row 0_0 (M-tile 0, element 0)
scf.if %out_valid_0_0 {
  %idx_00 = arith.addi %out_base_0_0, %out_col_0 : index
  %idx_01 = arith.addi %out_base_0_0, %out_col_1 : index
  %idx_02 = arith.addi %out_base_0_0, %out_col_2 : index
  %idx_03 = arith.addi %out_base_0_0, %out_col_3 : index

  memref.store %r00_0, %c_flat[%idx_00] : memref<1073741824xf16>
  memref.store %r01_0, %c_flat[%idx_01] : memref<1073741824xf16>
  memref.store %r02_0, %c_flat[%idx_02] : memref<1073741824xf16>
  memref.store %r03_0, %c_flat[%idx_03] : memref<1073741824xf16>
}

// Row 0_1 (M-tile 0, element 1)
scf.if %out_valid_0_1 {
  %idx_00 = arith.addi %out_base_0_1, %out_col_0 : index
  %idx_01 = arith.addi %out_base_0_1, %out_col_1 : index
  %idx_02 = arith.addi %out_base_0_1, %out_col_2 : index
  %idx_03 = arith.addi %out_base_0_1, %out_col_3 : index

  memref.store %r00_1, %c_flat[%idx_00] : memref<1073741824xf16>
  memref.store %r01_1, %c_flat[%idx_01] : memref<1073741824xf16>
  memref.store %r02_1, %c_flat[%idx_02] : memref<1073741824xf16>
  memref.store %r03_1, %c_flat[%idx_03] : memref<1073741824xf16>
}

// Row 0_2 (M-tile 0, element 2)
scf.if %out_valid_0_2 {
  %idx_00 = arith.addi %out_base_0_2, %out_col_0 : index
  %idx_01 = arith.addi %out_base_0_2, %out_col_1 : index
  %idx_02 = arith.addi %out_base_0_2, %out_col_2 : index
  %idx_03 = arith.addi %out_base_0_2, %out_col_3 : index

  memref.store %r00_2, %c_flat[%idx_00] : memref<1073741824xf16>
  memref.store %r01_2, %c_flat[%idx_01] : memref<1073741824xf16>
  memref.store %r02_2, %c_flat[%idx_02] : memref<1073741824xf16>
  memref.store %r03_2, %c_flat[%idx_03] : memref<1073741824xf16>
}

// Row 0_3 (M-tile 0, element 3)
scf.if %out_valid_0_3 {
  %idx_00 = arith.addi %out_base_0_3, %out_col_0 : index
  %idx_01 = arith.addi %out_base_0_3, %out_col_1 : index
  %idx_02 = arith.addi %out_base_0_3, %out_col_2 : index
  %idx_03 = arith.addi %out_base_0_3, %out_col_3 : index

  memref.store %r00_3, %c_flat[%idx_00] : memref<1073741824xf16>
  memref.store %r01_3, %c_flat[%idx_01] : memref<1073741824xf16>
  memref.store %r02_3, %c_flat[%idx_02] : memref<1073741824xf16>
  memref.store %r03_3, %c_flat[%idx_03] : memref<1073741824xf16>
}

// Row 16_0 (M-tile 1, element 0)
scf.if %out_valid_16_0 {
  %idx_10 = arith.addi %out_base_16_0, %out_col_0 : index
  %idx_11 = arith.addi %out_base_16_0, %out_col_1 : index
  %idx_12 = arith.addi %out_base_16_0, %out_col_2 : index
  %idx_13 = arith.addi %out_base_16_0, %out_col_3 : index

  memref.store %r10_0, %c_flat[%idx_10] : memref<1073741824xf16>
  memref.store %r11_0, %c_flat[%idx_11] : memref<1073741824xf16>
  memref.store %r12_0, %c_flat[%idx_12] : memref<1073741824xf16>
  memref.store %r13_0, %c_flat[%idx_13] : memref<1073741824xf16>
}

// Row 16_1 (M-tile 1, element 1)
scf.if %out_valid_16_1 {
  %idx_10 = arith.addi %out_base_16_1, %out_col_0 : index
  %idx_11 = arith.addi %out_base_16_1, %out_col_1 : index
  %idx_12 = arith.addi %out_base_16_1, %out_col_2 : index
  %idx_13 = arith.addi %out_base_16_1, %out_col_3 : index

  memref.store %r10_1, %c_flat[%idx_10] : memref<1073741824xf16>
  memref.store %r11_1, %c_flat[%idx_11] : memref<1073741824xf16>
  memref.store %r12_1, %c_flat[%idx_12] : memref<1073741824xf16>
  memref.store %r13_1, %c_flat[%idx_13] : memref<1073741824xf16>
}

// Row 16_2 (M-tile 1, element 2)
scf.if %out_valid_16_2 {
  %idx_10 = arith.addi %out_base_16_2, %out_col_0 : index
  %idx_11 = arith.addi %out_base_16_2, %out_col_1 : index
  %idx_12 = arith.addi %out_base_16_2, %out_col_2 : index
  %idx_13 = arith.addi %out_base_16_2, %out_col_3 : index

  memref.store %r10_2, %c_flat[%idx_10] : memref<1073741824xf16>
  memref.store %r11_2, %c_flat[%idx_11] : memref<1073741824xf16>
  memref.store %r12_2, %c_flat[%idx_12] : memref<1073741824xf16>
  memref.store %r13_2, %c_flat[%idx_13] : memref<1073741824xf16>
}

// Row 16_3 (M-tile 1, element 3)
scf.if %out_valid_16_3 {
  %idx_10 = arith.addi %out_base_16_3, %out_col_0 : index
  %idx_11 = arith.addi %out_base_16_3, %out_col_1 : index
  %idx_12 = arith.addi %out_base_16_3, %out_col_2 : index
  %idx_13 = arith.addi %out_base_16_3, %out_col_3 : index

  memref.store %r10_3, %c_flat[%idx_10] : memref<1073741824xf16>
  memref.store %r11_3, %c_flat[%idx_11] : memref<1073741824xf16>
  memref.store %r12_3, %c_flat[%idx_12] : memref<1073741824xf16>
  memref.store %r13_3, %c_flat[%idx_13] : memref<1073741824xf16>
}

// Row 32_0 (M-tile 2, element 0)
scf.if %out_valid_32_0 {
  %idx_20 = arith.addi %out_base_32_0, %out_col_0 : index
  %idx_21 = arith.addi %out_base_32_0, %out_col_1 : index
  %idx_22 = arith.addi %out_base_32_0, %out_col_2 : index
  %idx_23 = arith.addi %out_base_32_0, %out_col_3 : index

  memref.store %r20_0, %c_flat[%idx_20] : memref<1073741824xf16>
  memref.store %r21_0, %c_flat[%idx_21] : memref<1073741824xf16>
  memref.store %r22_0, %c_flat[%idx_22] : memref<1073741824xf16>
  memref.store %r23_0, %c_flat[%idx_23] : memref<1073741824xf16>
}

// Row 32_1 (M-tile 2, element 1)
scf.if %out_valid_32_1 {
  %idx_20 = arith.addi %out_base_32_1, %out_col_0 : index
  %idx_21 = arith.addi %out_base_32_1, %out_col_1 : index
  %idx_22 = arith.addi %out_base_32_1, %out_col_2 : index
  %idx_23 = arith.addi %out_base_32_1, %out_col_3 : index

  memref.store %r20_1, %c_flat[%idx_20] : memref<1073741824xf16>
  memref.store %r21_1, %c_flat[%idx_21] : memref<1073741824xf16>
  memref.store %r22_1, %c_flat[%idx_22] : memref<1073741824xf16>
  memref.store %r23_1, %c_flat[%idx_23] : memref<1073741824xf16>
}

// Row 32_2 (M-tile 2, element 2)
scf.if %out_valid_32_2 {
  %idx_20 = arith.addi %out_base_32_2, %out_col_0 : index
  %idx_21 = arith.addi %out_base_32_2, %out_col_1 : index
  %idx_22 = arith.addi %out_base_32_2, %out_col_2 : index
  %idx_23 = arith.addi %out_base_32_2, %out_col_3 : index

  memref.store %r20_2, %c_flat[%idx_20] : memref<1073741824xf16>
  memref.store %r21_2, %c_flat[%idx_21] : memref<1073741824xf16>
  memref.store %r22_2, %c_flat[%idx_22] : memref<1073741824xf16>
  memref.store %r23_2, %c_flat[%idx_23] : memref<1073741824xf16>
}

// Row 32_3 (M-tile 2, element 3)
scf.if %out_valid_32_3 {
  %idx_20 = arith.addi %out_base_32_3, %out_col_0 : index
  %idx_21 = arith.addi %out_base_32_3, %out_col_1 : index
  %idx_22 = arith.addi %out_base_32_3, %out_col_2 : index
  %idx_23 = arith.addi %out_base_32_3, %out_col_3 : index

  memref.store %r20_3, %c_flat[%idx_20] : memref<1073741824xf16>
  memref.store %r21_3, %c_flat[%idx_21] : memref<1073741824xf16>
  memref.store %r22_3, %c_flat[%idx_22] : memref<1073741824xf16>
  memref.store %r23_3, %c_flat[%idx_23] : memref<1073741824xf16>
}

// Row 48_0 (M-tile 3, element 0)
scf.if %out_valid_48_0 {
  %idx_30 = arith.addi %out_base_48_0, %out_col_0 : index
  %idx_31 = arith.addi %out_base_48_0, %out_col_1 : index
  %idx_32 = arith.addi %out_base_48_0, %out_col_2 : index
  %idx_33 = arith.addi %out_base_48_0, %out_col_3 : index

  memref.store %r30_0, %c_flat[%idx_30] : memref<1073741824xf16>
  memref.store %r31_0, %c_flat[%idx_31] : memref<1073741824xf16>
  memref.store %r32_0, %c_flat[%idx_32] : memref<1073741824xf16>
  memref.store %r33_0, %c_flat[%idx_33] : memref<1073741824xf16>
}

// Row 48_1 (M-tile 3, element 1)
scf.if %out_valid_48_1 {
  %idx_30 = arith.addi %out_base_48_1, %out_col_0 : index
  %idx_31 = arith.addi %out_base_48_1, %out_col_1 : index
  %idx_32 = arith.addi %out_base_48_1, %out_col_2 : index
  %idx_33 = arith.addi %out_base_48_1, %out_col_3 : index

  memref.store %r30_1, %c_flat[%idx_30] : memref<1073741824xf16>
  memref.store %r31_1, %c_flat[%idx_31] : memref<1073741824xf16>
  memref.store %r32_1, %c_flat[%idx_32] : memref<1073741824xf16>
  memref.store %r33_1, %c_flat[%idx_33] : memref<1073741824xf16>
}

// Row 48_2 (M-tile 3, element 2)
scf.if %out_valid_48_2 {
  %idx_30 = arith.addi %out_base_48_2, %out_col_0 : index
  %idx_31 = arith.addi %out_base_48_2, %out_col_1 : index
  %idx_32 = arith.addi %out_base_48_2, %out_col_2 : index
  %idx_33 = arith.addi %out_base_48_2, %out_col_3 : index

  memref.store %r30_2, %c_flat[%idx_30] : memref<1073741824xf16>
  memref.store %r31_2, %c_flat[%idx_31] : memref<1073741824xf16>
  memref.store %r32_2, %c_flat[%idx_32] : memref<1073741824xf16>
  memref.store %r33_2, %c_flat[%idx_33] : memref<1073741824xf16>
}

// Row 48_3 (M-tile 3, element 3)
scf.if %out_valid_48_3 {
  %idx_30 = arith.addi %out_base_48_3, %out_col_0 : index
  %idx_31 = arith.addi %out_base_48_3, %out_col_1 : index
  %idx_32 = arith.addi %out_base_48_3, %out_col_2 : index
  %idx_33 = arith.addi %out_base_48_3, %out_col_3 : index

  memref.store %r30_3, %c_flat[%idx_30] : memref<1073741824xf16>
  memref.store %r31_3, %c_flat[%idx_31] : memref<1073741824xf16>
  memref.store %r32_3, %c_flat[%idx_32] : memref<1073741824xf16>
  memref.store %r33_3, %c_flat[%idx_33] : memref<1073741824xf16>
}

        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_16x16x16_padding::@fused_moe_kernel_16x16x16_padding(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8 = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 32)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_32x32x8 {
    stream.executable.export public @fused_moe_kernel_32x32x8 workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_32x32x8(
          // Input memrefs
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // N = 32768
        // K = 6144
        // EM = 33335
        // top_k = 2
        // num_valid_tokens = 32768
        // GROUP_SIZE_M = 8
        // BLOCK_SIZE_M = BLOCK_SIZE_N = 64
        // BLOCK_SIZE_K = 32
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
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
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c24 = arith.constant 24 : index
        %c32 = arith.constant 32 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<16xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
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

          // Load expert ID
          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Allocate shared memory: 64×32 for A, 64×32 for B (instead of 64×6144)
          %c4096 = arith.constant 4096 : index
          %alloc = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>

          // Thread-level indices for MFMA loading
          %load_col = affine.apply #map_load_col()[%thread_id]  // 0 or 4
          %load_row = affine.apply #map_load_row()[%thread_id]  // 0 - 31
          // Column offsets for K dimension (8 elements per MFMA, 4 iterations per 32-element block)
          %load_col_8 = arith.addi %load_col, %c8 : index
          %load_col_16 = arith.addi %load_col, %c16 : index
          %load_col_24 = arith.addi %load_col, %c24 : index

          // Row offsets for second 32x32 tile in M dimension
          %load_row_32 = arith.addi %load_row, %c32 : index

          // =========================================================================
          // SWIZZLED LOADING PATTERN (for bank conflict avoidance)
          // =========================================================================

          // Compute thread's row and column assignment
          %thread_row_base = arith.divui %thread_id, %c2 : index  // 0-31
          %thread_col_group = arith.remui %thread_id, %c2 : index  // 0 or 1
          %thread_col_offset = arith.muli %thread_col_group, %c16 : index  // 0 or 16

          // Compute second row (32 rows apart)
          %thread_row_second = arith.addi %thread_row_base, %c32 : index

          // Get token IDs for both rows
          %thread_token_id_first = arith.addi %offs_token_id_base, %thread_row_base : index
          %token_id_val_first = memref.load %sorted_token_ids_ptr[%thread_token_id_first] : memref<33335xi32>
          %token_id_first = arith.index_cast %token_id_val_first : i32 to index
          %a_row_first = arith.divui %token_id_first, %top_k : index

          %thread_token_id_second = arith.addi %offs_token_id_base, %thread_row_second : index
          %token_id_val_second = memref.load %sorted_token_ids_ptr[%thread_token_id_second] : memref<33335xi32>
          %token_id_second = arith.index_cast %token_id_val_second : i32 to index
          %a_row_second = arith.divui %token_id_second, %top_k : index

          // Compute validity masks
          %token_valid_first = arith.cmpi slt, %token_id_first, %num_valid_tokens : index
          %token_mask_first = vector.broadcast %token_valid_first : i1 to vector<16xi1>
          %token_valid_second = arith.cmpi slt, %token_id_second, %num_valid_tokens : index
          %token_mask_second = vector.broadcast %token_valid_second : i1 to vector<16xi1>

          // Compute B rows
          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row_first = arith.addi %offs_bn_base, %thread_row_base : index
          %b_row_second = arith.addi %offs_bn_base, %thread_row_second : index

          // =========================================================================
          // PROLOGUE: Load first iteration (K=0)
          // =========================================================================
          %k_start_0 = arith.constant 0 : index

          // Compute column start
          %k_col_start = arith.addi %k_start_0, %thread_col_offset : index

          // Load A - first row (16 elements)
          %a_row_vec_0_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start], %f0_f16, %token_mask_first :
            memref<16384x6144xf16>, vector<16xf16>

          // Load A - second row (16 elements)
          %a_row_vec_0_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start], %f0_f16, %token_mask_second :
            memref<16384x6144xf16>, vector<16xf16>

          // Store A to shared memory
          vector.store %a_row_vec_0_first, %shared_a[%thread_row_base, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %a_row_vec_0_second, %shared_a[%thread_row_second, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // Load B - first row (16 elements)
          %b_row_vec_0_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Load B - second row (16 elements)
          %b_row_vec_0_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store B to shared memory
          vector.store %b_row_vec_0_first, %shared_b[%thread_row_base, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0_second, %shared_b[%thread_row_second, %thread_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

//%tid_cond = arith.cmpi eq, %thread_id, %c0 : index
//%pid_cond = arith.cmpi eq, %pid, %c0 : index
//%print = arith.andi %tid_cond, %pid_cond : i1
%print = arith.cmpi eq, %pid, %c0 : index

          // =========================================================================
          // MAIN LOOP: Process iterations 0 to N-2
          // =========================================================================
          // For 64x64 output with 32x32 MFMAs, we need 4 tiles (2x2)
          %result:4 = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
              iter_args(%acc00=%cst_mfma, %acc01=%cst_mfma,
                        %acc10=%cst_mfma, %acc11=%cst_mfma)
              -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {

            // Compute K offset for this iteration
            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index

            // =========================================================================
            // FIRST HALF: K[0:16] - Load from shared memory
            // =========================================================================

//scf.if %print { gpu.printf "k_start %d tid %d load_row %d load_col %d\\n", %k_start, %thread_id, %load_row, %load_col : index, index, index, index }
            // Iteration 1: K[0:8]
            %a0_0 = vector.load %shared_a[%load_row, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1_0 = vector.load %shared_a[%load_row_32, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0_0 = vector.load %shared_b[%load_row, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1_0 = vector.load %shared_b[%load_row_32, %load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Iteration 2: K[8:16]
            %a0_1 = vector.load %shared_a[%load_row, %load_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1_1 = vector.load %shared_a[%load_row_32, %load_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0_1 = vector.load %shared_b[%load_row, %load_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1_1 = vector.load %shared_b[%load_row_32, %load_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // PREFETCH NEXT ITERATION from global memory
            // =========================================================================
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index

            %k_col_start_next = arith.addi %k_start_next, %thread_col_offset : index

            // Load A - first row (16 elements)
            %a_row_vec_next_first = vector.transfer_read %a_ptr[%a_row_first, %k_col_start_next], %f0_f16, %token_mask_first :
              memref<16384x6144xf16>, vector<16xf16>

            // Load A - second row (16 elements)
            %a_row_vec_next_second = vector.transfer_read %a_ptr[%a_row_second, %k_col_start_next], %f0_f16, %token_mask_second :
              memref<16384x6144xf16>, vector<16xf16>

            // Load B - first row (16 elements)
            %b_row_vec_next_first = vector.transfer_read %b_ptr[%expert_id, %b_row_first, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // Load B - second row (16 elements)
            %b_row_vec_next_second = vector.transfer_read %b_ptr[%expert_id, %b_row_second, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            // =========================================================================
            // SECOND HALF: K[16:32] - Load from shared memory
            // =========================================================================

            // Iteration 3: K[16:24]
            %a0_2 = vector.load %shared_a[%load_row, %load_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1_2 = vector.load %shared_a[%load_row_32, %load_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0_2 = vector.load %shared_b[%load_row, %load_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1_2 = vector.load %shared_b[%load_row_32, %load_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Iteration 4: K[24:32]
            %a0_3 = vector.load %shared_a[%load_row, %load_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %a1_3 = vector.load %shared_a[%load_row_32, %load_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0_3 = vector.load %shared_b[%load_row, %load_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1_3 = vector.load %shared_b[%load_row_32, %load_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // =========================================================================
            // MFMA OPERATIONS - FIRST HALF (K[0:16])
            // =========================================================================

            // K[0:8]
            %r00_0 = amdgpu.mfma %a0_0 * %b0_0 + %acc00 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r01_0 = amdgpu.mfma %a0_0 * %b1_0 + %acc01 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r10_0 = amdgpu.mfma %a1_0 * %b0_0 + %acc10 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r11_0 = amdgpu.mfma %a1_0 * %b1_0 + %acc11 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            // K[8:16]
            %r00_1 = amdgpu.mfma %a0_1 * %b0_1 + %r00_0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r01_1 = amdgpu.mfma %a0_1 * %b1_1 + %r01_0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r10_1 = amdgpu.mfma %a1_1 * %b0_1 + %r10_0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r11_1 = amdgpu.mfma %a1_1 * %b1_1 + %r11_0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            // =========================================================================
            // STORE PREFETCHED DATA to shared memory (after first half compute)
            // =========================================================================
            amdgpu.lds_barrier

            vector.store %a_row_vec_next_first, %shared_a[%thread_row_base, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %a_row_vec_next_second, %shared_a[%thread_row_second, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            vector.store %b_row_vec_next_first, %shared_b[%thread_row_base, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next_second, %shared_b[%thread_row_second, %thread_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            // =========================================================================
            // MFMA OPERATIONS - SECOND HALF (K[16:32]) accumulate on 1st half results
            // =========================================================================
            // K[16:24]
            %r00_2 = amdgpu.mfma %a0_2 * %b0_2 + %r00_1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r01_2 = amdgpu.mfma %a0_2 * %b1_2 + %r01_1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r10_2 = amdgpu.mfma %a1_2 * %b0_2 + %r10_1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r11_2 = amdgpu.mfma %a1_2 * %b1_2 + %r11_1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            // K[24:32]
            %r00 = amdgpu.mfma %a0_3 * %b0_3 + %r00_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r01 = amdgpu.mfma %a0_3 * %b1_3 + %r01_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r10 = amdgpu.mfma %a1_3 * %b0_3 + %r10_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r11 = amdgpu.mfma %a1_3 * %b1_3 + %r11_2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            scf.yield %r00, %r01, %r10, %r11 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
          }

          // =========================================================================
          // EPILOGUE: Process last iteration (K = num_blocks - 1)
          // =========================================================================
          %a0_0_last = vector.load %shared_a[%load_row, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_0_last = vector.load %shared_a[%load_row_32, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_0_last = vector.load %shared_b[%load_row, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_0_last = vector.load %shared_b[%load_row_32, %load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_1_last = vector.load %shared_a[%load_row, %load_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_1_last = vector.load %shared_a[%load_row_32, %load_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_1_last = vector.load %shared_b[%load_row, %load_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_1_last = vector.load %shared_b[%load_row_32, %load_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_2_last = vector.load %shared_a[%load_row, %load_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_2_last = vector.load %shared_a[%load_row_32, %load_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_2_last = vector.load %shared_b[%load_row, %load_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_2_last = vector.load %shared_b[%load_row_32, %load_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a0_3_last = vector.load %shared_a[%load_row, %load_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %a1_3_last = vector.load %shared_a[%load_row_32, %load_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_3_last = vector.load %shared_b[%load_row, %load_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_3_last = vector.load %shared_b[%load_row_32, %load_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %r00_0_last = amdgpu.mfma %a0_0_last * %b0_0_last + %result#0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r01_0_last = amdgpu.mfma %a0_0_last * %b1_0_last + %result#1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r10_0_last = amdgpu.mfma %a1_0_last * %b0_0_last + %result#2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r11_0_last = amdgpu.mfma %a1_0_last * %b1_0_last + %result#3 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          %r00_1_last = amdgpu.mfma %a0_1_last * %b0_1_last + %r00_0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r01_1_last = amdgpu.mfma %a0_1_last * %b1_1_last + %r01_0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r10_1_last = amdgpu.mfma %a1_1_last * %b0_1_last + %r10_0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r11_1_last = amdgpu.mfma %a1_1_last * %b1_1_last + %r11_0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          %r00_2_last = amdgpu.mfma %a0_2_last * %b0_2_last + %r00_1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r01_2_last = amdgpu.mfma %a0_2_last * %b1_2_last + %r01_1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r10_2_last = amdgpu.mfma %a1_2_last * %b0_2_last + %r10_1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r11_2_last = amdgpu.mfma %a1_2_last * %b1_2_last + %r11_1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          %r00_final = amdgpu.mfma %a0_3_last * %b0_3_last + %r00_2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r01_final = amdgpu.mfma %a0_3_last * %b1_3_last + %r01_2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r10_final = amdgpu.mfma %a1_3_last * %b0_3_last + %r10_2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r11_final = amdgpu.mfma %a1_3_last * %b1_3_last + %r11_2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          // =========================================================================
          // STORE RESULTS
          // =========================================================================

          // Truncate to f16
          %r00_f16 = arith.truncf %r00_final : vector<16xf32> to vector<16xf16>
          %r01_f16 = arith.truncf %r01_final : vector<16xf32> to vector<16xf16>
          %r10_f16 = arith.truncf %r10_final : vector<16xf32> to vector<16xf16>
          %r11_f16 = arith.truncf %r11_final : vector<16xf32> to vector<16xf16>

          // Flatten c_ptr for easier indexing
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

          // MFMA 32x32x8 layout for 64 threads:
          // Threads 0-31: column = thread_id, rows = 0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27
          // Threads 32-63: column = thread_id - 32, rows = 4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31

          %thread_col_in_tile = arith.remui %thread_id, %c32 : index  // 0-31
          %thread_row_group = arith.divui %thread_id, %c32 : index  // 0 (tid 0-31) or 1 (tid 32-63)
          %thread_row_base_store = arith.muli %thread_row_group, %c4 : index  // 0 (tid 0-31) or 4 (tid 32-63)

          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index

          // Store tile (0,0) - top-left 32x32
          %store_col_00 = arith.addi %out_col_base, %thread_col_in_tile : index  // all 32 columns in tile (tid 0-31/32-63)

          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index  // 0, 8, 16, 24

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %store_row = arith.addi %offs_token_id_base, %row_in_tile : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index  // 0, 4, 8, 12
                %elem_idx = arith.addi %elem_idx_base, %i : index  // 0-15
                %elem_val = vector.extract %r00_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col_00 : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }

          // Store tile (0,1) - top-right 32x32 (add 32 to column)
          %store_col_01 = arith.addi %store_col_00, %c32 : index

          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %store_row = arith.addi %offs_token_id_base, %row_in_tile : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index
                %elem_idx = arith.addi %elem_idx_base, %i : index
                %elem_val = vector.extract %r01_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col_01 : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }

          // Store tile (1,0) - bottom-left 32x32 (add 32 to row)
          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile_base = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %row_in_tile = arith.addi %row_in_tile_base, %c32 : index
              %store_row = arith.addi %offs_token_id_base, %row_in_tile : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index
                %elem_idx = arith.addi %elem_idx_base, %i : index
                %elem_val = vector.extract %r10_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col_00 : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }

          // Store tile (1,1) - bottom-right 32x32 (add 32 to both row and column)
          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile_base = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %row_in_tile = arith.addi %row_in_tile_base, %c32 : index
              %store_row = arith.addi %offs_token_id_base, %row_in_tile : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index
                %elem_idx = arith.addi %elem_idx_base, %i : index
                %elem_val = vector.extract %r11_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col_01 : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }

        }
        return
      }
    }
  }
  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
       // %a_ptr: memref<16384x6144xf16>,
       // %b_ptr: memref<8x32768x6144xf16>,
       // %sorted_token_ids_ptr: memref<33335xi32>,
       // %expert_ids_ptr: memref<521xi32>,
       // %num_tokens_post_padded_ptr: memref<1xi32>,
       // %c_ptr: memref<16384x2x32768xf16>
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_32x32x8::@fused_moe_kernel_32x32x8(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 32)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_32x32x8_4_waves {
    stream.executable.export public @fused_moe_kernel_32x32x8_4_waves workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_32x32x8_4_waves(
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // Constants
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c24 = arith.constant 24 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<16xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Thread ID calculation
        %thread_id_x = gpu.thread_id x upper_bound 128
        %thread_id_y = gpu.thread_id y upper_bound 2
        %thread_id_y_scaled = arith.muli %thread_id_y, %c128 : index
        %thread_id = arith.addi %thread_id_x, %thread_id_y_scaled : index

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
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index

          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // Allocate shared memory
          %c4096 = arith.constant 4096 : index
          %alloc = memref.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>>
            to memref<64x32xf16, #gpu.address_space<workgroup>>

          // Wave and tile assignment
          %wave_id = arith.divui %thread_id, %c64 : index
          %lane_id = arith.remui %thread_id, %c64 : index

          %tile_m = arith.divui %wave_id, %c2 : index
          %tile_n = arith.remui %wave_id, %c2 : index

          %tile_m_offset = arith.muli %tile_m, %c32 : index
          %tile_n_offset = arith.muli %tile_n, %c32 : index

          // Cooperative loading (NO SWIZZLING for now - to verify correctness)
          %thread_in_half = arith.remui %thread_id, %c128 : index

          %load_row_base = arith.divui %thread_in_half, %c2 : index
          %load_col_group = arith.remui %thread_in_half, %c2 : index
          %load_col_offset = arith.muli %load_col_group, %c16 : index

          // Get token IDs
          %thread_token_id = arith.addi %offs_token_id_base, %load_row_base : index
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index
          %a_row = arith.divui %token_id, %top_k : index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<16xi1>

          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %load_row_base : index

          // PROLOGUE: Load first K iteration (NO SWIZZLING)
          %k_start_0 = arith.constant 0 : index
          %k_col_start = arith.addi %k_start_0, %load_col_offset : index

          %a_row_vec_0 = vector.transfer_read %a_ptr[%a_row, %k_col_start], %f0_f16, %token_mask :
            memref<16384x6144xf16>, vector<16xf16>
          %b_row_vec_0 = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store WITHOUT swizzling
          vector.store %a_row_vec_0, %shared_a[%load_row_base, %load_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0, %shared_b[%load_row_base, %load_col_offset] :
            memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          amdgpu.lds_barrier

          // MFMA load indices
          %mfma_load_col = affine.apply #map_load_col()[%lane_id]
          %mfma_load_row = affine.apply #map_load_row()[%lane_id]

          %mfma_row_a = arith.addi %mfma_load_row, %tile_m_offset : index
          %mfma_row_b = arith.addi %mfma_load_row, %tile_n_offset : index

          %mfma_col_8 = arith.addi %mfma_load_col, %c8 : index
          %mfma_col_16 = arith.addi %mfma_load_col, %c16 : index
          %mfma_col_24 = arith.addi %mfma_load_col, %c24 : index

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // MAIN LOOP
          %result = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
              iter_args(%acc = %cst_mfma) -> (vector<16xf32>) {

            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index

            // Load from shared memory (NO SWIZZLING)
            %a0 = vector.load %shared_a[%mfma_row_a, %mfma_load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0 = vector.load %shared_b[%mfma_row_b, %mfma_load_col] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a1 = vector.load %shared_a[%mfma_row_a, %mfma_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%mfma_row_b, %mfma_col_8] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Prefetch
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index
            %k_col_start_next = arith.addi %k_start_next, %load_col_offset : index

            %a_row_vec_next = vector.transfer_read %a_ptr[%a_row, %k_col_start_next], %f0_f16, %token_mask :
              memref<16384x6144xf16>, vector<16xf16>
            %b_row_vec_next = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            %a2 = vector.load %shared_a[%mfma_row_a, %mfma_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%mfma_row_b, %mfma_col_16] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a3 = vector.load %shared_a[%mfma_row_a, %mfma_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%mfma_row_b, %mfma_col_24] :
                memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // MFMA operations
            %r0 = amdgpu.mfma %a0 * %b0 + %acc {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r1 = amdgpu.mfma %a1 * %b1 + %r0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r2 = amdgpu.mfma %a2 * %b2 + %r1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r3 = amdgpu.mfma %a3 * %b3 + %r2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            amdgpu.lds_barrier
            vector.store %a_row_vec_next, %shared_a[%load_row_base, %load_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next, %shared_b[%load_row_base, %load_col_offset] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            amdgpu.lds_barrier

            scf.yield %r3 : vector<16xf32>
          }

          // EPILOGUE
          %a0_last = vector.load %shared_a[%mfma_row_a, %mfma_load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_last = vector.load %shared_b[%mfma_row_b, %mfma_load_col] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a1_last = vector.load %shared_a[%mfma_row_a, %mfma_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%mfma_row_b, %mfma_col_8] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a2_last = vector.load %shared_a[%mfma_row_a, %mfma_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%mfma_row_b, %mfma_col_16] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a3_last = vector.load %shared_a[%mfma_row_a, %mfma_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%mfma_row_b, %mfma_col_24] :
              memref<64x32xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %r0_last = amdgpu.mfma %a0_last * %b0_last + %result {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r1_last = amdgpu.mfma %a1_last * %b1_last + %r0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r2_last = amdgpu.mfma %a2_last * %b2_last + %r1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %result_final = amdgpu.mfma %a3_last * %b3_last + %r2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          // STORE RESULTS
          %result_f16 = arith.truncf %result_final : vector<16xf32> to vector<16xf16>

          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

          %thread_col_in_tile = arith.remui %lane_id, %c32 : index
          %thread_row_group = arith.divui %lane_id, %c32 : index
          %thread_row_base_store = arith.muli %thread_row_group, %c4 : index

          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %store_col_offset = arith.addi %out_col_base, %tile_n_offset : index
          %store_col = arith.addi %store_col_offset, %thread_col_in_tile : index

          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %row_in_block = arith.addi %tile_m_offset, %row_in_tile : index
              %store_row = arith.addi %offs_token_id_base, %row_in_block : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index
                %elem_idx = arith.addi %elem_idx_base, %i : index
                %elem_val = vector.extract %result_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }
        }
        return
      }
    }
  }

  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_32x32x8_4_waves::@fused_moe_kernel_32x32x8_4_waves(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 32)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_32x32x8_4_waves_padding {
    stream.executable.export public @fused_moe_kernel_32x32x8_4_waves_padding workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_32x32x8_4_waves_padding(
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // Constants
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c24 = arith.constant 24 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<16xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Thread ID calculation
        %thread_id_x = gpu.thread_id x upper_bound 128
        %thread_id_y = gpu.thread_id y upper_bound 2
        %thread_id_y_scaled = arith.muli %thread_id_y, %c128 : index
        %thread_id = arith.addi %thread_id_x, %thread_id_y_scaled : index

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
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index

          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

          // =========================================================================
          // PADDED SHARED MEMORY LAYOUT
          // Use 64x34 with 8 columns of padding for optimal bank conflict avoidance
          // 64x34x2 = 4352 bytes per buffer
          // Total: 8704 bytes (well within 64KB LDS limit)
          // Row stride = 80 bytes = 20 banks (not a divisor of 32, good!)
          // =========================================================================
          %c40 = arith.constant 40 : index
          %c4352 = arith.constant 4352 : index
          %alloc = memref.alloc() : memref<8704xi8, #gpu.address_space<workgroup>>
          %shared_a = memref.view %alloc[%c0][] : memref<8704xi8, #gpu.address_space<workgroup>>
            to memref<64x34xf16, #gpu.address_space<workgroup>>
          %shared_b = memref.view %alloc[%c4352][] : memref<8704xi8, #gpu.address_space<workgroup>>
            to memref<64x34xf16, #gpu.address_space<workgroup>>

          // Wave and tile assignment
          %wave_id = arith.divui %thread_id, %c64 : index
          %lane_id = arith.remui %thread_id, %c64 : index

          %tile_m = arith.divui %wave_id, %c2 : index
          %tile_n = arith.remui %wave_id, %c2 : index

          %tile_m_offset = arith.muli %tile_m, %c32 : index
          %tile_n_offset = arith.muli %tile_n, %c32 : index

          // Cooperative loading
          %thread_in_half = arith.remui %thread_id, %c128 : index

          %load_row_base = arith.divui %thread_in_half, %c2 : index
          %load_col_group = arith.remui %thread_in_half, %c2 : index
          %load_col_offset = arith.muli %load_col_group, %c16 : index

          // Get token IDs
          %thread_token_id = arith.addi %offs_token_id_base, %load_row_base : index
          %token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
          %token_id = arith.index_cast %token_id_val : i32 to index
          %a_row = arith.divui %token_id, %top_k : index

          %token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
          %token_mask = vector.broadcast %token_valid : i1 to vector<16xi1>

          %offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %b_row = arith.addi %offs_bn_base, %load_row_base : index

          // PROLOGUE
          %k_start_0 = arith.constant 0 : index
          %k_col_start = arith.addi %k_start_0, %load_col_offset : index

          %a_row_vec_0 = vector.transfer_read %a_ptr[%a_row, %k_col_start], %f0_f16, %token_mask :
            memref<16384x6144xf16>, vector<16xf16>
          %b_row_vec_0 = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start], %f0_f16 :
            memref<8x32768x6144xf16>, vector<16xf16>

          // Store to padded shared memory
          vector.store %a_row_vec_0, %shared_a[%load_row_base, %load_col_offset] :
            memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
          vector.store %b_row_vec_0, %shared_b[%load_row_base, %load_col_offset] :
            memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>

          // MFMA load indices
          %mfma_load_col = affine.apply #map_load_col()[%lane_id]
          %mfma_load_row = affine.apply #map_load_row()[%lane_id]

          %mfma_row_a = arith.addi %mfma_load_row, %tile_m_offset : index
          %mfma_row_b = arith.addi %mfma_load_row, %tile_n_offset : index

          %mfma_col_8 = arith.addi %mfma_load_col, %c8 : index
          %mfma_col_16 = arith.addi %mfma_load_col, %c16 : index
          %mfma_col_24 = arith.addi %mfma_load_col, %c24 : index

          %num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
          %num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

          // MAIN LOOP
          %result = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
              iter_args(%acc = %cst_mfma) -> (vector<16xf32>) {

            %k_start = arith.muli %k_block, %BLOCK_SIZE_K : index

            amdgpu.lds_barrier

            // Load from padded shared memory
            %a0 = vector.load %shared_a[%mfma_row_a, %mfma_load_col] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b0 = vector.load %shared_b[%mfma_row_b, %mfma_load_col] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a1 = vector.load %shared_a[%mfma_row_a, %mfma_col_8] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b1 = vector.load %shared_b[%mfma_row_b, %mfma_col_8] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // Prefetch
            %k_start_next = arith.addi %k_start, %BLOCK_SIZE_K : index
            %k_col_start_next = arith.addi %k_start_next, %load_col_offset : index

            %a_row_vec_next = vector.transfer_read %a_ptr[%a_row, %k_col_start_next], %f0_f16, %token_mask :
              memref<16384x6144xf16>, vector<16xf16>
            %b_row_vec_next = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start_next], %f0_f16 :
              memref<8x32768x6144xf16>, vector<16xf16>

            %a2 = vector.load %shared_a[%mfma_row_a, %mfma_col_16] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b2 = vector.load %shared_b[%mfma_row_b, %mfma_col_16] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            %a3 = vector.load %shared_a[%mfma_row_a, %mfma_col_24] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
            %b3 = vector.load %shared_b[%mfma_row_b, %mfma_col_24] :
                memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

            // MFMA operations
            %r0 = amdgpu.mfma %a0 * %b0 + %acc {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r1 = amdgpu.mfma %a1 * %b1 + %r0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            %r2 = amdgpu.mfma %a2 * %b2 + %r1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
            %r3 = amdgpu.mfma %a3 * %b3 + %r2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

            vector.store %a_row_vec_next, %shared_a[%load_row_base, %load_col_offset] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
            vector.store %b_row_vec_next, %shared_b[%load_row_base, %load_col_offset] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>

            amdgpu.lds_barrier

            scf.yield %r3 : vector<16xf32>
          }

          amdgpu.lds_barrier

          // EPILOGUE
          %a0_last = vector.load %shared_a[%mfma_row_a, %mfma_load_col] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b0_last = vector.load %shared_b[%mfma_row_b, %mfma_load_col] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a1_last = vector.load %shared_a[%mfma_row_a, %mfma_col_8] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b1_last = vector.load %shared_b[%mfma_row_b, %mfma_col_8] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a2_last = vector.load %shared_a[%mfma_row_a, %mfma_col_16] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b2_last = vector.load %shared_b[%mfma_row_b, %mfma_col_16] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %a3_last = vector.load %shared_a[%mfma_row_a, %mfma_col_24] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %b3_last = vector.load %shared_b[%mfma_row_b, %mfma_col_24] :
              memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>

          %r0_last = amdgpu.mfma %a0_last * %b0_last + %result {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r1_last = amdgpu.mfma %a1_last * %b1_last + %r0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %r2_last = amdgpu.mfma %a2_last * %b2_last + %r1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
          %result_final = amdgpu.mfma %a3_last * %b3_last + %r2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          // =========================================================================
          // VECTORIZED STORE USING vector.scatter
          // =========================================================================
          
          %result_f16 = arith.truncf %result_final : vector<16xf32> to vector<16xf16>

          // Flatten output memref
          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

          %thread_col_in_tile = arith.remui %lane_id, %c32 : index
          %thread_row_group = arith.divui %lane_id, %c32 : index
          %thread_row_base_store = arith.muli %thread_row_group, %c4 : index

          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %store_col_offset = arith.addi %out_col_base, %tile_n_offset : index
          %store_col = arith.addi %store_col_offset, %thread_col_in_tile : index

          // Build index vector for scatter
          %c0_idx = arith.constant 0 : index
          %c1_idx = arith.constant 1 : index
          %c2_idx = arith.constant 2 : index
          %c3_idx = arith.constant 3 : index
          %c4_idx = arith.constant 4 : index
          %c5_idx = arith.constant 5 : index
          %c6_idx = arith.constant 6 : index
          %c7_idx = arith.constant 7 : index
          %c8_idx = arith.constant 8 : index
          %c9_idx = arith.constant 9 : index
          %c10_idx = arith.constant 10 : index
          %c11_idx = arith.constant 11 : index
          %c12_idx = arith.constant 12 : index
          %c13_idx = arith.constant 13 : index
          %c14_idx = arith.constant 14 : index
          %c15_idx = arith.constant 15 : index

          // Compute row offsets for all 16 elements
          %row0_offset = arith.muli %c0_idx, %c8 : index
          %row0_in_tile = arith.addi %thread_row_base_store, %row0_offset : index
          %row0_in_block = arith.addi %tile_m_offset, %row0_in_tile : index
          %store_row0 = arith.addi %offs_token_id_base, %row0_in_block : index

          %row1_offset = arith.addi %row0_offset, %c1 : index
          %row1_in_tile = arith.addi %thread_row_base_store, %row1_offset : index
          %row1_in_block = arith.addi %tile_m_offset, %row1_in_tile : index
          %store_row1 = arith.addi %offs_token_id_base, %row1_in_block : index

          %row2_offset = arith.addi %row0_offset, %c2 : index
          %row2_in_tile = arith.addi %thread_row_base_store, %row2_offset : index
          %row2_in_block = arith.addi %tile_m_offset, %row2_in_tile : index
          %store_row2 = arith.addi %offs_token_id_base, %row2_in_block : index

          %row3_offset = arith.addi %row0_offset, %c3 : index
          %row3_in_tile = arith.addi %thread_row_base_store, %row3_offset : index
          %row3_in_block = arith.addi %tile_m_offset, %row3_in_tile : index
          %store_row3 = arith.addi %offs_token_id_base, %row3_in_block : index

          %row4_offset = arith.muli %c1_idx, %c8 : index
          %row4_in_tile = arith.addi %thread_row_base_store, %row4_offset : index
          %row4_in_block = arith.addi %tile_m_offset, %row4_in_tile : index
          %store_row4 = arith.addi %offs_token_id_base, %row4_in_block : index

          %row5_offset = arith.addi %row4_offset, %c1 : index
          %row5_in_tile = arith.addi %thread_row_base_store, %row5_offset : index
          %row5_in_block = arith.addi %tile_m_offset, %row5_in_tile : index
          %store_row5 = arith.addi %offs_token_id_base, %row5_in_block : index

          %row6_offset = arith.addi %row4_offset, %c2 : index
          %row6_in_tile = arith.addi %thread_row_base_store, %row6_offset : index
          %row6_in_block = arith.addi %tile_m_offset, %row6_in_tile : index
          %store_row6 = arith.addi %offs_token_id_base, %row6_in_block : index

          %row7_offset = arith.addi %row4_offset, %c3 : index
          %row7_in_tile = arith.addi %thread_row_base_store, %row7_offset : index
          %row7_in_block = arith.addi %tile_m_offset, %row7_in_tile : index
          %store_row7 = arith.addi %offs_token_id_base, %row7_in_block : index

          %row8_offset = arith.muli %c2_idx, %c8 : index
          %row8_in_tile = arith.addi %thread_row_base_store, %row8_offset : index
          %row8_in_block = arith.addi %tile_m_offset, %row8_in_tile : index
          %store_row8 = arith.addi %offs_token_id_base, %row8_in_block : index

          %row9_offset = arith.addi %row8_offset, %c1 : index
          %row9_in_tile = arith.addi %thread_row_base_store, %row9_offset : index
          %row9_in_block = arith.addi %tile_m_offset, %row9_in_tile : index
          %store_row9 = arith.addi %offs_token_id_base, %row9_in_block : index

          %row10_offset = arith.addi %row8_offset, %c2 : index
          %row10_in_tile = arith.addi %thread_row_base_store, %row10_offset : index
          %row10_in_block = arith.addi %tile_m_offset, %row10_in_tile : index
          %store_row10 = arith.addi %offs_token_id_base, %row10_in_block : index

          %row11_offset = arith.addi %row8_offset, %c3 : index
          %row11_in_tile = arith.addi %thread_row_base_store, %row11_offset : index
          %row11_in_block = arith.addi %tile_m_offset, %row11_in_tile : index
          %store_row11 = arith.addi %offs_token_id_base, %row11_in_block : index

          %row12_offset = arith.muli %c3_idx, %c8 : index
          %row12_in_tile = arith.addi %thread_row_base_store, %row12_offset : index
          %row12_in_block = arith.addi %tile_m_offset, %row12_in_tile : index
          %store_row12 = arith.addi %offs_token_id_base, %row12_in_block : index

          %row13_offset = arith.addi %row12_offset, %c1 : index
          %row13_in_tile = arith.addi %thread_row_base_store, %row13_offset : index
          %row13_in_block = arith.addi %tile_m_offset, %row13_in_tile : index
          %store_row13 = arith.addi %offs_token_id_base, %row13_in_block : index

          %row14_offset = arith.addi %row12_offset, %c2 : index
          %row14_in_tile = arith.addi %thread_row_base_store, %row14_offset : index
          %row14_in_block = arith.addi %tile_m_offset, %row14_in_tile : index
          %store_row14 = arith.addi %offs_token_id_base, %row14_in_block : index

          %row15_offset = arith.addi %row12_offset, %c3 : index
          %row15_in_tile = arith.addi %thread_row_base_store, %row15_offset : index
          %row15_in_block = arith.addi %tile_m_offset, %row15_in_tile : index
          %store_row15 = arith.addi %offs_token_id_base, %row15_in_block : index

          // Load token IDs
          %tok_id0_i32 = memref.load %sorted_token_ids_ptr[%store_row0] : memref<33335xi32>
          %tok_id1_i32 = memref.load %sorted_token_ids_ptr[%store_row1] : memref<33335xi32>
          %tok_id2_i32 = memref.load %sorted_token_ids_ptr[%store_row2] : memref<33335xi32>
          %tok_id3_i32 = memref.load %sorted_token_ids_ptr[%store_row3] : memref<33335xi32>
          %tok_id4_i32 = memref.load %sorted_token_ids_ptr[%store_row4] : memref<33335xi32>
          %tok_id5_i32 = memref.load %sorted_token_ids_ptr[%store_row5] : memref<33335xi32>
          %tok_id6_i32 = memref.load %sorted_token_ids_ptr[%store_row6] : memref<33335xi32>
          %tok_id7_i32 = memref.load %sorted_token_ids_ptr[%store_row7] : memref<33335xi32>
          %tok_id8_i32 = memref.load %sorted_token_ids_ptr[%store_row8] : memref<33335xi32>
          %tok_id9_i32 = memref.load %sorted_token_ids_ptr[%store_row9] : memref<33335xi32>
          %tok_id10_i32 = memref.load %sorted_token_ids_ptr[%store_row10] : memref<33335xi32>
          %tok_id11_i32 = memref.load %sorted_token_ids_ptr[%store_row11] : memref<33335xi32>
          %tok_id12_i32 = memref.load %sorted_token_ids_ptr[%store_row12] : memref<33335xi32>
          %tok_id13_i32 = memref.load %sorted_token_ids_ptr[%store_row13] : memref<33335xi32>
          %tok_id14_i32 = memref.load %sorted_token_ids_ptr[%store_row14] : memref<33335xi32>
          %tok_id15_i32 = memref.load %sorted_token_ids_ptr[%store_row15] : memref<33335xi32>

          %tok_id0 = arith.index_cast %tok_id0_i32 : i32 to index
          %tok_id1 = arith.index_cast %tok_id1_i32 : i32 to index
          %tok_id2 = arith.index_cast %tok_id2_i32 : i32 to index
          %tok_id3 = arith.index_cast %tok_id3_i32 : i32 to index
          %tok_id4 = arith.index_cast %tok_id4_i32 : i32 to index
          %tok_id5 = arith.index_cast %tok_id5_i32 : i32 to index
          %tok_id6 = arith.index_cast %tok_id6_i32 : i32 to index
          %tok_id7 = arith.index_cast %tok_id7_i32 : i32 to index
          %tok_id8 = arith.index_cast %tok_id8_i32 : i32 to index
          %tok_id9 = arith.index_cast %tok_id9_i32 : i32 to index
          %tok_id10 = arith.index_cast %tok_id10_i32 : i32 to index
          %tok_id11 = arith.index_cast %tok_id11_i32 : i32 to index
          %tok_id12 = arith.index_cast %tok_id12_i32 : i32 to index
          %tok_id13 = arith.index_cast %tok_id13_i32 : i32 to index
          %tok_id14 = arith.index_cast %tok_id14_i32 : i32 to index
          %tok_id15 = arith.index_cast %tok_id15_i32 : i32 to index

          // Check validity
          %valid0 = arith.cmpi slt, %tok_id0, %num_valid_tokens : index
          %valid1 = arith.cmpi slt, %tok_id1, %num_valid_tokens : index
          %valid2 = arith.cmpi slt, %tok_id2, %num_valid_tokens : index
          %valid3 = arith.cmpi slt, %tok_id3, %num_valid_tokens : index
          %valid4 = arith.cmpi slt, %tok_id4, %num_valid_tokens : index
          %valid5 = arith.cmpi slt, %tok_id5, %num_valid_tokens : index
          %valid6 = arith.cmpi slt, %tok_id6, %num_valid_tokens : index
          %valid7 = arith.cmpi slt, %tok_id7, %num_valid_tokens : index
          %valid8 = arith.cmpi slt, %tok_id8, %num_valid_tokens : index
          %valid9 = arith.cmpi slt, %tok_id9, %num_valid_tokens : index
          %valid10 = arith.cmpi slt, %tok_id10, %num_valid_tokens : index
          %valid11 = arith.cmpi slt, %tok_id11, %num_valid_tokens : index
          %valid12 = arith.cmpi slt, %tok_id12, %num_valid_tokens : index
          %valid13 = arith.cmpi slt, %tok_id13, %num_valid_tokens : index
          %valid14 = arith.cmpi slt, %tok_id14, %num_valid_tokens : index
          %valid15 = arith.cmpi slt, %tok_id15, %num_valid_tokens : index

          // Compute output indices
          %out_row_base0 = arith.muli %tok_id0, %N : index
          %out_row_base1 = arith.muli %tok_id1, %N : index
          %out_row_base2 = arith.muli %tok_id2, %N : index
          %out_row_base3 = arith.muli %tok_id3, %N : index
          %out_row_base4 = arith.muli %tok_id4, %N : index
          %out_row_base5 = arith.muli %tok_id5, %N : index
          %out_row_base6 = arith.muli %tok_id6, %N : index
          %out_row_base7 = arith.muli %tok_id7, %N : index
          %out_row_base8 = arith.muli %tok_id8, %N : index
          %out_row_base9 = arith.muli %tok_id9, %N : index
          %out_row_base10 = arith.muli %tok_id10, %N : index
          %out_row_base11 = arith.muli %tok_id11, %N : index
          %out_row_base12 = arith.muli %tok_id12, %N : index
          %out_row_base13 = arith.muli %tok_id13, %N : index
          %out_row_base14 = arith.muli %tok_id14, %N : index
          %out_row_base15 = arith.muli %tok_id15, %N : index

          %out_idx0 = arith.addi %out_row_base0, %store_col : index
          %out_idx1 = arith.addi %out_row_base1, %store_col : index
          %out_idx2 = arith.addi %out_row_base2, %store_col : index
          %out_idx3 = arith.addi %out_row_base3, %store_col : index
          %out_idx4 = arith.addi %out_row_base4, %store_col : index
          %out_idx5 = arith.addi %out_row_base5, %store_col : index
          %out_idx6 = arith.addi %out_row_base6, %store_col : index
          %out_idx7 = arith.addi %out_row_base7, %store_col : index
          %out_idx8 = arith.addi %out_row_base8, %store_col : index
          %out_idx9 = arith.addi %out_row_base9, %store_col : index
          %out_idx10 = arith.addi %out_row_base10, %store_col : index
          %out_idx11 = arith.addi %out_row_base11, %store_col : index
          %out_idx12 = arith.addi %out_row_base12, %store_col : index
          %out_idx13 = arith.addi %out_row_base13, %store_col : index
          %out_idx14 = arith.addi %out_row_base14, %store_col : index
          %out_idx15 = arith.addi %out_row_base15, %store_col : index

          // Build index vector
          %indices_undef = llvm.mlir.undef : vector<16xindex>
          %indices_0 = vector.insert %out_idx0, %indices_undef[0] : index into vector<16xindex>
          %indices_1 = vector.insert %out_idx1, %indices_0[1] : index into vector<16xindex>
          %indices_2 = vector.insert %out_idx2, %indices_1[2] : index into vector<16xindex>
          %indices_3 = vector.insert %out_idx3, %indices_2[3] : index into vector<16xindex>
          %indices_4 = vector.insert %out_idx4, %indices_3[4] : index into vector<16xindex>
          %indices_5 = vector.insert %out_idx5, %indices_4[5] : index into vector<16xindex>
          %indices_6 = vector.insert %out_idx6, %indices_5[6] : index into vector<16xindex>
          %indices_7 = vector.insert %out_idx7, %indices_6[7] : index into vector<16xindex>
          %indices_8 = vector.insert %out_idx8, %indices_7[8] : index into vector<16xindex>
          %indices_9 = vector.insert %out_idx9, %indices_8[9] : index into vector<16xindex>
          %indices_10 = vector.insert %out_idx10, %indices_9[10] : index into vector<16xindex>
          %indices_11 = vector.insert %out_idx11, %indices_10[11] : index into vector<16xindex>
          %indices_12 = vector.insert %out_idx12, %indices_11[12] : index into vector<16xindex>
          %indices_13 = vector.insert %out_idx13, %indices_12[13] : index into vector<16xindex>
          %indices_14 = vector.insert %out_idx14, %indices_13[14] : index into vector<16xindex>
          %indices_15 = vector.insert %out_idx15, %indices_14[15] : index into vector<16xindex>

          // Build mask vector
          %mask_undef = llvm.mlir.undef : vector<16xi1>
          %mask_0 = vector.insert %valid0, %mask_undef[0] : i1 into vector<16xi1>
          %mask_1 = vector.insert %valid1, %mask_0[1] : i1 into vector<16xi1>
          %mask_2 = vector.insert %valid2, %mask_1[2] : i1 into vector<16xi1>
          %mask_3 = vector.insert %valid3, %mask_2[3] : i1 into vector<16xi1>
          %mask_4 = vector.insert %valid4, %mask_3[4] : i1 into vector<16xi1>
          %mask_5 = vector.insert %valid5, %mask_4[5] : i1 into vector<16xi1>
          %mask_6 = vector.insert %valid6, %mask_5[6] : i1 into vector<16xi1>
          %mask_7 = vector.insert %valid7, %mask_6[7] : i1 into vector<16xi1>
          %mask_8 = vector.insert %valid8, %mask_7[8] : i1 into vector<16xi1>
          %mask_9 = vector.insert %valid9, %mask_8[9] : i1 into vector<16xi1>
          %mask_10 = vector.insert %valid10, %mask_9[10] : i1 into vector<16xi1>
          %mask_11 = vector.insert %valid11, %mask_10[11] : i1 into vector<16xi1>
          %mask_12 = vector.insert %valid12, %mask_11[12] : i1 into vector<16xi1>
          %mask_13 = vector.insert %valid13, %mask_12[13] : i1 into vector<16xi1>
          %mask_14 = vector.insert %valid14, %mask_13[14] : i1 into vector<16xi1>
          %mask_15 = vector.insert %valid15, %mask_14[15] : i1 into vector<16xi1>

          // SCATTER STORE - This is the key operation!
          %base_idx = arith.constant 0 : index
          vector.scatter %c_flat[%base_idx][%indices_15], %mask_15, %result_f16 
            : memref<1073741824xf16>, vector<16xindex>, vector<16xi1>, vector<16xf16>
        }
        return
      }
    }
  }

  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_32x32x8_4_waves_padding::@fused_moe_kernel_32x32x8_4_waves_padding(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}
    """
)

asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering = (
    """
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>

#map_load_row = affine_map<()[s0] -> (s0 mod 32)>
#map_load_col = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 4)>

module attributes {transform.with_named_sequence} {
  stream.executable private @fused_moe_kernel_32x32x8_4_waves_padding_double_buffering {
    stream.executable.export public @fused_moe_kernel_32x32x8_4_waves_padding_double_buffering workgroups() -> (index, index, index) {
      %c266752 = arith.constant 266752 : index
      %c1 = arith.constant 1 : index
      stream.return %c266752, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @fused_moe_kernel_32x32x8_4_waves_padding_double_buffering(
          %arg0: !stream.binding,
          %arg1: !stream.binding,
          %arg2: !stream.binding,
          %arg3: !stream.binding,
          %arg4: !stream.binding,
          %arg5: !stream.binding
      ) attributes {translation_info = #translation} {
        // Constants
        %N = arith.constant 32768 : index
        %K = arith.constant 6144 : index
        %EM = arith.constant 33335 : index
        %top_k = arith.constant 2 : index
        %num_valid_tokens = arith.constant 32768 : index
        %GROUP_SIZE_M = arith.constant 8 : index
        %BLOCK_SIZE_M = arith.constant 64 : index
        %BLOCK_SIZE_N = arith.constant 64 : index
        %BLOCK_SIZE_K = arith.constant 32 : index

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c24 = arith.constant 24 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %f0 = arith.constant 0.0 : f32
        %f0_f16 = arith.constant 0.0 : f16
        %cst_mfma = arith.constant dense<0.000000e+00> : vector<16xf32>

        %a_ptr = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16384x6144xf16>
        %b_ptr = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x32768x6144xf16>
        %c_ptr = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16384x2x32768xf16>
        %sorted_token_ids_ptr = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<33335xi32>
        %expert_ids_ptr = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<521xi32>
        %num_tokens_post_padded_ptr = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<1xi32>

        // Thread ID calculation
        %thread_id_x = gpu.thread_id x upper_bound 128
        %thread_id_y = gpu.thread_id y upper_bound 2
        %thread_id_y_scaled = arith.muli %thread_id_y, %c128 : index
        %thread_id = arith.addi %thread_id_x, %thread_id_y_scaled : index

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
          %offs_token_id_base = arith.muli %pid_m, %BLOCK_SIZE_M : index

          %expert_id_val = memref.load %expert_ids_ptr[%pid_m] : memref<521xi32>
          %expert_id = arith.index_cast %expert_id_val : i32 to index

// =========================================================================
// DOUBLE BUFFERED SHARED MEMORY LAYOUT
// Buffer 0: offset 0
// Buffer 1: offset 4352 (64 * 34 * 2 bytes)
// Total per matrix: 8704 bytes
// Total: 17408 bytes (well within 64KB LDS limit)
// =========================================================================
%c34 = arith.constant 34 : index
%c4352 = arith.constant 4352 : index  // 64 * 34 * 2 bytes
%c8704 = arith.constant 8704 : index  // 2 * 4352
%c13056 = arith.constant 13056 : index // 3 * 4352

%alloc = memref.alloc() : memref<17408xi8, #gpu.address_space<workgroup>>

// A buffers (ping-pong)
%shared_a_0 = memref.view %alloc[%c0][] : memref<17408xi8, #gpu.address_space<workgroup>>
  to memref<64x34xf16, #gpu.address_space<workgroup>>
%shared_a_1 = memref.view %alloc[%c4352][] : memref<17408xi8, #gpu.address_space<workgroup>>
  to memref<64x34xf16, #gpu.address_space<workgroup>>

// B buffers (ping-pong)
%shared_b_0 = memref.view %alloc[%c8704][] : memref<17408xi8, #gpu.address_space<workgroup>>
  to memref<64x34xf16, #gpu.address_space<workgroup>>
%shared_b_1 = memref.view %alloc[%c13056][] : memref<17408xi8, #gpu.address_space<workgroup>>
  to memref<64x34xf16, #gpu.address_space<workgroup>>

// Wave and tile assignment (same as before)
%wave_id = arith.divui %thread_id, %c64 : index
%lane_id = arith.remui %thread_id, %c64 : index

%tile_m = arith.divui %wave_id, %c2 : index
%tile_n = arith.remui %wave_id, %c2 : index

%tile_m_offset = arith.muli %tile_m, %c32 : index
%tile_n_offset = arith.muli %tile_n, %c32 : index

// Cooperative loading
%thread_in_half = arith.remui %thread_id, %c128 : index
%load_row_base = arith.divui %thread_in_half, %c2 : index
%load_col_group = arith.remui %thread_in_half, %c2 : index
%load_col_offset = arith.muli %load_col_group, %c16 : index

// Get token IDs
%thread_token_id = arith.addi %offs_token_id_base, %load_row_base : index
%token_id_val = memref.load %sorted_token_ids_ptr[%thread_token_id] : memref<33335xi32>
%token_id = arith.index_cast %token_id_val : i32 to index
%a_row = arith.divui %token_id, %top_k : index

%token_valid = arith.cmpi slt, %token_id, %num_valid_tokens : index
%token_mask = vector.broadcast %token_valid : i1 to vector<16xi1>

%offs_bn_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
%b_row = arith.addi %offs_bn_base, %load_row_base : index

// MFMA load indices
%mfma_load_col = affine.apply #map_load_col()[%lane_id]
%mfma_load_row = affine.apply #map_load_row()[%lane_id]

%mfma_row_a = arith.addi %mfma_load_row, %tile_m_offset : index
%mfma_row_b = arith.addi %mfma_load_row, %tile_n_offset : index

%mfma_col_8 = arith.addi %mfma_load_col, %c8 : index
%mfma_col_16 = arith.addi %mfma_load_col, %c16 : index
%mfma_col_24 = arith.addi %mfma_load_col, %c24 : index

%num_blocks = arith.ceildivui %K, %BLOCK_SIZE_K : index
%num_blocks_minus_1 = arith.subi %num_blocks, %c1 : index

// =========================================================================
// PROLOGUE: Load iteration 0 into buffer 0
// =========================================================================
%k_start_0 = arith.constant 0 : index
%k_col_start_0 = arith.addi %k_start_0, %load_col_offset : index

%a_row_vec_0 = vector.transfer_read %a_ptr[%a_row, %k_col_start_0], %f0_f16, %token_mask :
  memref<16384x6144xf16>, vector<16xf16>
%b_row_vec_0 = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start_0], %f0_f16 :
  memref<8x32768x6144xf16>, vector<16xf16>

// Store to buffer 0
vector.store %a_row_vec_0, %shared_a_0[%load_row_base, %load_col_offset] :
  memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
vector.store %b_row_vec_0, %shared_b_0[%load_row_base, %load_col_offset] :
  memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>

amdgpu.lds_barrier

// =========================================================================
// MAIN LOOP with DOUBLE BUFFERING
// =========================================================================
%loop_result = scf.for %k_block = %c0 to %num_blocks_minus_1 step %c1
    iter_args(%acc = %cst_mfma) -> (vector<16xf32>) {

  // Determine which buffer to read from (current iteration)
  %read_buffer_idx = arith.remui %k_block, %c2 : index
  %is_buffer_0 = arith.cmpi eq, %read_buffer_idx, %c0 : index

  // Determine which buffer to write to (next iteration)
  %k_block_plus_1 = arith.addi %k_block, %c1 : index
  %write_buffer_idx = arith.remui %k_block_plus_1, %c2 : index
  %write_to_buffer_0 = arith.cmpi eq, %write_buffer_idx, %c0 : index

  // -----------------------------------------------------------------------
  // LOAD from current buffer and execute MFMA
  // -----------------------------------------------------------------------

  %a0 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %a = vector.load %shared_a_0[%mfma_row_a, %mfma_load_col] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  } else {
    %a = vector.load %shared_a_1[%mfma_row_a, %mfma_load_col] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  }

  %b0 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %b = vector.load %shared_b_0[%mfma_row_b, %mfma_load_col] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  } else {
    %b = vector.load %shared_b_1[%mfma_row_b, %mfma_load_col] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  }

  %a1 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_8] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  } else {
    %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_8] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  }

  %b1 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_8] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  } else {
    %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_8] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  }

  // -----------------------------------------------------------------------
  // PREFETCH: Load next iteration from global memory
  // -----------------------------------------------------------------------
  %k_start_next = arith.addi %k_block, %c1 : index
  %k_start_next_scaled = arith.muli %k_start_next, %BLOCK_SIZE_K : index
  %k_col_start_next = arith.addi %k_start_next_scaled, %load_col_offset : index

  %a_row_vec_next = vector.transfer_read %a_ptr[%a_row, %k_col_start_next], %f0_f16, %token_mask :
    memref<16384x6144xf16>, vector<16xf16>
  %b_row_vec_next = vector.transfer_read %b_ptr[%expert_id, %b_row, %k_col_start_next], %f0_f16 :
    memref<8x32768x6144xf16>, vector<16xf16>

  // Continue loading from current buffer
  %a2 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_16] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  } else {
    %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_16] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  }

  %b2 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_16] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  } else {
    %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_16] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  }

  %a3 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_24] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  } else {
    %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_24] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %a : vector<4xf16>
  }

  %b3 = scf.if %is_buffer_0 -> (vector<4xf16>) {
    %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_24] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  } else {
    %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_24] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    scf.yield %b : vector<4xf16>
  }

  // -----------------------------------------------------------------------
  // MFMA OPERATIONS
  // -----------------------------------------------------------------------
  %r0 = amdgpu.mfma %a0 * %b0 + %acc {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
  %r1 = amdgpu.mfma %a1 * %b1 + %r0 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
  %r2 = amdgpu.mfma %a2 * %b2 + %r1 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
  %r3 = amdgpu.mfma %a3 * %b3 + %r2 {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

  // -----------------------------------------------------------------------
  // STORE: Write prefetched data to next buffer
  // -----------------------------------------------------------------------
  amdgpu.lds_barrier  // Wait for MFMA to finish reading current buffer

  scf.if %write_to_buffer_0 {
    vector.store %a_row_vec_next, %shared_a_0[%load_row_base, %load_col_offset] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
    vector.store %b_row_vec_next, %shared_b_0[%load_row_base, %load_col_offset] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  } else {
    vector.store %a_row_vec_next, %shared_a_1[%load_row_base, %load_col_offset] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
    vector.store %b_row_vec_next, %shared_b_1[%load_row_base, %load_col_offset] :
      memref<64x34xf16, #gpu.address_space<workgroup>>, vector<16xf16>
  }

  amdgpu.lds_barrier  // Wait for stores to complete before next iteration reads

  scf.yield %r3 : vector<16xf32>
}

// =========================================================================
// EPILOGUE: Process last iteration
// =========================================================================
amdgpu.lds_barrier

%last_buffer_idx = arith.remui %num_blocks_minus_1, %c2 : index
%use_buffer_0_last = arith.cmpi eq, %last_buffer_idx, %c0 : index

%a0_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %a = vector.load %shared_a_0[%mfma_row_a, %mfma_load_col] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
} else {
  %a = vector.load %shared_a_1[%mfma_row_a, %mfma_load_col] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
}

%b0_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %b = vector.load %shared_b_0[%mfma_row_b, %mfma_load_col] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
} else {
  %b = vector.load %shared_b_1[%mfma_row_b, %mfma_load_col] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
}

%a1_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_8] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
} else {
  %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_8] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
}

%b1_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_8] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
} else {
  %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_8] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
}

%a2_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_16] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
} else {
  %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_16] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
}

%b2_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_16] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
} else {
  %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_16] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
}

%a3_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %a = vector.load %shared_a_0[%mfma_row_a, %mfma_col_24] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
} else {
  %a = vector.load %shared_a_1[%mfma_row_a, %mfma_col_24] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %a : vector<4xf16>
}

%b3_last = scf.if %use_buffer_0_last -> (vector<4xf16>) {
  %b = vector.load %shared_b_0[%mfma_row_b, %mfma_col_24] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
} else {
  %b = vector.load %shared_b_1[%mfma_row_b, %mfma_col_24] :
    memref<64x34xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  scf.yield %b : vector<4xf16>
}

%r0_last = amdgpu.mfma %a0_last * %b0_last + %loop_result {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
%r1_last = amdgpu.mfma %a1_last * %b1_last + %r0_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
%r2_last = amdgpu.mfma %a2_last * %b2_last + %r1_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
%result_final = amdgpu.mfma %a3_last * %b3_last + %r2_last {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>

          // STORE RESULTS
          %result_f16 = arith.truncf %result_final : vector<16xf32> to vector<16xf16>

          %c_flat = memref.collapse_shape %c_ptr [[0, 1, 2]] : memref<16384x2x32768xf16> into memref<1073741824xf16>

          %thread_col_in_tile = arith.remui %lane_id, %c32 : index
          %thread_row_group = arith.divui %lane_id, %c32 : index
          %thread_row_base_store = arith.muli %thread_row_group, %c4 : index

          %out_col_base = arith.muli %pid_n, %BLOCK_SIZE_N : index
          %store_col_offset = arith.addi %out_col_base, %tile_n_offset : index
          %store_col = arith.addi %store_col_offset, %thread_col_in_tile : index

          scf.for %group = %c0 to %c4 step %c1 {
            %group_base = arith.muli %group, %c8 : index

            scf.for %i = %c0 to %c4 step %c1 {
              %row_offset_in_group = arith.addi %group_base, %i : index
              %row_in_tile = arith.addi %thread_row_base_store, %row_offset_in_group : index
              %row_in_block = arith.addi %tile_m_offset, %row_in_tile : index
              %store_row = arith.addi %offs_token_id_base, %row_in_block : index

              %tok_id_i32 = memref.load %sorted_token_ids_ptr[%store_row] : memref<33335xi32>
              %tok_id = arith.index_cast %tok_id_i32 : i32 to index
              %out_valid = arith.cmpi slt, %tok_id, %num_valid_tokens : index

              scf.if %out_valid {
                %elem_idx_base = arith.muli %group, %c4 : index
                %elem_idx = arith.addi %elem_idx_base, %i : index
                %elem_val = vector.extract %result_f16[%elem_idx] : f16 from vector<16xf16>

                %out_row_base = arith.muli %tok_id, %N : index
                %out_idx = arith.addi %out_row_base, %store_col : index
                memref.store %elem_val, %c_flat[%out_idx] : memref<1073741824xf16>
              }
              scf.yield
            }
            scf.yield
          }
        }
        return
      }
    }
  }

  func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view {
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<16384x6144xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<8x32768x6144xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<33335xi32>
    %3 = hal.tensor.import wait(%arg6) => %arg3 : !hal.buffer_view -> tensor<521xi32>
    %4 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<1xi32>
    %5 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<16384x2x32768xf16>
    %6 = flow.dispatch @fused_moe_kernel_32x32x8_4_waves_padding_double_buffering::@fused_moe_kernel_32x32x8_4_waves_padding_double_buffering(%0, %1, %2, %3, %4, %5) : (tensor<16384x6144xf16>, tensor<8x32768x6144xf16>, tensor<33335xi32>, tensor<521xi32>, tensor<1xi32>, tensor<16384x2x32768xf16>) -> %5
    %7 = hal.tensor.barrier join(%6 : tensor<16384x2x32768xf16>) => %arg7 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<16384x2x32768xf16> -> !hal.buffer_view
    return %8 : !hal.buffer_view
  }
}

    """
)

def compare_once(M, N, K):
    topk = 2
    num_experts = 8
    block_m, block_n, block_k = 64, 64, 32
    dtype = torch.float16
    device = torch.device("cuda")

    scores = torch.rand(M, num_experts, device=device)
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    sorted_ids, expert_ids, num_tokens_post_pad, max_num_tokens_padded, max_num_m_blocks = build_sorted_ids_and_expert_blocks(
        topk_ids, num_experts, block_m
    )

    #  a = torch.rand((num_tokens, k), dtype=dtype, device='cuda')
    # b = torch.rand((num_experts, n, k), dtype=dtype, device='cuda')
    # c = torch.zeros(num_tokens, topk, n, dtype=dtype, device='cuda')

    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((num_experts, N, K), dtype=dtype, device=device)

    ref_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    for expert_idx in range(M):
        mask = topk_ids == expert_idx
        token_indices, topk_indices = torch.where(mask)

        if len(token_indices) > 0:
            tokens = a[token_indices]
            expert_out = torch.matmul(tokens, b[expert_idx].T)

            ref_c[token_indices, topk_indices] = expert_out

    rtol, atol = 1e-1, 1e-2

    def verify(res):
        for i in range(M):
            num = N // 16384
            for j in range(num):
                k = j * 16384
                torch.testing.assert_close(
                    ref_c[i][0][k:k+16384],
                    res[i][0][k:k+16384],
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    ref_c[i][1][k:k+16384],
                    res[i][1][k:k+16384],
                    rtol=rtol,
                    atol=atol,
                )

    c_tri = torch.zeros(M, topk, N, dtype=dtype, device="cuda")
    moe_gemm_triton(a, b, c_tri, sorted_ids, expert_ids, num_tokens_post_pad, topk,
                    block_m=block_m, block_n=block_n, block_k=block_k, group_m=8)

    verify(c_tri)

    gemm_kernel, symbols = get_moe_gemm_kernel(
        M,
        topk,
        block_m,
        num_experts,
        K,
        N,
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
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16 = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64 = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96,
    )
    options = set_default_run_config(options)

    mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96 = wave_compile(options, gemm_kernel)
    gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68,
    )
    options = set_default_run_config(options)

    mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68 = wave_compile(options, gemm_kernel)
    gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding,
    )
    options = set_default_run_config(options)

  # mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding = wave_compile(options, gemm_kernel)
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
  # verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8,
    )
    options = set_default_run_config(options)

  # mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8 = wave_compile(options, gemm_kernel)
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
  # verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves,
    )
    options = set_default_run_config(options)

  # mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves = wave_compile(options, gemm_kernel)
  # gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
  # verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
 #      mlir_print_ir_after_all=True,
        override_mlir=asm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering,
    )
    options = set_default_run_config(options)

 #  mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering = wave_compile(options, gemm_kernel)
 #  gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
 #  verify(mlir_c)

    num_warmups = 3
    for i in range(1, num_warmups + 1):
        print(f"WARMUP {i}/{num_warmups}")
        c_tri = torch.zeros(M, topk, N, dtype=dtype, device="cuda")
        moe_gemm_triton(a, b, c_tri, sorted_ids, expert_ids, num_tokens_post_pad, topk,
                        block_m=block_m, block_n=block_n, block_k=block_k, group_m=8)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
        mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
        gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
        mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
        gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)

    num_measurements = 20
    for i in range(1, num_measurements + 1):
        print(f"MEASUREMENT {i}/{num_measurements}")
        c_tri = torch.zeros(M, topk, N, dtype=dtype, device="cuda")
        moe_gemm_triton(a, b, c_tri, sorted_ids, expert_ids, num_tokens_post_pad, topk,
                        block_m=block_m, block_n=block_n, block_k=block_k, group_m=8)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_db(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64_lds_sorted_tok_ids(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96_block_k_64(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
        mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
        gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_96(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
        mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
        gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_lds_68(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_16_16_16_padding_masked_store(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)
    #   mlir_c = torch.zeros(M, topk, N, dtype=dtype, device='cuda')
    #   gemm_dtype0_32768_6144_8_64_2_16384_mfma_32_32_8_4_waves_padding_double_buffering(a, b, sorted_ids, expert_ids, num_tokens_post_pad, mlir_c)


if __name__ == "__main__":
    import torch

    # Reproducibility + device sanity
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "No HIP/CUDA device visible"

    # Match your test case
    #M, N, K = 33, 256, 128


    #M,N,K= 2048,1024,256
    M,N,K= 16384,32768,6144

    compare_once(M, N, K)


