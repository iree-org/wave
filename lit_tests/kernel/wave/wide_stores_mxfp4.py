# RUN: python %s | FileCheck %s

"""
Test wide store coalescing for preshuffle-B MXFP4 GEMM with bf16 output.

When wide_stores=True, the kernel swaps MFMA operands (B as LHS, A as RHS)
so the accumulator's 4-contiguous values align with the output's stride-1
dimension. The coalesce_epilogue_stores pass tags eligible bf16 global
writes, and the codegen emits v_permlane16_swap_b32 to exchange data
between lane pairs 16 apart, producing 8 consecutive bf16 values written
as a single buffer_store_dwordx4.

Key structural invariants verified:
  1. Function signature accepts dynamic index arguments for M, N, K.
  2. Register type is [N, M] (swapped from standard [M, N]).
  3. scaled_mfma has B as LHS and A as RHS (swapped operands).
  4. rocdl.permlane16.swap used for lane exchange.
  5. vector.store of vector<8xbf16> for wide stores.
  6. arith.truncf for f32 -> bf16 conversion.
"""

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.utils.general_utils import run_test


@run_test
def test_wide_stores_preshuffle_b_mxfp4():
    shape = (1024, 3072, 8192)
    block = (256, 192, 256)
    kernel, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        wide_stores=True,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    schedule = get_mxfp4_asymmetric_schedule(is_bscale_shuffled=True)
    options.use_buffer_ops = True
    options.compile_to_mlir = True
    options.device = "hip"
    options.target = "gfx950"
    result = wave_compile(options, kernel, schedule)
    print(result.asm)

    # CHECK-LABEL: test_wide_stores_preshuffle_b_mxfp4

    # 1. Dynamic index arguments for M, N, K in function signature.
    # CHECK: func.func @gemm(%arg0: {{.*}}, %arg1: {{.*}}, %arg2: {{.*}}, %arg3: {{.*}}, %arg4: {{.*}}, %arg5: index, %arg6: index, %arg7: index)

    # 2. f32 -> bf16 conversion in the epilogue.
    # CHECK: arith.truncf %{{.*}} : vector<4xf32> to vector<4xbf16>

    # 3. vector.bitcast from bf16 to i32 for permlane swap.
    # CHECK: vector.bitcast %{{.*}} : vector<4xbf16> to vector<2xi32>

    # 4. rocdl.permlane16.swap for lane exchange.
    # CHECK: rocdl.permlane16.swap

    # 5. llvm.extractvalue to get the swapped value.
    # CHECK: llvm.extractvalue %{{.*}}[0]

    # 6. arith.select to choose between original and swapped values.
    # CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32

    # 7. vector.from_elements to pack 4 i32 into vector<4xi32>.
    # CHECK: vector.from_elements %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>

    # 8. vector.bitcast from i32 to bf16 for the wide store.
    # CHECK: vector.bitcast %{{.*}} : vector<4xi32> to vector<8xbf16>

    # 9. Wide store of 8 bf16 values.
    # CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<{{.*}}xbf16, {{.*}}>, vector<8xbf16>
