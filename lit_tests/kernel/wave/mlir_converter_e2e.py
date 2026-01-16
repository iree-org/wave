# REQUIRES: water
# RUN: python %s | FileCheck %s

import torch
from torch.testing import assert_close
from typing import Any
import sympy

from wave_lang.kernel._support.indexing import IndexSymbol
import wave_lang.kernel.wave as wave
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.mlir_converter.mlir_converter import emit_wave_dialect
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.utils.torch_utils import device_randn, device_zeros
from wave_lang.kernel.wave.water import apply_water_middle_end_passes
from wave_lang.support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
)


@run_test
def test_matrix_add_water_e2e():
    """Test Water PassManager with Wave MLIR dialect generation and e2e execution."""
    torch.manual_seed(0)

    # Simple matrix addition kernel
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE_A = tkl.sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = tkl.sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = tkl.sym.ADDRESS_SPACE_C

    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 2)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 2)),
        tkw.HardwareConstraint(
            threads_per_wave=64, vector_shapes={M: BLOCK_M, N: BLOCK_N}
        ),
    ]

    @wave.wave(constraints)
    def matrix_add(
        a: Memory[M, N, ADDRESS_SPACE_A, tkl.f16],
        b: Memory[M, N, ADDRESS_SPACE_B, tkl.f16],
        c: Memory[M, N, ADDRESS_SPACE_C, tkl.f16],
    ):
        # Load values from memory into registers
        a_reg = wave.read(a)
        b_reg = wave.read(b)

        # Compute the sum
        c_reg = a_reg + b_reg

        # Write results back to memory
        wave.write(c_reg, c)

    # Set parameters for compilation
    subs: dict[str | IndexSymbol, Any] = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        M: 128,
        N: 128,
    }

    options_mlir = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, matrix_add)
    trace = compiled_kernel.compiled_graph
    constraints = matrix_add.constraints

    # Emit Wave dialect MLIR
    wave_dialect_mlir, diagnostics, _ = emit_wave_dialect(
        trace, constraints, options_mlir
    )

    # Apply Water PassManager lowering
    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    print(lowered_mlir)

    # Create test tensors
    shape = (128, 128)
    a_tensor = device_randn(*shape, dtype=torch.float16)
    b_tensor = device_randn(*shape, dtype=torch.float16)
    c_tensor = device_zeros(*shape, dtype=torch.float16)

    # Expected result (CPU computation)
    expected = a_tensor + b_tensor

    # Test execution with lowered MLIR
    options_e2e = WaveCompileOptions(
        subs=subs,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
    )
    options_e2e = set_default_run_config(options_e2e)

    compiled_e2e = wave_compile(options_e2e, matrix_add)

    compiled_e2e(a_tensor, b_tensor, c_tensor)

    assert_close(c_tensor, expected, rtol=1e-4, atol=1e-4)


# CHECK-LABEL:  test_matrix_add_water_e2e
# CHECK:        module
# CHECK-NOT:    wave.normal_form
# CHECK:        func.func @kernel(
# CHECK-NOT:    wave.read
# CHECK:        vector.maskedload
# CHECK:        vector.maskedload
# CHECK-NOT:    wave.add
# CHECK:        arith.addf
# CHECK-NOT:    wave.write
# CHECK:        vector.maskedstore


@run_test
def test_vanilla_attention_water_e2e():
    """Test Water PassManager with vanilla attention kernel and e2e execution.

    TODO: This test is expected to fail until the following operations are
    implemented in the water emitter:

    | Operation | Count | Status           |
    |-----------|-------|------------------|
    | extract   | 36    | Not implemented  |
    | broadcast | 12    | Not implemented  |
    | exp       | 5     | Not implemented  |
    | shuffle   | 4     | Not implemented  |
    | permute   | 4     | Not implemented  |
    | cast      | 4     | Not implemented  |
    | reshape   | 2     | Not implemented  |
    """
    import sys
    import math
    from wave_lang.kernel.wave.constraints import MMAType

    # Increase recursion limit for large attention graph serialization.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)

    torch.manual_seed(0)

    # Input sizes.
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes.
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space.
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Attention shape parameters.
    num_heads = 4
    query_seq_len = 64
    kv_seq_len = 64
    head_size = 64
    head_size_kv = 64

    # Compute scale factor.
    scale = (1.0 / math.sqrt(head_size)) * math.log2(math.e)

    # Define constraints.
    mfma_variant = MMAType.F32_16x16x16_F16
    attn_constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(B, BLOCK_B, 2),
        tkw.TilingConstraint(K2, BLOCK_K2),
        tkw.WaveConstraint(M, sympy.floor(BLOCK_M / 4)),
        tkw.WaveConstraint(N, sympy.floor(BLOCK_N / 1)),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    output_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(attn_constraints)
    def vanilla_attention(
        q: Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k_mem: Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: Memory[B, K2, N, ADDRESS_SPACE, tkl.f16],
        c: Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        qk_scaling = tkl.Register[B, M, K2, tkl.f32](scale)

        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            k_reg = tkw.read(k_mem)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            x_j = x_j * qk_scaling
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=output_mapping)

    # Hyperparameters.
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: num_heads,
        M: query_seq_len,
        N: head_size_kv,
        K1: head_size,
        K2: kv_seq_len,
    }

    q_shape = (num_heads, query_seq_len, head_size)
    k_shape = (num_heads, kv_seq_len, head_size)
    v_shape = (num_heads, kv_seq_len, head_size_kv)
    o_shape = (num_heads, query_seq_len, head_size_kv)

    # Compile to get trace for MLIR emission.
    options_mlir = WaveCompileOptions(
        subs=hyperparams,
        compile_to_mlir=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
    )
    options_mlir = set_default_run_config(options_mlir)

    compiled_kernel = wave_compile(options_mlir, vanilla_attention)
    trace = compiled_kernel.compiled_graph
    constraints = vanilla_attention.constraints

    # Emit Wave dialect MLIR.
    # Expected to fail with NotImplementedError for missing operations.
    try:
        wave_dialect_mlir, diagnostics, _ = emit_wave_dialect(
            trace, constraints, options_mlir
        )
    except RuntimeError as e:
        error_str = str(e)
        if "Missing support for" in error_str or "NotImplementedError" in error_str:
            # Extract the key error message from the full traceback.
            if "Missing support for" in error_str:
                idx = error_str.find("Missing support for")
                short_msg = error_str[idx:].split("\n")[0]
            else:
                short_msg = "NotImplementedError in water_emitter"
            print(f"XFAIL: {short_msg}")
            sys.setrecursionlimit(old_limit)
            return
        raise

    # Apply Water PassManager lowering.
    lowered_mlir = apply_water_middle_end_passes(wave_dialect_mlir)

    print(lowered_mlir)

    # Create test tensors.
    q_tensor = device_randn(*q_shape, dtype=torch.float16)
    k_tensor = device_randn(*k_shape, dtype=torch.float16)
    v_tensor = device_randn(*v_shape, dtype=torch.float16)
    output = device_zeros(*o_shape, dtype=torch.float32)

    # Expected result using PyTorch reference.
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, attn_mask=None
    )

    # Test execution with lowered MLIR.
    options_e2e = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        location_capture_config=LocationCaptureConfig(level=LocationCaptureLevel.NONE),
        enforce_locations=False,
        override_mlir=lowered_mlir,
    )
    options_e2e = set_default_run_config(options_e2e)

    compiled_e2e = wave_compile(options_e2e, vanilla_attention)

    compiled_e2e(q_tensor, k_tensor, v_tensor, output)

    assert_close(output, expected, rtol=1e-3, atol=1e-3, check_dtype=False)

    # Restore original recursion limit.
    sys.setrecursionlimit(old_limit)


# CHECK-LABEL:  test_vanilla_attention_water_e2e
# CHECK:        XFAIL: Missing support for
