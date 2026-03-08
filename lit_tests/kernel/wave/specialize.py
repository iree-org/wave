# RUN: python %s | FileCheck %s

import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_wmma_specialize():
    m = 1024
    n = 1024
    k = 1024
    shape = (m, n, k)
    waves_per_block = (2, 2)
    mma_type = tkw.MMAType.GFX1250_F32_16x16x32_F16
    tpw = 32
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape,
        False,
        mma_type,
        torch.float16,
        threads_per_wave=tpw,
        waves_per_block=waves_per_block,
        n_service_waves=1,
    )

    compile_options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        compile_to_mlir=True,
        specialize=True,
        target="gfx1250",
    )
    gemm_kern = wave_compile(compile_options, gemm)
    print(gemm_kern.asm)

    # CHECK-LABEL: test_wmma_specialize
    # CHECK:        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 3, 1] subgroup_size = 32

    # CHECK:            stream.return  %c16, %c16, %c1 : index, index, index
    # CHECK:            func.func @gemm
    # CHECK:                gpu.block_id  x upper_bound 16
    # CHECK:                gpu.block_id  y upper_bound 16
    # CHECK:                gpu.thread_id  x upper_bound 64
    # CHECK:                gpu.thread_id  y upper_bound 3
    # CHECK:                scf.for
    # CHECK:                    rocdl.s.barrier.wait
    # CHECK:                    rocdl.s.barrier.signal
    # CHECK-COUNT-4:            rocdl.wmma.f32.16x16x32.f16
    # CHECK-COUNT-32:           vector.extract_strided_slice
    # CHECK:                    vector.store
    # CHECK:                    return
