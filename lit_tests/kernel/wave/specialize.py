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

    #### workgroups ####
    # CHECK:            stream.return  %c16, %c16, %c1 : index, index, index
    # CHECK:        }

    #### pre-specialize conditions ####
    # CHECK-DAG:            func.func @gemm
    # CHECK-DAG:                %[[BIDX:.*]] = gpu.block_id  x upper_bound 16
    # CHECK-DAG:                %[[BIDY:.*]] = gpu.block_id  y upper_bound 16
    # CHECK-DAG:                %[[TIDX:.*]] = gpu.thread_id  x upper_bound 64
    # CHECK-DAG:                %[[TIDY:.*]] = gpu.thread_id  y upper_bound 3

    # CHECK-DAG:                %[[WAVE_INDEX:.*]] = affine.apply #map()[%[[TIDX]], %[[TIDY]]]
    # CHECK-DAG:                %[[WAVE_ID:.*]] = arith.index_cast %[[WAVE_INDEX]] : index to i32

    # CHECK-DAG:                %[[LOAD_COND:.*]] = arith.cmpi sge, %[[WAVE_ID]]
    # CHECK-DAG:                %[[COMP_COND:.*]] = arith.cmpi slt, %[[WAVE_ID]]

    #### iterate subgraph ####
    # CHECK-DAG:                %[[ITER:.*]]:4 = scf.for %[[K:.*]] = {{.*}} iter_args(%[[A4:.*]] = {{.*}}, %[[A5:.*]] = {{.*}}, %[[A6:.*]] = {{.*}}, %[[A7:.*]] = {{.*}})

    #### load partition ####
    # CHECK-DAG:                    scf.if %[[LOAD_COND]] {
    # CHECK-COUNT-4:                    vector.load
    # CHECK-DAG:                        %[[FIRST_ITER:.*]] = arith.cmpi ne, %[[K]], %c0 : index
    # CHECK-DAG:                        %[[WAVE_MAP:.*]] = affine.apply #{{.*}}()[%[[TIDX]], %[[TIDY]]]

    # CHECK-COUNT-4:                    arith.cmpi eq, %[[WAVE_MAP]]
    # CHECK-NEXT:                       arith.andi %[[FIRST_ITER]]
    # CHECK-NEXT:                       scf.if {{.*}} {
    # CHECK-NEXT:                           rocdl.s.barrier.wait
    # CHECK-NEXT:                       }

    # CHECK-COUNT-8:                    vector.store

    # CHECK-COUNT-4:                    scf.if {{.*}} {
    # CHECK-NEXT:                           rocdl.s.wait.dscnt 0
    # CHECK-NEXT:                           rocdl.s.barrier.signal
    # CHECK-NEXT:                       }

    # CHECK-DAG:                    }

    #### compute partition ####
    # CHECK-DAG:                    %[[COMP_RESULT:.*]]:4 = scf.if %[[COMP_COND]] -> {{.*}} {

    # CHECK-COUNT-4:                    arith.cmpi eq, %[[WAVE_INDEX]]
    # CHECK-NEXT:                       scf.if {{.*}} {
    # CHECK-NEXT:                           rocdl.s.barrier.wait
    # CHECK-NEXT:                       }

    # CHECK-DAG:                        %[[R1:.*]] = rocdl.wmma.f32.16x16x32.f16
    # CHECK-NEXT:                       %[[R2:.*]] = rocdl.wmma.f32.16x16x32.f16
    # CHECK-NEXT:                       %[[R3:.*]] = rocdl.wmma.f32.16x16x32.f16
    # CHECK-NEXT:                       %[[R4:.*]] = rocdl.wmma.f32.16x16x32.f16
    # CHECK-NEXT:                       scf.yield %[[R4]], %[[R3]], %[[R2]], %[[R1]]
    # CHECK-NEXT:                   } else {
    # CHECK-NEXT:                       scf.yield
    # CHECK-NEXT:                   }

    # CHECK-NEXT:                   scf.yield %[[COMP_RESULT]]#0, %[[COMP_RESULT]]#1, %[[COMP_RESULT]]#2, %[[COMP_RESULT]]#3
    # CHECK-NEXT:               }

    #### store graph ####
    # CHECK-DAG:                scf.if %[[COMP_COND]] {
    # CHECK-COUNT-32:               vector.extract_strided_slice
    # CHECK-NEXT:                   vector.store

    # CHECK:                    }
    # CHECK:                    return
