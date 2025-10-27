# RUN: python %s | FileCheck %s

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test
import wave_lang.kernel.wave as wave
from wave_lang.kernel.wave.templates.test_kernels import (
    get_gemm_prefetch_kernel_and_schedule,
)


@run_test
def test_gemm_with_wave_schedule():
    # Define test parameters
    shape = (128, 128, 128)  # M, N, K dimensions
    mfma_variant = wave.MMAType.F32_16x16x16_F16
    compile_to_mlir = True

    # Get the kernel, schedule, and options from the template
    gemm_prefetch, prefetch_schedule, options = get_gemm_prefetch_kernel_and_schedule(
        shape, mfma_variant, compile_to_mlir
    )

    gemm_prefetch = wave_compile(options, gemm_prefetch, prefetch_schedule)
    print(gemm_prefetch.asm)

    # This is the same test as gemm_prefetch, but with wave_schedule instead of manually setting the schedule.
    # CHECK-LABEL:    func.func @gemm_prefetch
    # CHECK:          %[[VIEW_0:.+]] = memref.view %alloc[%c0][]
    # CHECK:          %[[VIEW_1:.+]] = memref.view %alloc[%c4608][]
    # Prologue
    # CHECK-COUNT-2:  vector.load
    # CHECK-COUNT-2:  vector.store

    # Steady State
    # CHECK:          scf.for
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # Steady State Local Read
    # CHECK-COUNT-4:    vector.load %[[VIEW_0]]
    # CHECK-COUNT-4:    vector.load %[[VIEW_1]]

    # Steady State Global Read
    # CHECK-COUNT-2:    vector.load {{.*}} : memref<128x128xf16, strided<[128, 1]>>, vector<8xf16>
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"

    # Compute
    # CHECK-COUNT-8:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier

    # Steady State Local Write
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK:          scf.yield

    # Prologue
    # CHECK-COUNT-4:  vector.load %[[VIEW_0]]
    # CHECK-COUNT-4:  vector.load %[[VIEW_1]]
    # CHECK-COUNT-8:  amdgpu.mfma
