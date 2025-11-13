# RUN: python %s | FileCheck %s

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.general_utils import run_test
import wave_lang.kernel.wave as wave
from wave_lang.kernel.wave.templates.test_kernels import (
    get_gemm_prefetch_kernel_and_schedule,
)
from wave_lang.kernel.wave.schedules.gemm_two_pp_cluster import (
    get_gemm_two_pp_cluster_schedule,
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

    options.use_scheduling_barriers = True

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

    # Epilogue
    # CHECK-COUNT-4:  vector.load %[[VIEW_0]]
    # CHECK-COUNT-4:  vector.load %[[VIEW_1]]
    # CHECK-COUNT-8:  amdgpu.mfma


@run_test
def test_gemm_prefetch_reorder_stagger():
    # Define test parameters for advanced scheduling
    shape = (4096, 4096, 4096)  # M, N, K dimensions
    mfma_variant = wave.MMAType.F32_16x16x16_F16
    compile_to_mlir = True

    # Get the kernel, schedule, and options from the template
    (
        gemm_prefetch_reorder,
        prefetch_reorder_schedule,
        options,
    ) = get_gemm_two_pp_cluster_schedule(shape, mfma_variant, compile_to_mlir)

    gemm_prefetch_reorder = wave_compile(
        options, gemm_prefetch_reorder, prefetch_reorder_schedule
    )
    print(gemm_prefetch_reorder.asm)

    # CHECK-LABEL:    func.func @gemm_prefetch_reorder
    # CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:       %[[C34816:.+]] = arith.constant 34816 : index
    # CHECK:           %[[VIEW_0:.*]] = memref.view %alloc[%[[C0]]][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<256x68xf16, #gpu.address_space<workgroup>>
    # CHECK:           %[[VIEW_1:.*]] = memref.view %alloc[%[[C34816]]][] : memref<52224xi8, #gpu.address_space<workgroup>> to memref<128x68xf16, #gpu.address_space<workgroup>>
    # Prologue
    # CHECK-COUNT-6:  vector.load
    # CHECK-COUNT-6:  vector.store
    # CHECK:          amdgpu.lds_barrier

    # Warp High and Warp Lo computation
    # CHECK:         %[[FLAT_WAVE_ID:.+]] = affine.apply #{{.*}}()[%thread_id_x, %thread_id_y]
    # CHECK:         %[[FLAT_WAVE_ID_I32:.+]] = arith.index_cast %[[FLAT_WAVE_ID]] : index to i32
    # CHECK:         %[[WARP_HI:.+]] = arith.cmpi sge, %[[FLAT_WAVE_ID_I32]], %c4_i32 : i32
    # CHECK:         %[[WARP_LO:.+]] = arith.cmpi slt, %[[FLAT_WAVE_ID_I32]], %c4_i32 : i32

    # cond_barrier on warp hi to brings assymetry between 2 wave in same SIMD and Block.
    # CHECK:          scf.if %[[WARP_HI]] {
    # CHECK-NEXT:       rocdl.s.barrier
    # CHECK-NEXT:     }

    # Steady State
    # CHECK:          scf.for

    # 1st cluster interleaved local and global reads.

    # 1st Cluster: First slice of Local read lhs and rhs
    # CHECK-COUNT-2:    vector.load %[[VIEW_1]][{{.*}}, %[[K0:.+]]] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-2:    vector.load %[[VIEW_1]][{{.*}}, %[[K1:.+]]] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-8:    vector.load %[[VIEW_0]][{{.*}}, %[[K0]]] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-8:    vector.load %[[VIEW_0]][{{.*}}, %[[K1]]] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier

    # 1st Cluster: Global load LHS
    # CHECK-COUNT-2:    vector.load {{.*}} : memref<4096x4096xf16, strided<[4096, 1]>>, vector<8xf16>
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier

    # 1st Cluster: Second slice of Local read lhs and rhs
    # CHECK-COUNT-2:    vector.load %[[VIEW_1]][{{.*}}, %[[K2:.+]]] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-2:    vector.load %[[VIEW_1]][{{.*}}, %[[K3:.+]]] : memref<128x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-8:    vector.load %[[VIEW_0]][{{.*}}, %[[K2]]] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-8:    vector.load %[[VIEW_0]][{{.*}}, %[[K3]]] : memref<256x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier

    # 1st Cluster: Global load RHS
    # CHECK-COUNT-4:    vector.load {{.*}} : memref<4096x4096xf16, strided<[4096, 1]>>, vector<8xf16>
    # CHECK:            rocdl.s.barrier
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier

    # First dot slice
    # CHECK:            rocdl.s.setprio 1
    # CHECK-COUNT-32:   amdgpu.mfma
    # CHECK:            rocdl.s.setprio 0
    # CHECK:            amdgpu.lds_barrier
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"

    # 2nd cluster local writes.
    # CHECK-COUNT-6:    vector.store
    # CHECK:            rocdl.s.barrier
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"

    # Second dot slice:
    # CHECK:            rocdl.s.setprio 1
    # CHECK-COUNT-32:   amdgpu.mfma
    # CHECK:            rocdl.s.setprio 0
    # CHECK:            llvm.call_intrinsic "llvm.amdgcn.sched.barrier"

    # Final LDS barrier to synchronize shared writes.
    # CHECK:            amdgpu.lds_barrier
    # CHECK:            scf.yield
    # CHECK:          }

    # Prologue

    # cond_barrier on warp low to even out assymetry between 2 wave in same SIMD and Block.
    # CHECK:          scf.if %[[WARP_LO]] {
    # CHECK-NEXT:       rocdl.s.barrier
    # CHECK-NEXT:     }

    # CHECK-COUNT-32: vector.load %[[VIEW_0]]
    # CHECK-COUNT-8:  vector.load %[[VIEW_1]]
    # CHECK-COUNT-64: amdgpu.mfma
