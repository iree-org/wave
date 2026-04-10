// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that GEPs on unsupported address spaces produce a diagnostic.

// CHECK: unsupported address space 5

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr<5>) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.getelementptr %arg0[%0] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, i8
    llvm.return
  }
}
