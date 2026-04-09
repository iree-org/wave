// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that non-zero multi-index bare-pointer GEPs fail instead of silently
// dropping the aggregate offset.

// CHECK: bare-pointer GEP must have a single index or all-zero structural indices

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
    llvm.return
  }
}
