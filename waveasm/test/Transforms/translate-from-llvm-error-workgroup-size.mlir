// RUN: not waveasm-translate %s --waveasm-translate-from-llvm 2>&1 | FileCheck %s
// Verify that contradicting workgroup size attributes produce a diagnostic.

// CHECK: contradicting workgroup size attributes

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {
    gpu.kernel,
    gpu.known_block_size = array<i32: 64, 1, 1>,
    rocdl.kernel,
    rocdl.reqd_work_group_size = array<i32: 128, 1, 1>
  } {
    llvm.return
  }
}
