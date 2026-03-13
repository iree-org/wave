// RUN: not waveasm-translate %s --waveasm-translate-from-llvm='target=gfx942' 2>&1 | FileCheck %s
// Verify that non-gfx950 targets are rejected.

// CHECK: LLVM->WaveASM translation only supports gfx950

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {
    gpu.kernel,
    gpu.known_block_size = array<i32: 64, 1, 1>,
    rocdl.kernel,
    rocdl.reqd_work_group_size = array<i32: 64, 1, 1>
  } {
    llvm.return
  }
}
