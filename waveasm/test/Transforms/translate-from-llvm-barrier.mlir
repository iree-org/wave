// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify barrier and fence translation.
// rocdl.barrier -> s_barrier; rocdl.s.barrier -> s_barrier; llvm.fence -> no-op.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.s_barrier
// CHECK: waveasm.s_barrier
// fence produces no output.
// CHECK-NOT: fence
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    rocdl.barrier
    rocdl.s.barrier
    llvm.fence acquire
    llvm.return
  }
}
