// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify barrier and fence translation.
// rocdl.barrier → s_barrier; rocdl.s.barrier → s_barrier; llvm.fence → no-op.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.s_barrier
// CHECK: waveasm.s_barrier
// fence produces no output.
// CHECK-NOT: fence
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    rocdl.barrier
    rocdl.s.barrier
    llvm.fence acquire
    llvm.return
  }
}
