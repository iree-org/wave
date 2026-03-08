// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify sdiv and srem with power-of-2 constants.
// sdiv → v_ashrrev_i32; srem → v_and_b32.

// CHECK: waveasm.program @test__waveasm
// sdiv by 16 → shift right by 4.
// CHECK: waveasm.v_ashrrev_i32
// srem by 16 → and with 15.
// CHECK: waveasm.v_and_b32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %16 = llvm.mlir.constant(16 : i32) : i32
    %div = llvm.sdiv %tid, %16 : i32
    %rem = llvm.srem %tid, %16 : i32
    llvm.return
  }
}
