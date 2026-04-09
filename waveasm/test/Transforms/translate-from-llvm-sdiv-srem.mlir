// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify sdiv and srem with power-of-2 constants.
// sdiv -> v_ashrrev_i32; srem -> v_and_b32.

// CHECK: waveasm.program @test__waveasm
// sdiv by 16 -> shift right by 4.
// CHECK: waveasm.v_ashrrev_i32
// srem by 16 -> and with 15.
// CHECK: waveasm.v_and_b32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %16 = llvm.mlir.constant(16 : i32) : i32
    %div = llvm.sdiv %tid, %16 : i32
    %rem = llvm.srem %tid, %16 : i32
    llvm.return
  }
}
