// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify workgroup ID intrinsics are translated to precolored SGPRs.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.precolored.sreg
// CHECK: waveasm.precolored.sreg
// CHECK: waveasm.precolored.sreg
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = rocdl.workgroup.id.x : i32
    %1 = rocdl.workgroup.id.y : i32
    %2 = rocdl.workgroup.id.z : i32
    llvm.return
  }
}
