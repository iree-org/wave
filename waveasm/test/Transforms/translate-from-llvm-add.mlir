// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify llvm.add is translated to v_add_u32 and constants become immediates.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.constant 42
// CHECK: waveasm.constant 7
// CHECK: waveasm.v_add_u32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(42 : i32) : i32
    %1 = llvm.mlir.constant(7 : i32) : i32
    %2 = llvm.add %0, %1 : i32
    llvm.return
  }
}
