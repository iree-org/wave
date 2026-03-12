// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify llvm.mlir.poison is translated as zero for both i32 and i64.

// CHECK: waveasm.program @test__waveasm
// CHECK: waveasm.constant 0
// CHECK: waveasm.v_mov_b32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.poison : i32
    llvm.return
  }
}

// -----

// CHECK: waveasm.program @test_i64__waveasm
// CHECK: [[ZERO:%.*]] = waveasm.constant 0
// CHECK: [[LO:%.*]] = waveasm.v_mov_b32 [[ZERO]]
// CHECK: [[HI:%.*]] = waveasm.v_mov_b32 [[ZERO]]
// CHECK: waveasm.pack [[LO]], [[HI]]
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module_i64 {
  llvm.func @test_i64() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.poison : i64
    llvm.return
  }
}
