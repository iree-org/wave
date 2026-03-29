// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify scalar (non-pointer) kernel arguments are mapped to preloaded SGPRs,
// not treated as buffer pointers. i64 args produce arith pseudo-ops that the
// legalization pass will lower to concrete SALU/VALU ops.

// CHECK: waveasm.program @test__waveasm
// Pointer arg gets SRD setup with typed ops.
// CHECK: waveasm.precolored.sreg [[SRD:[0-9]+]], 4
// CHECK: waveasm.s_mov_b32
// CHECK: waveasm.dce_protect
// CHECK: waveasm.s_mov_b32
// CHECK: waveasm.dce_protect
// Scalar arg mapped to preloaded SGPR pair, sext/cmp emitted as pseudo-ops.
// CHECK: waveasm.precolored.sreg [[PAIR:[0-9]+]], 2
// CHECK: waveasm.arith.sext
// CHECK: waveasm.arith.cmp slt
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %1 = llvm.sext %0 : i32 to i64
    %2 = llvm.icmp "slt" %1, %arg1 : i64
    llvm.return
  }
}
