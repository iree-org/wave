// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify scalar (non-pointer) kernel arguments are mapped to preloaded SGPRs,
// not treated as buffer pointers. i64 args are truncated to 32-bit for VALU ops.
// TODO: Proper i64 legalization instead of truncation.

// CHECK: waveasm.program @test__waveasm
// Pointer arg gets SRD setup.
// CHECK: waveasm.precolored.sreg [[SRD:[0-9]+]], 4
// CHECK: waveasm.raw "s_mov_b32
// CHECK: waveasm.raw "s_mov_b32
// Scalar arg mapped to preloaded SGPR pair, then narrowed to low register.
// CHECK: waveasm.precolored.sreg [[PAIR:[0-9]+]], 2
// CHECK: waveasm.precolored.sreg [[PAIR]] :
// CHECK: waveasm.v_cmp_lt_i32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %1 = llvm.sext %0 : i32 to i64
    %2 = llvm.icmp "slt" %1, %arg1 : i64
    llvm.return
  }
}
