// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify the LLVM→WaveASM translation pass handles a copy kernel.

// CHECK: gpu.module @gpu_module
// CHECK:   llvm.func @test
// CHECK:   waveasm.program @test__waveasm
// CHECK-SAME: kernel_name = "test"
// CHECK: waveasm.precolored.vreg
// CHECK: waveasm.v_cmp_lt_i32
// CHECK: waveasm.v_cndmask_b32
// CHECK: waveasm.v_mul_lo_u32
// CHECK: waveasm.buffer_load_ushort
// CHECK: waveasm.buffer_store_short
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(1073741823 : index) : i64
    %1 = llvm.mlir.constant(2147483645 : i64) : i64
    %2 = llvm.mlir.constant(27 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(0 : i16) : i16
    %5 = llvm.mlir.constant(822243328 : i32) : i32
    %6 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %7 = llvm.sext %6 : i32 to i64
    %8 = llvm.trunc %7 : i64 to i32
    %9 = llvm.icmp "slt" %8, %2 : i32
    %10 = rocdl.make.buffer.rsrc %arg0, %4, %1, %5 : !llvm.ptr to <7>
    %11 = llvm.select %9, %7, %0 : i1, i64
    %12 = llvm.trunc %11 : i64 to i32
    %13 = llvm.mul %12, %3 overflow<nsw> : i32
    %14 = llvm.zext %13 : i32 to i64
    %15 = llvm.getelementptr nusw %10[%14] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    %16 = llvm.load %15 : !llvm.ptr<7> -> vector<1xf16>
    %17 = rocdl.make.buffer.rsrc %arg1, %4, %1, %5 : !llvm.ptr to <7>
    %18 = llvm.getelementptr nusw %17[%14] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    llvm.store %16, %18 : vector<1xf16>, !llvm.ptr<7>
    llvm.return
  }
}
