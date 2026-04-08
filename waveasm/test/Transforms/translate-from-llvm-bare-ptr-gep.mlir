// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify bare-pointer GEPs before make.buffer.rsrc are handled.
// The byte offset from bare-pointer GEPs must be added to the buffer voffset.

// CHECK: waveasm.program @test__waveasm
// Base offset (bare-ptr GEP) + element offset combined into voffset.
// CHECK: waveasm.arith.add
// SRD word 1 stride bits cleared, flags patched from make.buffer.rsrc.
// CHECK: waveasm.s_and_b32
// CHECK: waveasm.buffer_load_ushort
// CHECK: waveasm.arith.add
// CHECK: waveasm.s_and_b32
// CHECK: waveasm.buffer_store_short
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"} {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(2147483645 : i64) : i64
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(128 : i32) : i32
    %3 = llvm.mlir.constant(159744 : i32) : i32
    %4 = llvm.mlir.constant(16448 : i16) : i16
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(2 : index) : i64
    %7 = rocdl.workgroup.id.y range <i32, 0, 256> : i32
    %8 = llvm.sext %7 : i32 to i64
    %9 = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %10 = llvm.sext %9 : i32 to i64
    %11 = llvm.trunc %8 : i64 to i32
    %12 = llvm.mul %11, %2 overflow<nsw> : i32
    %13 = llvm.zext %12 : i32 to i64
    // Bare-pointer GEP on arg — computes row offset.
    %14 = llvm.getelementptr nusw %arg0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %15 = llvm.mul %5, %6 overflow<nsw> : i64
    // Chained bare-pointer GEP.
    %16 = llvm.getelementptr nusw %14[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    // make.buffer.rsrc on the offset pointer.
    %17 = rocdl.make.buffer.rsrc %16, %4, %0, %3 : !llvm.ptr to <7>
    %18 = llvm.trunc %10 : i64 to i32
    %19 = llvm.mul %18, %1 overflow<nsw> : i32
    %20 = llvm.zext %19 : i32 to i64
    // Buffer GEP — base offset from bare GEPs should be added here.
    %21 = llvm.getelementptr nusw %17[%20] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    %22 = llvm.load %21 : !llvm.ptr<7> -> vector<1xf16>
    // Same pattern for store side.
    %23 = llvm.getelementptr nusw %arg1[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %24 = llvm.getelementptr nusw %23[%15] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %25 = rocdl.make.buffer.rsrc %24, %4, %0, %3 : !llvm.ptr to <7>
    %26 = llvm.getelementptr nusw %25[%20] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    llvm.store %22, %26 : vector<1xf16>, !llvm.ptr<7>
    llvm.return
  }
}
