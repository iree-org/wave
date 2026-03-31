// RUN: waveasm-translate %s --waveasm-translate-from-llvm --waveasm-arith-legalization | FileCheck %s
// Verify that i64 GEP indices are truncated to i32 for buffer voffset.

// CHECK-LABEL: waveasm.program @test_i64_gep_offset__waveasm

// zext i32 -> i64: pack {mul_result, 0} into vreg<2>.
// CHECK: %[[MUL:.*]] = waveasm.v_mul_lo_u32
// CHECK: waveasm.pack %[[MUL]], %{{.*}} : (!waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<2, 2>

// Trunc looks through pack, uses mul result directly as buffer voffset.
// CHECK: waveasm.buffer_load_ushort %{{.*}}, %[[MUL]], %{{.*}} : !waveasm.sreg<4, 4>, !waveasm.vreg, !waveasm.imm<0>

// No arith pseudo-ops should remain.
// CHECK-NOT: waveasm.arith.
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test_i64_gep_offset(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.constant(2147483645 : i64) : i64
    %1 = llvm.mlir.constant(822243328 : i32) : i32
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.mlir.constant(2 : i32) : i32
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %mul = llvm.mul %tid, %3 : i32
    %ext = llvm.zext %mul : i32 to i64
    %rsrc = rocdl.make.buffer.rsrc %arg0, %2, %0, %1 : !llvm.ptr to <7>
    %gep = llvm.getelementptr nusw %rsrc[%ext] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8
    %val = llvm.load %gep : !llvm.ptr<7> -> vector<1xf16>
    llvm.return
  }
}
