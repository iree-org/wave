// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify bare-pointer GEP before make.buffer.rsrc followed by chained buffer
// GEPs. The bare-pointer offset is seeded into the GEP map so subsequent
// buffer GEPs chain via v_add_u32 without special-casing.

// CHECK: waveasm.program @test__waveasm

// SRD word 1 patched.
// CHECK: waveasm.raw "s_and_b32

// Bare-pointer offset folded into SRD base via readfirstlane + s_add_u32.
// CHECK: waveasm.arith.readfirstlane
// CHECK: waveasm.arith.trunc
// CHECK: waveasm.s_add_u32
// CHECK: waveasm.s_addc_u32

// Buffer GEP voffset starts at 0: tid (truncated) + col_bytes (truncated).
// CHECK: [[TID:%.*]] = waveasm.arith.trunc
// CHECK: [[COL:%.*]] = waveasm.arith.trunc
// CHECK: [[OFF:%.*]] = waveasm.v_add_u32 [[TID]], [[COL]]

// Load uses the chained voffset (no bare-pointer component).
// CHECK: waveasm.buffer_load_dword %{{.*}}, [[OFF]], %{{.*}}
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.func @test(%arg0: !llvm.ptr) attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %num_records = llvm.mlir.constant(2147483645 : i64) : i64
    %stride = llvm.mlir.constant(0 : i16) : i16
    %flags = llvm.mlir.constant(131072 : i32) : i32
    %row_bytes = llvm.mlir.constant(512 : i64) : i64
    %col_bytes = llvm.mlir.constant(4 : i64) : i64

    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %tid64 = llvm.sext %tid : i32 to i64

    // Bare-pointer GEP: row offset before make.buffer.rsrc.
    %row_ptr = llvm.getelementptr nusw %arg0[%row_bytes] : (!llvm.ptr, i64) -> !llvm.ptr, i8

    // Convert to buffer resource -- seeds GEP map with bare-pointer offset.
    %rsrc = rocdl.make.buffer.rsrc %row_ptr, %stride, %num_records, %flags : !llvm.ptr to <7>

    // First buffer GEP: tid-based byte offset chains on seeded entry.
    %gep0 = llvm.getelementptr nusw %rsrc[%tid64] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8

    // Second buffer GEP: column offset chains on first.
    %gep1 = llvm.getelementptr nusw %gep0[%col_bytes] : (!llvm.ptr<7>, i64) -> !llvm.ptr<7>, i8

    %val = llvm.load %gep1 : !llvm.ptr<7> -> i32
    llvm.return
  }
}
