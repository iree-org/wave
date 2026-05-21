// RUN: waveasm-translate %s --waveasm-translate-from-llvm | FileCheck %s
// Verify LDS addressof, GEP, load, and store translation.
// addressof assigns per-global LDS byte offsets; LDS GEPs produce byte
// offsets; loads/stores dispatch to ds_read/ds_write by width; and LDS size is
// counted in bytes.

// CHECK: waveasm.program @test__waveasm
// CHECK: lds_size = 1040 : i64
// First addressof uses byte offset 0.
// CHECK: waveasm.constant 0
// CHECK: waveasm.v_mov_b32
// Second addressof uses the next available LDS byte offset.
// CHECK: waveasm.constant 1024
// GEP [0,0] on the array type passes through (all-zero indices).
// GEP with constant offset 512 produces an arith.add.
// CHECK: waveasm.arith.add
// 4-byte LDS load -> ds_read_b32.
// CHECK: waveasm.ds_read_b32
// 4-byte LDS store -> ds_write_b32.
// CHECK: waveasm.ds_write_b32
// CHECK: waveasm.s_endpgm

gpu.module @gpu_module {
  llvm.mlir.global private @alloca() {addr_space = 3 : i32} : !llvm.array<256 x i32>
  llvm.mlir.global private @scratch() {addr_space = 3 : i32} : !llvm.array<4 x i32>
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %0 = llvm.mlir.addressof @alloca : !llvm.ptr<3>
    %scratch = llvm.mlir.addressof @scratch : !llvm.ptr<3>
    %1 = llvm.mlir.constant(42 : i32) : i32
    // Multi-index GEP with all-zero indices -> passthrough.
    %2 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.array<256 x i32>
    // Constant-offset GEP. 128 * sizeof(i32) = 512 bytes.
    %3 = llvm.getelementptr %2[128] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i32
    // Dynamic-offset GEP.
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %tidext = llvm.sext %tid : i32 to i64
    %4 = llvm.getelementptr nusw %2[%tidext] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    // LDS load (4 bytes).
    %5 = llvm.load %4 : !llvm.ptr<3> -> i32
    // LDS store (4 bytes).
    llvm.store %1, %3 : i32, !llvm.ptr<3>
    // Distinct LDS global uses a distinct non-zero base offset.
    llvm.store %1, %scratch : i32, !llvm.ptr<3>
    llvm.return
  }
}
