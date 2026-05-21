// RUN: waveasm-translate %s --waveasm-llvm-sdiv-srem-legalization | FileCheck %s
// Verify that the LLVM pre-pass rewrites signed div/rem by positive power-of-2
// constants before LLVM->WaveASM translation.

// CHECK-LABEL: llvm.func @test
// CHECK: llvm.icmp "slt"
// CHECK: llvm.select
// CHECK: llvm.ashr
// CHECK: llvm.and
// CHECK: llvm.icmp "ne"
// CHECK: llvm.select

gpu.module @gpu_module {
  llvm.mlir.global private @scratch() {addr_space = 3 : i32} : i32
  llvm.func @test() attributes {gpu.kernel, gpu.known_block_size = array<i32: 64, 1, 1>, rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 64, 1, 1>} {
    %scratch = llvm.mlir.addressof @scratch : !llvm.ptr<3>
    %tid = rocdl.workitem.id.x range <i32, 0, 64> : i32
    %16 = llvm.mlir.constant(16 : i32) : i32
    %neg5 = llvm.mlir.constant(-5 : i32) : i32
    %4 = llvm.mlir.constant(4 : i32) : i32
    %div = llvm.sdiv %tid, %16 : i32
    %rem = llvm.srem %tid, %16 : i32
    %negdiv = llvm.sdiv %neg5, %4 : i32
    %negrem = llvm.srem %neg5, %4 : i32
    %sum0 = llvm.add %div, %rem : i32
    %sum1 = llvm.add %sum0, %negdiv : i32
    %sum2 = llvm.add %sum1, %negrem : i32
    llvm.store %sum2, %scratch : i32, !llvm.ptr<3>
    llvm.return
  }
}
