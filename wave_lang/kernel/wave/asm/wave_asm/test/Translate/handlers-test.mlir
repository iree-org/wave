// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s

// Test that the MLIR-to-WAVEASM handlers emit the correct ops

// CHECK: waveasm.program @mma_test target = <"gfx942">

// Test gpu.thread_id -> v_mbcnt_lo/hi sequence
// CHECK: waveasm.constant -1
// CHECK: waveasm.constant 0
// CHECK: waveasm.v_mbcnt_lo_u32_b32
// CHECK: waveasm.v_mbcnt_hi_u32_b32

// Test arith.constant translation
// CHECK: waveasm.constant 0

// CHECK: waveasm.s_endpgm

gpu.module @test_kernel {
  gpu.func @mma_test() kernel {
    %tid = gpu.thread_id x
    %c0 = arith.constant 0 : i32
    gpu.return
  }
}
