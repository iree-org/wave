// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s

// Test LDS barrier handler translation

// CHECK: waveasm.program @barrier_kernel

// Test gpu.thread_id
// CHECK: waveasm.v_mbcnt_lo_u32_b32
// CHECK: waveasm.v_mbcnt_hi_u32_b32

// Test amdgpu.lds_barrier -> s_waitcnt + s_barrier
// CHECK: waveasm.s_waitcnt_lgkmcnt
// CHECK: waveasm.s_barrier

// CHECK: waveasm.s_endpgm

gpu.module @barrier_module {
  gpu.func @barrier_kernel() kernel {
    // Get thread ID
    %tid = gpu.thread_id x

    // Some constant
    %c0 = arith.constant 0 : i32

    // LDS barrier
    amdgpu.lds_barrier

    gpu.return
  }
}
