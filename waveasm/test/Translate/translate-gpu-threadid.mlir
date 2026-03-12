// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s

// Test gpu.thread_id translation to v_mbcnt instructions

module {
  gpu.module @test_kernel {
    gpu.func @threadid_x_kernel() kernel {
      // CHECK: waveasm.program @threadid_x_kernel
      // CHECK: waveasm.v_mbcnt_lo_u32_b32
      // CHECK: waveasm.v_mbcnt_hi_u32_b32
      %tid_x = gpu.thread_id x
      gpu.return
    }
  }
}
