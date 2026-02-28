// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test amdgpu.lds_barrier translation

module {
  gpu.module @test_barrier {
    // CHECK: waveasm.program @barrier_kernel
    gpu.func @barrier_kernel() kernel {
      // CHECK: waveasm.s_waitcnt_lgkmcnt 0
      // CHECK: waveasm.s_barrier
      amdgpu.lds_barrier

      gpu.return
    }
  }
}
