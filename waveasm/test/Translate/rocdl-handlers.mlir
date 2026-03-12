// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: ROCDL and AMDGPU op handlers that were previously silently dropped.
// Without these handlers, s_barrier, s_setprio, and s_waitcnt vmcnt(N)
// instructions were missing from generated assembly.

// CHECK: waveasm.program @sync_ops_test

// rocdl.s.setprio -> waveasm.s_setprio
// CHECK: waveasm.s_setprio 1

// rocdl.s.barrier -> waveasm.s_barrier
// CHECK: waveasm.s_barrier

// rocdl.s.setprio -> waveasm.s_setprio
// CHECK: waveasm.s_setprio 0

// amdgpu.memory_counter_wait with load(10) -> waveasm.s_waitcnt_vmcnt 10
// CHECK: waveasm.s_waitcnt_vmcnt 10

// amdgpu.memory_counter_wait with ds(0) -> waveasm.s_waitcnt_lgkmcnt 0
// CHECK: waveasm.s_waitcnt_lgkmcnt 0

// CHECK: waveasm.s_endpgm

module {
  gpu.module @test_sync_ops {
    gpu.func @sync_ops_test() kernel {
      rocdl.s.setprio 1
      rocdl.s.barrier
      rocdl.s.setprio 0
      amdgpu.memory_counter_wait load(10)
      amdgpu.memory_counter_wait ds(0)
      gpu.return
    }
  }
}
