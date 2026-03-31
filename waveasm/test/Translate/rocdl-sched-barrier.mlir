// RUN: waveasm-translate %s | FileCheck %s
//
// Test: rocdl.sched.barrier lowers to a comment for the ASM backend.

// CHECK-LABEL: waveasm.program @sched_barrier_test

// rocdl.sched.barrier 0 -> waveasm.comment preserving the source position
// CHECK: waveasm.comment "s_sched_barrier 0x0 (not emitted)"

// rocdl.sched.barrier 1 -> waveasm.comment preserving the source position
// CHECK: waveasm.comment "s_sched_barrier 0x1 (not emitted)"

// rocdl.sched.barrier 255 -> waveasm.comment preserving the source position
// CHECK: waveasm.comment "s_sched_barrier 0xFF (not emitted)"

// CHECK: waveasm.s_endpgm

module {
  gpu.module @test_sched_barrier {
    gpu.func @sched_barrier_test() kernel {
      rocdl.sched.barrier 0
      rocdl.sched.barrier 1
      rocdl.sched.barrier 255
      gpu.return
    }
  }
}
