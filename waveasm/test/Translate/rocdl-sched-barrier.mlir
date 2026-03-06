// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: rocdl.sched.barrier handler emits waveasm.raw "s_sched_barrier".

// CHECK-LABEL: waveasm.program @sched_barrier_test

// rocdl.sched.barrier 0 -> s_sched_barrier 0x0
// CHECK: waveasm.raw "s_sched_barrier 0x0"

// rocdl.sched.barrier 1 -> s_sched_barrier 0x1
// CHECK: waveasm.raw "s_sched_barrier 0x1"

// rocdl.sched.barrier 255 -> s_sched_barrier 0xFF
// CHECK: waveasm.raw "s_sched_barrier 0xFF"

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
