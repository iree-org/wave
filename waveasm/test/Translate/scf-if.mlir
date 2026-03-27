// RUN: waveasm-translate --target=gfx942 %s | FileCheck %s
//
// Test: scf.if translation to waveasm.if

//===----------------------------------------------------------------------===//
// Test 1: Simple if-then-else with computed results
//===----------------------------------------------------------------------===//

module {
  gpu.module @test_scf_if {
    // CHECK-LABEL: waveasm.program @simple_if
    gpu.func @simple_if() kernel {
      %tid = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      %cond = arith.cmpi ult, %tid, %c10 : index

      // CHECK: = waveasm.if {{.*}} -> !waveasm.vreg {
      // CHECK:   waveasm.yield
      // CHECK: } else {
      // CHECK:   waveasm.yield
      // CHECK: }
      %result = scf.if %cond -> index {
        %sum = arith.addi %tid, %c1 : index
        scf.yield %sum : index
      } else {
        %diff = arith.subi %tid, %c1 : index
        scf.yield %diff : index
      }
      gpu.return
    }
  }
}

//===----------------------------------------------------------------------===//
// Test 2: If-then-else nested inside scf.for loop
//===----------------------------------------------------------------------===//

module {
  gpu.module @test_scf_if_in_loop {
    // CHECK-LABEL: waveasm.program @if_in_loop
    gpu.func @if_in_loop() kernel {
      %tid = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c10 = arith.constant 10 : index

      %cond = arith.cmpi ult, %tid, %c10 : index

      // CHECK: waveasm.loop
      %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> index {
        // CHECK: = waveasm.if {{.*}} -> !waveasm.vreg {
        %step = scf.if %cond -> index {
          %v = arith.addi %acc, %c1 : index
          scf.yield %v : index
        } else {
          %v = arith.addi %acc, %c0 : index
          scf.yield %v : index
        }
        scf.yield %step : index
      }
      gpu.return
    }
  }
}

//===----------------------------------------------------------------------===//
// Test 3: If without results (void if)
//===----------------------------------------------------------------------===//

module {
  gpu.module @test_void_if {
    // CHECK-LABEL: waveasm.program @void_if
    gpu.func @void_if() kernel {
      %tid = gpu.thread_id x
      %c10 = arith.constant 10 : index

      %cond = arith.cmpi ult, %tid, %c10 : index

      // CHECK: waveasm.if
      // CHECK: } else {
      // CHECK: }
      scf.if %cond {
        // empty
      } else {
        // empty
      }
      gpu.return
    }
  }
}
