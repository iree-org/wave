// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s

// Test gpu.thread_id translation: single-wave uses v_mbcnt,
// multi-wave (wgX > 64) uses v_and_b32 on hardware v0.

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

  // Multi-wave kernel: 256 threads in X dimension requires reading
  // workitem_id_x from v0 instead of computing lane_id via v_mbcnt.
  func.func @threadid_x_multiwave() attributes {
    translation_info = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
  } {
    // CHECK: waveasm.program @threadid_x_multiwave
    // CHECK: waveasm.precolored.vreg
    // CHECK: waveasm.v_and_b32
    // CHECK-NOT: waveasm.v_mbcnt_lo_u32_b32
    %tid_x = gpu.thread_id x
    return
  }
}
