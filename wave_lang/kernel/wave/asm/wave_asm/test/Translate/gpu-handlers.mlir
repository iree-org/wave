// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s
//
// Test GPU dialect handlers: block_dim, grid_dim, lane_id

module {
  gpu.module @test_gpu_handlers {
    // CHECK-LABEL: waveasm.program @test_block_dim
    gpu.func @test_block_dim() kernel {
      // gpu.block_dim returns workgroup dimensions
      // CHECK: waveasm.precolored.sreg
      %bx = gpu.block_dim x
      %by = gpu.block_dim y
      %bz = gpu.block_dim z
      gpu.return
    }

    // CHECK-LABEL: waveasm.program @test_grid_dim
    gpu.func @test_grid_dim() kernel {
      // gpu.grid_dim returns grid dimensions
      // CHECK: waveasm.precolored.sreg
      %gx = gpu.grid_dim x
      %gy = gpu.grid_dim y
      %gz = gpu.grid_dim z
      gpu.return
    }

    // CHECK-LABEL: waveasm.program @test_lane_id
    gpu.func @test_lane_id() kernel {
      // gpu.lane_id uses v_mbcnt pattern
      // CHECK: waveasm.v_mbcnt_lo_u32_b32
      // CHECK: waveasm.v_mbcnt_hi_u32_b32
      %lid = gpu.lane_id
      gpu.return
    }
  }
}
