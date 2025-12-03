// RUN: water-opt %s --water-gpu-module-to-binary | FileCheck %s

// Test that the pass converts a gpu.module with ROCDL target to a gpu.binary
// The gpu.module contains already-lowered LLVM IR
//
// This test requires ROCm to be installed. It uses mlir::ROCDL::getROCMPath()
// which checks ROCM_PATH, ROCM_ROOT, ROCM_HOME environment variables or uses
// the CMake-detected path.

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-NOT: gpu.module
  // CHECK: gpu.binary @kernel_module [#gpu.object<#rocdl.target<chip = "gfx942">, bin =
  gpu.module @kernel_module [#rocdl.target<chip = "gfx942", O = 2>] {
    llvm.func @simple_kernel(%arg0: f32) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
