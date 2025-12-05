// RUN: rm -rf %t
// RUN: water-opt %s --water-gpu-module-to-binary="dump-intermediates=%t" | FileCheck %s
// RUN: test -f %t/kernel_module_original.ll
// RUN: test -f %t/kernel_module_linked.ll
// RUN: test -f %t/kernel_module_optimized.ll
// RUN: test -f %t/kernel_module.s
// RUN: test -f %t/kernel_module.hsaco

// Test that the pass dumps intermediate compilation files when dump-intermediates is specified

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
