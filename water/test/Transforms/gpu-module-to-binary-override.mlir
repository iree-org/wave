// REQUIRES: rocdl
// RUN: rm -rf %t && mkdir -p %t/dump1 %t/dump2 %t/override
// RUN: water-opt %s --water-gpu-module-to-binary="dump-intermediates=%t/dump1" | FileCheck %s
// RUN: cp %t/dump1/kernel_module_linked.ll %t/override/kernel_module_linked.ll
// RUN: sed -i 's/i32/i64/g' %t/override/kernel_module_linked.ll
// RUN: water-opt %s --water-gpu-module-to-binary="dump-intermediates=%t/dump2 override-intermediates=%t/override" | FileCheck %s
// RUN: grep "define.*i64" %t/dump2/kernel_module_optimized.ll

// Test that override-intermediates works by:
// 1. First run dumps all intermediates to dump1
// 2. Copy linked LLVM IR to override directory and modify it (i32 -> i64)
// 3. Second run uses the modified linked IR from override directory
// 4. Verify the modification appears in the optimized IR (next stage after linked)

// CHECK-LABEL: module attributes {gpu.container_module}
module attributes {gpu.container_module} {
  // CHECK-NOT: gpu.module
  // CHECK: gpu.binary @kernel_module [#gpu.object<#rocdl.target<chip = "gfx942">, bin =
  gpu.module @kernel_module [#rocdl.target<chip = "gfx942", O = 2>] {
    llvm.func @simple_kernel(%arg0: i32) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
