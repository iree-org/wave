// RUN: water-opt %s --water-gpu-module-to-binary="lld-path=/invalid/path" --verify-diagnostics

module attributes {gpu.container_module} {
  // expected-error @below {{and explicit lld path does not point to a valid file}}
  gpu.module @kernel_module [#rocdl.target<chip = "gfx942", O = 2>] {
    llvm.func @simple_kernel(%arg0: f32) attributes {gpu.kernel} {
      llvm.return
    }
  }
}
