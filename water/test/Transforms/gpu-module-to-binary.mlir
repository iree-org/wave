// RUN: water-opt %s --water-gpu-module-to-binary | FileCheck %s

// CHECK-LABEL: module
module attributes {gpu.container_module} {
  // Simple test to verify the pass stub runs without errors
  // TODO: Add actual gpu.module operations once serialization is implemented

  func.func @dummy() {
    return
  }
}
