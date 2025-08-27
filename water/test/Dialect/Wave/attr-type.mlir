// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.symbol<"A">
func.func private @attr() attributes { test.foo = #wave.symbol<"A"> }

// CHECK: !wave.tensor<[@A, @B] of bf16>
func.func private @foo() -> !wave.tensor<[@A, @B] of bf16>

// CHECK: !wave.tensor<any of bf16>
func.func private @unspecified_tensor() -> !wave.tensor<any of bf16>
