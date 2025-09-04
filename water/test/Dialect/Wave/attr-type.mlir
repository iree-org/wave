// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.symbol<"A">
func.func private @attr() attributes { test.foo = #wave.symbol<"A"> }

// CHECK: !wave.tensor<[@A, @B] of bf16>
func.func private @foo() -> !wave.tensor<[@A, @B] of bf16>

// CHECK: !wave.tensor<any of bf16>
func.func private @unspecified_tensor() -> !wave.tensor<any of bf16>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space() -> !wave.tensor<any of i32, <global>>

// CHECK: !wave.tensor<any of i32, <global>>
func.func private @address_space_full() -> !wave.tensor<any of i32, #wave.address_space<global>>

// CHECK: !wave.tensor<any of i8>
func.func private @address_space_default() -> !wave.tensor<any of i8, <unspecified>>
