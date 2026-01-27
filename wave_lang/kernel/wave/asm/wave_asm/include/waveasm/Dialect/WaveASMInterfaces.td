// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMINTERFACES
#define WaveASM_DIALECT_WAVEASMINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// WaveASM Operation Traits
//
// These traits categorize operations for passes like CSE, liveness, etc.
// Using traits instead of string matching provides type safety and
// allows the compiler to verify correctness at compile time.
//===----------------------------------------------------------------------===//

// Trait for arithmetic/ALU operations that are pure and CSE-eligible
def WaveASM_ArithmeticOp : NativeOpTrait<"ArithmeticOp">;

// Trait for memory operations (loads/stores)
def WaveASM_MemoryOp : NativeOpTrait<"MemoryOp">;

// Trait for control flow operations (branches, barriers)
def WaveASM_ControlFlowOp : NativeOpTrait<"ControlFlowOp">;

// Trait for special register operations (M0, VCC, EXEC)
def WaveASM_SpecialRegOp : NativeOpTrait<"SpecialRegOp">;

#endif // WaveASM_DIALECT_WAVEASMINTERFACES
