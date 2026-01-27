// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMINTERFACES_H
#define WaveASM_DIALECT_WAVEASMINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

//===----------------------------------------------------------------------===//
// WaveASM Operation Traits
//
// These traits categorize operations for passes like CSE, liveness, etc.
// Using traits instead of string matching provides type safety.
// Defined in mlir::OpTrait namespace as required by NativeOpTrait<>.
//===----------------------------------------------------------------------===//

/// Trait for arithmetic/ALU operations that are pure and CSE-eligible.
/// Includes VALU and SALU operations like v_add_u32, s_add_u32, etc.
template <typename ConcreteType>
class ArithmeticOp : public TraitBase<ConcreteType, ArithmeticOp> {
};

/// Trait for memory operations (loads/stores).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class MemoryOp : public TraitBase<ConcreteType, MemoryOp> {
};

/// Trait for control flow operations (branches, barriers, waitcnt).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class ControlFlowOp : public TraitBase<ConcreteType, ControlFlowOp> {
};

/// Trait for special register operations (M0, VCC, EXEC).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class SpecialRegOp : public TraitBase<ConcreteType, SpecialRegOp> {
};

} // namespace OpTrait
} // namespace mlir

#endif // WaveASM_DIALECT_WAVEASMINTERFACES_H
