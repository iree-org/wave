// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVETYPES_H
#define WATER_DIALECT_WAVE_IR_WAVETYPES_H

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Types.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveTypes.h.inc"

namespace wave {
// Returns true if the given type is a Wave dialect tensor definitely living in
// the register address space.
[[maybe_unused]] static inline bool isaTensorInRegister(mlir::Type type) {
  auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
  return tensorType &&
         tensorType.getAddressSpaceValue() == wave::WaveAddressSpace::Register;
}
}; // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVETYPES_H
