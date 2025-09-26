// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_UTILS_H

#define WATER_DIALECT_WAVE_UTILS_H

#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace wave {

/// Resolve named Wave symbols to concrete integer values using the
/// hyperparameter table.
std::optional<llvm::SmallVector<int64_t>>
resolveSymbolNames(llvm::ArrayRef<wave::WaveSymbolAttr> names,
                   wave::WaveHyperparameterAttr hyper);

/// Substitute symbolValues[i] into `map`, fold each result, and return the
/// constant extents.
std::optional<llvm::SmallVector<int64_t>>
evaluateMapWithSymbols(mlir::AffineMap map,
                       llvm::ArrayRef<int64_t> symbolValues);
} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVEUTILS_H
