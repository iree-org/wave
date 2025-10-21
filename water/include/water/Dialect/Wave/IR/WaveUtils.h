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
/// Return the position of the dimension that is vectorized based on the index
/// sequence. The dimension with the largest step is considered to be
/// vectorized. In case of a tie, take the dimension that is farther in the
/// index dictionary, which is secretly a list. Return failure when the index
/// sequence step cannot be evaluated statically.
std::optional<uint64_t>
getPositionOfVectorizedDim(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                           mlir::DictionaryAttr indexDict,
                           wave::WaveHyperparameterAttr hyper);

// Return the vector shape implied by the index sequence and hyperparameteters,
// i.e., the step expression of the index sequence evaluated using the
// hyperparameter values. The step may be indicated as ShapedType::kDynamic if
// it cannot be fully evaluated.
llvm::SmallVector<int64_t>
getUncollapsedVectorShape(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                          mlir::DictionaryAttr indexDict,
                          wave::WaveHyperparameterAttr hyper);

/// Resolve named Wave symbols to concrete integer values using the
/// hyperparameter table.
std::optional<llvm::SmallVector<int64_t>>
resolveSymbolNames(llvm::ArrayRef<wave::WaveSymbolAttr> names,
                   wave::WaveHyperparameterAttr hyper);

/// Substitute named symbol values used in the affine map by the constant values
/// defined in the hyperparameter list then evaluate the expressions to get
/// concrete integer results. Return nullopt if the substitution doesn't yield
/// constant results, in particular, if some symbols are not defined.
std::optional<llvm::SmallVector<int64_t>>
evaluateMapWithHyperparams(mlir::AffineMap map,
                           llvm::ArrayRef<wave::WaveSymbolAttr> symbolNames,
                           wave::WaveHyperparameterAttr hyperparams);
} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVEUTILS_H
