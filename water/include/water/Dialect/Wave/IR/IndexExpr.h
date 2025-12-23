// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_INDEXEXPR_H
#define WATER_DIALECT_WAVE_IR_INDEXEXPR_H

#include "mlir/IR/Attributes.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class AffineMap;
class Attribute;
} // namespace mlir

namespace wave {

// Aggregate all symbols from a range of ranges of attributes into a single
// range of attributes without duplicates.
template <typename RangeT>
static inline void
aggregateAllSymbols(RangeT &&symbolLists,
                    llvm::SmallVectorImpl<mlir::Attribute> &symbols) {
  static_assert(
      std::is_same_v<mlir::Attribute,
                     std::decay_t<decltype(*symbolLists.begin()->begin())>>,
      "expected symbols to be a range of ranges of attributes");
  llvm::SetVector<mlir::Attribute> allSymbols;
  for (auto &&symbols : symbolLists)
    allSymbols.insert_range(symbols);
  symbols = allSymbols.takeVector();
}

// Given the `symbols` associated with symbols of the affine `map`, update the
// map so the positions of symbols become aligned with `allSymbols`. For
// example, given a map [s0, s1]() -> (s0 + s1) currently using symbols ["a",
// "b"], and allSymbols = ["_", "b", "c", "a"], the resulting map will be
// [s0, s1, s2, s3]() -> (s3 + s1).
mlir::AffineMap alignMapSymbols(mlir::AffineMap map,
                                llvm::ArrayRef<mlir::Attribute> symbols,
                                llvm::ArrayRef<mlir::Attribute> allSymbols);

// Create an index mapping induced by the given workgroup constraint. Combine
// it with the base index mapping if provided.
WaveIndexMappingAttr
applyConstraint(WorkgroupConstraintAttr constraint,
                WaveIndexMappingAttr baseMapping = nullptr);

// Create an index mapping induced by the given tiling constraint. Combine it
// with the base index mapping if provided.
WaveIndexMappingAttr
applyConstraint(TilingConstraintAttr constraint,
                WaveIndexMappingAttr baseMapping = nullptr);

// Create an index mapping induced by the given constraint. Combine it with the
// base index mapping if provided. Call `applyConstraint` if the specific kind
// of constraint is already known.
WaveIndexMappingAttr
applyConstraintGeneric(mlir::Attribute constraint,
                       WaveIndexMappingAttr baseMapping = nullptr);

} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_INDEXEXPR_H
