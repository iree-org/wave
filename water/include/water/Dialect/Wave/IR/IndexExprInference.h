// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_INDEXEXPRINFERENCE_H
#define WATER_DIALECT_WAVE_IR_INDEXEXPRINFERENCE_H

#include "water/Dialect/Wave/IR/WaveInterfaces.h"

namespace wave {
void operator<<(mlir::Diagnostic &diag, const IndexExprsLatticeStorage &value);
} // namespace wave

namespace llvm {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const wave::IndexExprsLatticeStorage &value);
} // namespace llvm

namespace wave {
namespace detail {

// Heuristic to stop index expression propagation. It stops propagation of index
// expressions based on the vector shape of the operation from which the index
// expression originally propagates, stored in `sourceVectorShape` field of the
// lattice. Propagation is skipped if:
//
//   1. Propagating towards an Mma operation.
//   2. The source vector shape doesn't cover all symbolic dimensions of the
//      value the lattice is about to be propagated to.
//   3. Any of the symbols already present in the index expression for the
//      target value (which is normally initialized from the target's value
//      shape) corresponds to a unit dimension in the source vector shape IF the
//      rank is less than the number of such non-unit dimensions.
//   4. Any of the non-unit dimensions in the source vector shape is not already
//      included in the target value's index expression IF the rank is greater
//      than or equal to the number of such dimensions.
//
// XXX: conditions 2-4 are carried over from the python prototype and are not
// principled.
bool shouldPropagateIndexExprs(const wave::IndexExprsLatticeStorage &from,
                               const wave::IndexExprsLatticeStorage &to,
                               mlir::Value toValue);

// Build thread-independent index mapping for a single tensor type and append to
// symbolMappings. Used by identity and reduction index expr initialization.
llvm::LogicalResult buildThreadIndependentIndexMappings(
    mlir::Operation *op, mlir::Type type,
    const IndexExprsAnalysisInit &initObject,
    llvm::SmallVectorImpl<std::pair<wave::WaveSymbolAttr, mlir::Attribute>>
        &symbolMappings);

// Create a new vector shape with only the provided symbols present.
wave::WaveSymbolMappingAttr
filterVectorShape(wave::WaveSymbolMappingAttr vectorShape,
                  llvm::ArrayRef<wave::WaveSymbolAttr> symbols);

// Check the index expressions is a concrete value rather lattice top/bottom and
// append it to the indexExprs list. If it is lattice top/bottom, report an
// error and return failure.
llvm::LogicalResult
checkAndAppendIndexExpr(mlir::Location loc,
                        const IndexExprsLatticeStorage &expr,
                        const llvm::Twine &description,
                        llvm::SmallVectorImpl<mlir::Attribute> &indexExprs);

} // namespace detail
} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_INDEXEXPRINFERENCE_H
