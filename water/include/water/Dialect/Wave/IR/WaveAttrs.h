// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEATTRS_H
#define WATER_DIALECT_WAVE_IR_WAVEATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"

namespace wave {

struct WaveMmaSpec {
  int64_t m;
  int64_t n;
  int64_t k;
  mlir::Type aType;
  mlir::Type bType;
  mlir::Type accType;
};

} // namespace wave

#include "water/Dialect/Wave/IR/WaveEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.h.inc"

namespace wave {

namespace detail {
/// Check that all symbols in the expression are of the types provided as
/// template arguments. Does NOT emit diagnostics.
template <typename... InputTypes>
bool allExprSymbolsOfType(WaveExprListAttr expr) {
  return llvm::all_of(expr.getSymbols(), llvm::IsaPred<InputTypes...>);
}

/// Check that all values in the symbol mapping are of the types provided as
/// template arguments. Does NOT emit diagnostics.
template <typename... ValueTypes>
bool areAllSymbolMappingValuesAllowed(WaveSymbolMappingAttr mapping) {
  return llvm::all_of(mapping.getValues(), llvm::IsaPred<ValueTypes...>);
}

/// Check that all expression lists used as values of the mapping have exactly
/// `n` results. Does NOT emit diagnostics. Intended ONLY for use in a trait.
static inline bool
areAllSymbolMappingValuesNResultExprLists(WaveSymbolMappingAttr mapping,
                                          unsigned n) {
  return llvm::all_of(mapping.getValues(), [&](mlir::Attribute attr) {
    return llvm::cast<WaveExprListAttr>(attr).getMap().getNumResults() == n;
  });
}
} // namespace detail

/// Verify that all provided ExprAttr attributes have the same rank. Returns
/// success if all ranks match, failure otherwise.
llvm::LogicalResult
verifyExprAttrsSameRank(llvm::ArrayRef<WaveExprListAttr> exprs);

/// Verify that all provided ExprAttr attributes have no symbols (i.e., they are
/// constant expressions). Returns success if all have zero symbols, failure
/// otherwise.
llvm::LogicalResult
verifyExprAttrsNoSymbols(llvm::ArrayRef<WaveExprListAttr> exprs);

} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVEATTRS_H
