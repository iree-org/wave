// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

std::optional<llvm::SmallVector<int64_t>>
wave::resolveSymbolNames(llvm::ArrayRef<wave::WaveSymbolAttr> names,
                         wave::WaveHyperparameterAttr hyper) {
  if (!hyper)
    return std::nullopt;
  // Collect concrete values for each symbol in stored order.
  llvm::SmallVector<int64_t> symVals;
  symVals.reserve(names.size());
  for (auto symbol : names) {
    auto value = hyper.getSymbolValue(symbol.getName());
    if (!value)
      return std::nullopt;
    symVals.push_back(*value);
  }
  return symVals;
}

std::optional<llvm::SmallVector<int64_t>>
wave::evaluateMapWithSymbols(AffineMap map,
                             llvm::ArrayRef<int64_t> symbolValues) {
  // Build AffineExpr replacements for symbols: s_i → const(symVals[i]).
  MLIRContext *ctx = map.getContext();
  llvm::SmallVector<AffineExpr> symRepls;
  symRepls.reserve(symbolValues.size());
  for (int64_t v : symbolValues)
    symRepls.push_back(getAffineConstantExpr(v, ctx));

  // For each result expr: substitute symbols and fold
  llvm::SmallVector<int64_t> out;
  out.reserve(map.getNumResults());
  for (AffineExpr affine : map.getResults()) {
    AffineExpr sub = affine.replaceSymbols(symRepls);
    sub = simplifyAffineExpr(sub, /*numDims=*/0, /*numSymbols=*/0);
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(sub)) {
      out.push_back(c.getValue());
    } else {
      return std::nullopt;
    }
  }
  return out;
}
