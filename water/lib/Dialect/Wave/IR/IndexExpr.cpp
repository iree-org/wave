// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

AffineMap wave::alignMapSymbols(AffineMap map, ArrayRef<Attribute> symbols,
                                ArrayRef<Attribute> allSymbols) {
  assert(map.getNumDims() == 0 && "maps should not involve dimensions");

  MLIRContext *ctx = map.getContext();
  unsigned newNumSyms = allSymbols.size();

  auto newSymbols = llvm::map_to_vector(symbols, [&](Attribute symbol) {
    auto it = llvm::find(allSymbols, symbol);
    assert(it != allSymbols.end() && "expected symbol to be in allSymbols");
    unsigned position = std::distance(allSymbols.begin(), it);
    return getAffineSymbolExpr(position, ctx);
  });

  SmallVector<AffineExpr> remapped;
  remapped.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    remapped.push_back(expr.replaceSymbols(newSymbols));

  return mlir::AffineMap::get(/*dimCount=*/0, newNumSyms, remapped, ctx);
}
