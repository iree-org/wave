// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/IndexExpr.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace wave {
AffineMap alignMapSymbols(AffineMap map, ArrayRef<Attribute> symbols,
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

  return AffineMap::get(/*dimCount=*/0, newNumSyms, remapped, ctx);
}
} // namespace wave

// Get the (indexed) affine symbol expression corresponding to the given symbol,
// insert it into the list if it isn't already present.
static AffineExpr
getOrInsertSymbolExpr(Attribute symbol,
                      llvm::SmallVectorImpl<Attribute> &symbolNames) {
  auto it = llvm::find(symbolNames, symbol);
  unsigned position = [&] {
    if (it != symbolNames.end())
      return static_cast<unsigned>(std::distance(symbolNames.begin(), it));
    symbolNames.push_back(symbol);
    return static_cast<unsigned>(symbolNames.size() - 1);
  }();
  return getAffineSymbolExpr(position, symbol.getContext());
}

// Helper function to create an index mapping from a symbol expression and
// constraint tile size. Combines with base mapping if provided.
static wave::WaveIndexMappingAttr createIndexMappingFromSymbolExpr(
    AffineExpr symbolExpr, ArrayRef<Attribute> symbols, AffineMap tileSizeMap,
    MLIRContext *context, wave::WaveIndexMappingAttr baseMapping) {
  assert(tileSizeMap.getNumResults() == 1 &&
         "expected a single result expression in affine map");
  AffineMap map = AffineMap::get(
      /*dimCount=*/0, symbols.size(), symbolExpr * tileSizeMap.getResult(0));
  if (baseMapping == nullptr)
    return wave::WaveIndexMappingAttr::get(context, symbols, map, AffineMap(),
                                           AffineMap());

  SmallVector<Attribute> allSymbols;
  wave::aggregateAllSymbols(
      std::initializer_list<ArrayRef<Attribute>>{baseMapping.getSymbols(),
                                                 symbols},
      allSymbols);

  AffineMap baseStart = wave::alignMapSymbols(
      baseMapping.getStart(), baseMapping.getSymbols(), allSymbols);
  AffineMap baseStep = wave::alignMapSymbols(
      baseMapping.getStep(), baseMapping.getSymbols(), allSymbols);
  AffineMap baseStride = wave::alignMapSymbols(
      baseMapping.getStride(), baseMapping.getSymbols(), allSymbols);
  map = wave::alignMapSymbols(map, symbols, allSymbols);
  map = AffineMap::get(/*dimCount=*/0, allSymbols.size(),
                       baseStart.getResult(0) + map.getResult(0));
  return wave::WaveIndexMappingAttr::get(context, allSymbols, map, baseStep,
                                         baseStride);
}

namespace wave {

WaveIndexMappingAttr applyConstraint(WorkgroupConstraintAttr constraint,
                                     WaveIndexMappingAttr baseMapping) {
  llvm::SmallVector<Attribute> symbols =
      llvm::map_to_vector(constraint.getTileSize().getSymbols(),
                          [](Attribute symbol) { return symbol; });

  MLIRContext *context = constraint.getContext();

  // TODO: we should just use workgroup attributes in expressions directly.
  WaveIndexSymbol symbol = [&] {
    switch (constraint.getWorkgroupDim().getValue()) {
    case WaveWorkgroupDim::X:
      return WaveIndexSymbol::WORKGROUP_0;
    case WaveWorkgroupDim::Y:
      return WaveIndexSymbol::WORKGROUP_1;
    case WaveWorkgroupDim::Z:
      return WaveIndexSymbol::WORKGROUP_2;
    }
    llvm::report_fatal_error("unsupported workgroup dimension");
  }();
  AffineExpr symbolExpr =
      getOrInsertSymbolExpr(WaveIndexSymbolAttr::get(context, symbol), symbols);

  return createIndexMappingFromSymbolExpr(symbolExpr, symbols,
                                          constraint.getTileSize().getMap(),
                                          context, baseMapping);
}

WaveIndexMappingAttr applyConstraint(TilingConstraintAttr constraint,
                                     WaveIndexMappingAttr baseMapping) {
  llvm::SmallVector<Attribute> symbols =
      llvm::map_to_vector(constraint.getTileSize().getSymbols(),
                          [](Attribute symbol) { return symbol; });

  MLIRContext *context = constraint.getContext();
  AffineExpr symbolExpr = getOrInsertSymbolExpr(
      WaveIterSymbolAttr::get(context, constraint.getDim().getName()), symbols);

  return createIndexMappingFromSymbolExpr(symbolExpr, symbols,
                                          constraint.getTileSize().getMap(),
                                          context, baseMapping);
}

WaveIndexMappingAttr applyConstraintGeneric(Attribute constraint,
                                            WaveIndexMappingAttr baseMapping) {
  return llvm::TypeSwitch<Attribute, WaveIndexMappingAttr>(constraint)
      .Case<WorkgroupConstraintAttr, TilingConstraintAttr>(
          [&](auto constraint) {
            // This double dispatching is necessary in absence of interfaces to
            // dispatch to a class method based on a specific type.
            return applyConstraint(constraint, baseMapping);
          })
      .Default([&](Attribute constraint) { return nullptr; });
}

} // namespace wave
