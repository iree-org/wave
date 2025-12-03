// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "mlir/IR/Attributes.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace mlir;

SmallVector<int64_t>
wave::getUncollapsedVectorShape(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                                mlir::DictionaryAttr indexDict,
                                wave::WaveHyperparameterAttr hyper) {
  return llvm::map_to_vector(shape, [&](wave::WaveSymbolAttr symbol) {
    Attribute entry = indexDict.get(symbol.getName());
    assert(entry && "expected dictionary to contain indices for the shape");
    auto mapAttr = cast<wave::WaveIndexMappingAttr>(entry);
    std::optional<SmallVector<int64_t>> folded =
        wave::evaluateMapWithHyperparams(mapAttr.getStep(),
                                         mapAttr.getSymbols(), hyper);
    if (!folded)
      return ShapedType::kDynamic;
    assert(folded->size() == 1 && "expected single-result map");
    return (*folded)[0];
  });
}

std::optional<uint64_t>
wave::getPositionOfVectorizedDim(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                                 mlir::DictionaryAttr indexDict,
                                 wave::WaveHyperparameterAttr hyper) {
  uint64_t bestIdx = static_cast<uint64_t>(-1);
  std::optional<int64_t> bestSize; // largest constant size seen so far
  for (auto [i, size] :
       llvm::enumerate(getUncollapsedVectorShape(shape, indexDict, hyper))) {
    if (ShapedType::isDynamic(size))
      return std::nullopt;
    if (!bestSize || size >= *bestSize) {
      bestSize = size;
      bestIdx = i;
    }
  }
  assert(bestIdx != static_cast<uint64_t>(-1));
  return bestIdx;
}

std::optional<llvm::SmallVector<int64_t>>
wave::resolveSymbolNames(llvm::ArrayRef<mlir::Attribute> symbols,
                         wave::WaveHyperparameterAttr hyper) {
  if (llvm::any_of(symbols, llvm::IsaPred<WaveIndexSymbolAttr>))
    return std::nullopt;

  if (!hyper)
    return std::nullopt;

  // Collect concrete values for each symbol in stored order.
  llvm::SmallVector<int64_t> symVals;
  symVals.reserve(symbols.size());
  for (Attribute attr : symbols) {
    wave::WaveSymbolAttr symbol = cast<wave::WaveSymbolAttr>(attr);
    auto value = hyper.getSymbolValue(symbol.getName());
    if (!value)
      return std::nullopt;
    symVals.push_back(*value);
  }
  return symVals;
}

std::optional<SmallVector<int64_t>>
wave::evaluateMapWithHyperparams(AffineMap map,
                                 ArrayRef<mlir::Attribute> symbols,
                                 wave::WaveHyperparameterAttr hyperparams) {
  SmallVector<AffineExpr> symReplacements;
  symReplacements.reserve(map.getNumSymbols());
  for (unsigned i = 0, e = map.getNumSymbols(); i < e; ++i) {
    if (llvm::none_of(map.getResults(), [i](AffineExpr expr) {
          return expr.isFunctionOfSymbol(i);
        })) {
      symReplacements.push_back(AffineExpr());
      continue;
    }

    auto symbol = dyn_cast<wave::WaveSymbolAttr>(symbols[i]);
    if (!symbol)
      return std::nullopt;

    std::optional<int64_t> value = hyperparams.getSymbolValue(symbol.getName());
    if (!value)
      return std::nullopt;
    symReplacements.push_back(getAffineConstantExpr(*value, map.getContext()));
  }

  SmallVector<int64_t> out;
  out.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineExpr sub = expr.replaceSymbols(symReplacements);
    sub = simplifyAffineExpr(sub, map.getNumDims(), map.getNumSymbols());
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(sub)) {
      out.push_back(c.getValue());
      continue;
    }

    return std::nullopt;
  }
  return out;
}
