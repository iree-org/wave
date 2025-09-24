// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/Dialect.h"

#include "water/Dialect/Wave/IR/WaveDialect.cpp.inc"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

void wave::WaveDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"
      >();
  registerTypes();
}

// Verify whether all types from the given range exclusively use symbols
// defined in the hyperparameter attribute, report errors otherwise using the
// provided callback. Collect used symbols into the given set for future checks.
static llvm::LogicalResult verifyTypeRangeHyperparamUses(
    wave::WaveHyperparameterAttr hyperparam, mlir::TypeRange types,
    llvm::StringSet<> &usedSymbols,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  for (auto [i, type] : llvm::enumerate(types)) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || !tensorType.getFullySpecified())
      continue;

    // TODO: we want symbol attrs rather than strings in hyperparam.
    for (wave::WaveSymbolAttr symbol : tensorType.getShape()) {
      usedSymbols.insert(symbol.getName());
      if (hyperparam.getMapping().contains(symbol.getName()))
        continue;

      mlir::InFlightDiagnostic diag =
          emitError() << "type #" << i << " uses symbolic value " << symbol
                      << " not provided as a hyperparameter";
      std::string availableSymbols =
          llvm::join(llvm::map_range(hyperparam.getMapping(),
                                     [](const mlir::NamedAttribute namedAttr) {
                                       return namedAttr.getName().getValue();
                                     }),
                     ", ");
      diag.attachNote() << "available symbols: " << availableSymbols;

      // TODO: we will want a special value of the hyperparameter that indicates
      // whether we want to turn the symbol into a dynamic value accepted by the
      // generated function.
      diag.attachNote() << "NYI support for symbol lowering";
      return diag;
    }
  }
  return llvm::success();
}

llvm::LogicalResult
wave::WaveDialect::verifyOperationAttribute(mlir::Operation *op,
                                            mlir::NamedAttribute attr) {
  if (attr.getName() == kNormalFormAttrName) {
    if (auto enumAttr = llvm::dyn_cast<WaveNormalFormAttr>(attr.getValue())) {
      return detail::verifyNormalFormAttr(op, enumAttr.getValue(),
                                          /*emitDiagnostics=*/true);
    }
    return op->emitError() << attr.getName() << " expects a WaveNormalFormAttr";
  }
  if (attr.getName() == kHyperparameterAttrName) {
    auto hyperparams =
        llvm::dyn_cast<wave::WaveHyperparameterAttr>(attr.getValue());
    if (!hyperparams) {
      return op->emitError()
             << attr.getName() << " expects a WaveHyperparameterAttr";
    }

    // TODO: consider a mode where parameters can be union'ed, but not
    // redefined. There are passes that currently assume a single set of
    // hyperparameters.
    for (mlir::Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kHyperparameterAttrName)) {
        mlir::InFlightDiagnostic diag =
            op->emitError()
            << "defines hyperparameters when its ancestor already had";
        diag.attachNote(parent->getLoc()) << "ancestor";
        return diag;
      }
    }

    llvm::StringSet<> usedSymbols;
    mlir::WalkResult walkResult = op->walk([&](mlir::Operation *op) {
      if (llvm::failed(verifyTypeRangeHyperparamUses(
              hyperparams, op->getResultTypes(), usedSymbols,
              [&]() { return op->emitOpError() << "result "; }))) {
        return mlir::WalkResult::interrupt();
      }

      for (mlir::Region &region : op->getRegions()) {
        // Can't use llvm::enumerate because of nested lambda capture defect.
        unsigned blockNo = 0;
        for (mlir::Block &block : region) {
          if (llvm::failed(verifyTypeRangeHyperparamUses(
                  hyperparams, block.getArgumentTypes(), usedSymbols, [&]() {
                    return op->emitOpError()
                           << "region #" << region.getRegionNumber()
                           << " block #" << blockNo << " argument ";
                  }))) {
            return mlir::WalkResult::interrupt();
          }
          ++blockNo;
        }
      }

      return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return llvm::failure();

    llvm::SmallVector<mlir::StringRef> unusedNames;
    for (const mlir::NamedAttribute &namedAttr :
         hyperparams.getMapping().getValue()) {
      if (!usedSymbols.contains(namedAttr.getName().getValue()))
        unusedNames.push_back(namedAttr.getName().getValue());
    }
    if (!unusedNames.empty()) {
      // XXX: cannot use op->emitWarning as that triggers the op verifier
      // leading to infinite recursion.
      emitWarning(op->getLoc())
          << "unused hyperparameter"
          << (llvm::hasSingleElement(unusedNames) ? "" : "s") << ": "
          << llvm::join(unusedNames, ", ");
    }

    return llvm::success();
  }
  if (attr.getName() == kElementsPerThreadAttrName) {
    if (!llvm::isa<mlir::IntegerAttr>(attr.getValue())) {
      return op->emitError() << attr.getName() << " expects an IntegerAttr";
    }
    return llvm::success();
  }
  return op->emitError() << "unexpected wave dialect attribute "
                         << attr.getName() << " = " << attr.getValue();
}
