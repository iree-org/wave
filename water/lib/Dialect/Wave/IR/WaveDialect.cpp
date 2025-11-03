// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/Dialect.h"

#include "water/Dialect/Wave/IR/WaveDialect.cpp.inc"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <optional>

void wave::WaveDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"
      >();
  registerTypes();
}

// Attach a note to the diagnostic listing the symbol names available in the
// hyperparameter set.
static void
attachAvailableSymbolsNote(mlir::InFlightDiagnostic &diag,
                           wave::WaveHyperparameterAttr hyperparam) {
  std::string availableSymbols =
      llvm::join(llvm::map_range(hyperparam.getMapping(),
                                 [](const mlir::NamedAttribute namedAttr) {
                                   return namedAttr.getName().getValue();
                                 }),
                 ", ");
  diag.attachNote() << "available symbols: " << availableSymbols;
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
      attachAvailableSymbolsNote(diag, hyperparam);

      // TODO: we will want a special value of the hyperparameter that indicates
      // whether we want to turn the symbol into a dynamic value accepted by the
      // generated function.
      diag.attachNote() << "NYI support for symbol lowering";
      return diag;
    }
  }
  return llvm::success();
}

// Verify whether occurrence of Wave symbols, either as attributes or as string
// names in relevant dictionaries, reference symbols listed as hyperparameters.
// Report errors otherwise using the provided callback. Collect used symbols
// into the given set for future checks.
static llvm::LogicalResult verifyAttributeHyperparamUses(
    wave::WaveHyperparameterAttr hyperparam,
    const mlir::NamedAttribute &namedAttr, llvm::StringSet<> &usedSymbols,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {

  // TODO: we need a first-class attribute for this mapping, at which point this
  // special-casing will disappear as the walker below would also visit symbols
  // used as dictionary keys.
  if (namedAttr.getName().strref() ==
      wave::WaveDialect::kIndexWaveExprListAttrName) {
    auto attr = namedAttr.getValue();
    // Skip verification if not an array of dictionaries, op-level verifiers
    // will detect this and complain.
    auto arrayDict = llvm::dyn_cast<mlir::ArrayAttr>(attr);
    if (!arrayDict)
      return llvm::success();

    for (mlir::Attribute a : arrayDict) {
      auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(a);
      if (!dict)
        continue;
      for (const mlir::NamedAttribute &entry : dict) {
        usedSymbols.insert(entry.getName().strref());

        if (hyperparam.getMapping().contains(entry.getName().strref()))
          continue;

        mlir::InFlightDiagnostic diag =
            emitError() << "uses symbolic value " << entry.getName()
                        << " not provided as a hyperparameter";
        attachAvailableSymbolsNote(diag, hyperparam);
        return llvm::failure();
      }
    }
  }

  // TODO: somehow get rid of these hardcoded magic names.
  static llvm::SmallVector<llvm::StringRef> fixmeMagicNames = {
      "_GPR_NUM", "_T0", "_T1", "_T2", "_WG0", "_WG1", "_WG2"};

  mlir::WalkResult walkResult =
      namedAttr.getValue().walk([&](wave::WaveSymbolAttr symbolAttr) {
        usedSymbols.insert(symbolAttr.getName());

        if (hyperparam.getMapping().contains(symbolAttr.getName()) ||
            llvm::is_contained(fixmeMagicNames, symbolAttr.getName()) ||
            symbolAttr.getName().starts_with("_ARG"))
          return mlir::WalkResult::advance();

        mlir::InFlightDiagnostic diag = emitError()
                                        << "uses symbolic value " << symbolAttr
                                        << " not provided as a hyperparameter";
        attachAvailableSymbolsNote(diag, hyperparam);
        return mlir::WalkResult::interrupt();
      });
  return mlir::failure(walkResult.wasInterrupted());
}

/// Verify DeviceConstraints, WorkgroupConstraints, WaveConstraints, and
/// TilingConstraints for a given set of hyperparameters. This verifcation
/// assumes that all symbols used in the wave.constraints attributes have a
/// coresponding entry in the hyperparameter attribute.
static llvm::LogicalResult
verifyConstraints(mlir::ArrayAttr constraints,
                  wave::WaveHyperparameterAttr hyperparams,
                  llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::DeviceConstraintAttr>
      deviceConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WorkgroupConstraintAttr>
      workgroupConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WaveConstraintAttr>
      waveConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::TilingConstraintAttr>
      tilingConstraints;

  // collect constraints for each dimension symbol
  for (const mlir::Attribute &attr : constraints) {
    if (auto dev = llvm::dyn_cast<wave::DeviceConstraintAttr>(attr)) {
      wave::WaveSymbolAttr dim = dev.getDim();
      auto [it, inserted] = deviceConstraints.try_emplace(dim, dev);
      if (!inserted) {
        return emitError() << "more than one device constraint for dimension: "
                           << dim;
      }
    } else if (auto wg = llvm::dyn_cast<wave::WorkgroupConstraintAttr>(attr)) {
      wave::WaveSymbolAttr dim = wg.getDim();
      auto [it, inserted] = workgroupConstraints.try_emplace(dim, wg);
      if (!inserted) {
        return emitError()
               << "more than one workgroup constraint for dimension: " << dim;
      }
    } else if (auto wave = llvm::dyn_cast<wave::WaveConstraintAttr>(attr)) {
      wave::WaveSymbolAttr dim = wave.getDim();
      auto [it, inserted] = waveConstraints.try_emplace(dim, wave);
      if (!inserted) {
        return emitError() << "more than one wave constraint for dimension: "
                           << dim;
      }
    } else if (auto tile = llvm::dyn_cast<wave::TilingConstraintAttr>(attr)) {
      wave::WaveSymbolAttr dim = tile.getDim();
      auto [it, inserted] = tilingConstraints.try_emplace(dim, tile);
      if (!inserted) {
        return emitError() << "more than one tiling constraint for dimension: "
                           << dim;
      }
    }
  }

  // verify DeviceConstraint
  // * The number of devices should be greater than or equal to one.
  for (auto &&[symbol, constraint] : deviceConstraints) {
    std::optional<llvm::SmallVector<int64_t>> evaluated =
        wave::evaluateMapWithHyperparams(constraint.getTileSize().getMap(),
                                         constraint.getTileSize().getSymbols(),
                                         hyperparams);
    assert(evaluated &&
           "failed to evaluate wave expression for device constraint");
    assert(evaluated->size() == 1 &&
           "invalid evaluation of wave expression for device constraint");

    std::optional<llvm::SmallVector<int64_t>> resolvedDims =
        wave::resolveSymbolNames(symbol, hyperparams);
    assert(resolvedDims && resolvedDims->size() == 1 &&
           "failed to resolve dimesion symbol");

    int64_t resolvedDeviceSize = evaluated->front();
    int64_t resolvedDim = resolvedDims->front();
    int64_t numDevices = resolvedDim / resolvedDeviceSize;
    if (numDevices < 1) {
      return emitError() << "invalid number of devices: " << numDevices
                         << " for dimension: " << symbol;
    }
  }

  // verify WorkgroupConstraint
  // * Each workgroup dimension should have at most one primary constraint
  // assigned.
  // * Each workgroup dimension with a non-primary constraint should have
  // at least one primary constraint.
  // * The number of workgroups should be greater than or equal to one.
  llvm::SmallDenseMap<wave::WaveSymbolAttr, int64_t> resolvedWorkgroupSizes(
      workgroupConstraints.size());
  llvm::SmallDenseSet<wave::WaveWorkgroupDimAttr, 4> assignedDims;
  llvm::SmallDenseSet<wave::WaveWorkgroupDimAttr, 4> needsPrimaryDim;
  for (auto &&[symbol, constraint] : workgroupConstraints) {
    bool isPrimary = constraint.getPrimary();
    wave::WaveWorkgroupDimAttr wgDim = constraint.getWorkgroupDim();

    if (isPrimary) {
      auto [it, inserted] = assignedDims.insert(wgDim);
      if (!inserted) {
        return emitError() << "workgroup dimension " << wgDim
                           << " has more than one primary workgroup constraint";
      }
      needsPrimaryDim.erase(wgDim);
    } else if (!assignedDims.contains(wgDim)) {
      needsPrimaryDim.insert(wgDim);
    }

    std::optional<llvm::SmallVector<int64_t>> evaluated =
        wave::evaluateMapWithHyperparams(constraint.getTileSize().getMap(),
                                         constraint.getTileSize().getSymbols(),
                                         hyperparams);
    assert(evaluated &&
           "failed to evaluate wave expression for workgroup constraint");
    assert(evaluated->size() == 1 &&
           "invalid evaluation of wave expression for workgroup constraint");

    int64_t workgroupSize = evaluated->front();
    resolvedWorkgroupSizes[symbol] = workgroupSize;

    std::optional<llvm::SmallVector<int64_t>> resolvedDims =
        wave::resolveSymbolNames(symbol, hyperparams);
    assert(resolvedDims && resolvedDims->size() == 1 &&
           "failed to resolve dimesion symbol");

    int64_t resolvedDim = resolvedDims->front();
    int64_t numWorkgroups = resolvedDim / workgroupSize;
    if (numWorkgroups < 1) {
      return emitError() << "invalid number of workgroups: " << numWorkgroups
                         << " for dimension: " << symbol;
    }
  }

  for (wave::WaveWorkgroupDimAttr &wgDim : needsPrimaryDim) {
    if (!assignedDims.contains(wgDim)) {
      return emitError()
             << "missing primary workgroup constraint for workgroup dimension: "
             << wgDim;
    }
  }

  // verify WaveConstraint
  // * For each WaveConstraint for a given symbol there should exist a
  // coresponding WorkgroupConstraint with the same dimension symbol.
  // * The number of waves in each workgroup should be greater than or equal to
  // one.
  for (auto &&[symbol, constraint] : waveConstraints) {
    if (!workgroupConstraints.contains(symbol)) {
      return emitError()
             << "missing corresponding workgroup constraint for dimension: "
             << symbol;
    }

    std::optional<llvm::SmallVector<int64_t>> evaluated =
        wave::evaluateMapWithHyperparams(constraint.getTileSize().getMap(),
                                         constraint.getTileSize().getSymbols(),
                                         hyperparams);
    assert(evaluated &&
           "failed to evaluate wave expression for wave constraint");
    assert(evaluated->size() == 1 &&
           "invalid evaluation of wave expression for wave constraint");

    int64_t resolvedWaveSize = evaluated->front();
    int64_t numWaves = resolvedWorkgroupSizes[symbol] / resolvedWaveSize;
    if (numWaves < 1) {
      return emitError() << "invalid number of waves: " << numWaves
                         << " for dimension: " << symbol;
    }
  }

  // verify TilingConstraint
  // * The number of tiles should be greater than or equal to one.
  for (auto &&[symbol, constraint] : tilingConstraints) {
    std::optional<llvm::SmallVector<int64_t>> evaluated =
        wave::evaluateMapWithHyperparams(constraint.getTileSize().getMap(),
                                         constraint.getTileSize().getSymbols(),
                                         hyperparams);
    assert(evaluated &&
           "failed to evaluate wave expression for tiling constraint");
    assert(evaluated->size() == 1 &&
           "invalid evaluation of wave expression for tiling constraint");

    std::optional<llvm::SmallVector<int64_t>> resolvedDims =
        wave::resolveSymbolNames(symbol, hyperparams);
    assert(resolvedDims && resolvedDims->size() == 1 &&
           "failed to resolve dimesion symbol");

    int64_t resolvedTileSize = evaluated->front();
    int64_t resolvedDim = resolvedDims->front();
    int64_t numTiles = resolvedDim / resolvedTileSize;
    if (numTiles < 1) {
      return emitError() << "invalid number of tiles: " << numTiles
                         << " for dimension: " << symbol;
    }
  }

  return llvm::success();
}

llvm::LogicalResult
wave::WaveDialect::verifyOperationAttribute(mlir::Operation *op,
                                            mlir::NamedAttribute attr) {
  // IMPORTANT NOTE: this verifier runs before nested ops have been verified, so
  // it should not assume anything but generic IR well-formedness.
  llvm::StringSet<> usedSymbols;

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

      for (const mlir::NamedAttribute &namedAttr : op->getAttrs()) {
        if (llvm::failed(verifyAttributeHyperparamUses(
                hyperparams, namedAttr, usedSymbols, [&]() {
                  return op->emitOpError()
                         << "attribute " << namedAttr.getName() << " ";
                }))) {
          return mlir::WalkResult::interrupt();
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

  if (attr.getName() == kWaveConstraintsAttrName) {
    mlir::ArrayAttr attrs = llvm::dyn_cast<mlir::ArrayAttr>(attr.getValue());
    bool needsHyperparams = false;

    for (auto attr : attrs) {
      if (!llvm::isa<wave::HardwareConstraintAttr, wave::DeviceConstraintAttr,
                     wave::WorkgroupConstraintAttr, wave::WaveConstraintAttr,
                     wave::TilingConstraintAttr>(attr)) {
        return op->emitError() << attr << " unexpected attribute";
      }
      if (llvm::isa<wave::DeviceConstraintAttr, wave::WorkgroupConstraintAttr,
                    wave::WaveConstraintAttr, wave::TilingConstraintAttr>(
              attr)) {
        needsHyperparams = true;
      }
    }

    // verfify no constraints above
    for (mlir::Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kWaveConstraintsAttrName)) {
        mlir::InFlightDiagnostic diag =
            op->emitError()
            << "defines wave constraints when its ancestor already had";
        diag.attachNote(parent->getLoc()) << "ancestor";
        return diag;
      }
    }

    if (!needsHyperparams) {
      return llvm::success();
    }

    // walk up to find hyperparameters
    wave::WaveHyperparameterAttr hyperparams;
    for (mlir::Operation *parent = op; parent != nullptr && !hyperparams;
         parent = parent->getParentOp()) {
      for (mlir::NamedAttribute attr : parent->getAttrs()) {
        if (attr.getName() != kHyperparameterAttrName)
          continue;

        if (auto params =
                llvm::dyn_cast<wave::WaveHyperparameterAttr>(attr.getValue())) {
          hyperparams = params;
          break;
        }
      }
    }

    if (!hyperparams) {
      return op->emitOpError() << "missing hyperparameters attribute";
    }

    auto emitError = [&]() {
      return op->emitOpError() << "attribute " << attr.getName() << " ";
    };

    // verifyConstraints assumes all used symbols are resolvable
    if (llvm::failed(verifyAttributeHyperparamUses(hyperparams, attr,
                                                   usedSymbols, emitError))) {
      return llvm::failure();
    }

    return verifyConstraints(attrs, hyperparams, emitError);
  }

  return op->emitError() << "unexpected wave dialect attribute "
                         << attr.getName() << " = " << attr.getValue();
}
