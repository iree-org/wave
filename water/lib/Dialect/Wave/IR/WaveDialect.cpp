// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/Dialect.h"

#include "water/Dialect/Wave/IR/WaveDialect.cpp.inc"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

using namespace mlir;

namespace {
// Interface to support module-level aliases for WaveIndexMappingAttr.
struct WaveOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<wave::WaveIndexMappingAttr>(attr)) {
      os << "wim";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void wave::WaveDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"
      >();
  registerTypes();
  addInterfaces<WaveOpAsmDialectInterface>();
}

// Attach a note to the diagnostic listing the symbol names available in the
// hyperparameter set.
static void
attachAvailableSymbolsNote(InFlightDiagnostic &diag,
                           wave::WaveHyperparameterAttr hyperparam) {
  std::string availableSymbols = llvm::join(
      llvm::map_range(llvm::make_first_range(hyperparam.getMapping()),
                      [](wave::WaveSymbolAttr sym) -> llvm::StringRef {
                        return sym.getName();
                      }),
      ", ");
  diag.attachNote() << "available symbols: " << availableSymbols;
}

// Verify whether all types from the given range exclusively use symbols
// defined in the hyperparameter attribute, report errors otherwise using the
// provided callback. Collect used symbols into the given set for future checks.
static llvm::LogicalResult verifyTypeRangeHyperparamUses(
    wave::WaveHyperparameterAttr hyperparam, TypeRange types,
    llvm::StringSet<> &usedSymbols,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  for (auto [i, type] : llvm::enumerate(types)) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || !tensorType.getFullySpecified())
      continue;

    for (wave::WaveSymbolAttr symbol : tensorType.getShape()) {
      usedSymbols.insert(symbol.getName());
      if (hyperparam.hasSymbol(symbol))
        continue;

      InFlightDiagnostic diag =
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

// Verify whether occurrences of Wave symbols reference symbols listed as
// hyperparameters. Report errors otherwise using the provided callback. Collect
// used symbols into the given set for future checks.
static llvm::LogicalResult verifyAttributeHyperparamUses(
    wave::WaveHyperparameterAttr hyperparam, const NamedAttribute &namedAttr,
    llvm::StringSet<> &usedSymbols,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  WalkResult walkResult =
      namedAttr.getValue().walk([&](wave::WaveSymbolAttr symbolAttr) {
        usedSymbols.insert(symbolAttr.getName());
        if (hyperparam.hasSymbol(symbolAttr))
          return WalkResult::advance();

        InFlightDiagnostic diag = emitError()
                                  << "uses symbolic value " << symbolAttr
                                  << " not provided as a hyperparameter";
        attachAvailableSymbolsNote(diag, hyperparam);
        return WalkResult::interrupt();
      });
  return failure(walkResult.wasInterrupted());
}

/// Verify DeviceConstraints, WorkgroupConstraints, WaveConstraints, and
/// TilingConstraints for a given set of hyperparameters. This verification
/// assumes that all symbols used in the wave.constraints attributes have a
/// corresponding entry in the hyperparameter attribute.
static llvm::LogicalResult
verifyConstraints(ArrayAttr constraints,
                  wave::WaveHyperparameterAttr hyperparams,
                  llvm::function_ref<InFlightDiagnostic()> emitError) {
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::DeviceConstraintAttr>
      deviceConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WorkgroupConstraintAttr>
      workgroupConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WaveConstraintAttr>
      waveConstraints;
  llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::TilingConstraintAttr>
      tilingConstraints;
  wave::HardwareConstraintAttr hardwareConstraint;

  // collect constraints for each dimension symbol
  for (const Attribute &attr : constraints) {
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
    } else if (auto hw = llvm::dyn_cast<wave::HardwareConstraintAttr>(attr)) {
      hardwareConstraint = hw;
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
  // * The wave constraint tile size should divide the workgroup constraint tile
  // size evenly.
  llvm::SmallDenseMap<wave::WaveSymbolAttr, int64_t> resolvedWaveCounts;
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
    int64_t workgroupSize = resolvedWorkgroupSizes[symbol];
    int64_t numWaves = workgroupSize / resolvedWaveSize;
    if (numWaves < 1) {
      return emitError() << "invalid number of waves: " << numWaves
                         << " for dimension: " << symbol;
    }
    if (workgroupSize % resolvedWaveSize != 0) {
      return emitError() << "wave constraint tile size " << resolvedWaveSize
                         << " does not evenly divide workgroup constraint tile "
                            "size "
                         << workgroupSize << " for dimension: " << symbol;
    }
    resolvedWaveCounts[symbol] = numWaves;
  }

  // verify consistency between wave constraints and waves_per_block
  // * If both wave constraints and waves_per_block are present, the computed
  // number of waves per dimension should match the waves_per_block attribute.
  if (hardwareConstraint && !hardwareConstraint.getWavesPerBlock().empty() &&
      !waveConstraints.empty()) {
    llvm::ArrayRef<unsigned> wavesPerBlock =
        hardwareConstraint.getWavesPerBlock();
    for (auto &&[symbol, waveConstraint] : waveConstraints) {
      wave::WorkgroupConstraintAttr wgConstraint = workgroupConstraints[symbol];
      unsigned wgDim =
          static_cast<unsigned>(wgConstraint.getWorkgroupDim().getValue());
      int64_t computedWaves = resolvedWaveCounts[symbol];
      if (computedWaves != wavesPerBlock[wgDim]) {
        return emitError() << "computed number of waves (" << computedWaves
                           << ") for dimension " << symbol
                           << " does not match waves_per_block[" << wgDim
                           << "] = " << wavesPerBlock[wgDim];
      }
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
wave::WaveDialect::verifyOperationAttribute(Operation *op,
                                            NamedAttribute attr) {
  // IMPORTANT NOTE: this verifier runs before nested ops have been verified, so
  // it should not assume anything but generic IR well-formedness.
  llvm::StringSet<> usedSymbols;

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
    for (Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kHyperparameterAttrName)) {
        InFlightDiagnostic diag =
            op->emitError()
            << "defines hyperparameters when its ancestor already had";
        diag.attachNote(parent->getLoc()) << "ancestor";
        return diag;
      }
    }

    // Verify expr_list values in the hyperparameters: each value must be an
    // integer or a valid expr_list, and all referenced symbols must exist as
    // entries in the same mapping.
    for (auto [key, value] : hyperparams.getMapping().getMapping()) {
      if (llvm::isa<IntegerAttr>(value))
        continue;

      wave::WaveExprListAttr exprList =
          llvm::dyn_cast<wave::WaveExprListAttr>(value);
      if (!exprList)
        return op->emitError()
               << "hyperparameter \"" << key.getName()
               << "\" must either be an integer or an expr_list";

      // Each expr_list must be a single-result affine map.
      if (exprList.getMap().getNumResults() != 1) {
        return op->emitError()
               << "hyperparameter \"" << key.getName()
               << "\" must be a single-result expr_list, but has "
               << exprList.getMap().getNumResults() << " results";
      }
      for (Attribute symAttr : exprList.getSymbols()) {
        wave::WaveSymbolAttr waveSym =
            llvm::dyn_cast<wave::WaveSymbolAttr>(symAttr);
        if (!waveSym) {
          return op->emitError()
                 << "hyperparameter \"" << key.getName()
                 << "\" expr_list may only contain wave symbols: " << symAttr;
        }
        usedSymbols.insert(waveSym.getName());
        if (!hyperparams.hasSymbol(waveSym)) {
          return op->emitError()
                 << "hyperparameter \"" << key.getName()
                 << "\" references symbol " << waveSym
                 << " not defined in the same hyperparameters mapping";
        }
      }
    }

    // Verify that derived symbols do not form cycles.
    if (llvm::failed(wave::verifyHyperparameterAcyclicity(
            hyperparams, [&]() { return op->emitError(); })))
      return llvm::failure();

    for (auto [key, value] : hyperparams.getMapping().getMapping()) {
      wave::WaveExprListAttr exprList =
          llvm::dyn_cast<wave::WaveExprListAttr>(value);
      if (!exprList)
        continue;

      AffineExpr result = exprList.getMap().getResult(0);

      // The sole expression must be a ceiling division of a symbol by a
      // constant.
      AffineBinaryOpExpr divExpr = llvm::dyn_cast<AffineBinaryOpExpr>(result);
      if (!divExpr || divExpr.getKind() != AffineExprKind::CeilDiv) {
        return op->emitError()
               << "hyperparameter \"" << key.getName()
               << "\" expr_list must be a ceiling division expression";
      }
      if (!llvm::isa<AffineSymbolExpr>(divExpr.getLHS())) {
        return op->emitError() << "hyperparameter \"" << key.getName()
                               << "\" expr_list dividend must be a symbol";
      }
      if (!llvm::isa<AffineConstantExpr>(divExpr.getRHS())) {
        return op->emitError() << "hyperparameter \"" << key.getName()
                               << "\" expr_list divisor must be a constant";
      }

      Attribute symAttr = exprList.getSymbols().back();
      wave::WaveSymbolAttr dep = llvm::cast<wave::WaveSymbolAttr>(symAttr);
      int64_t lhs = hyperparams.getKnownSymbolValue(dep);

      AffineExpr rhsExpr = divExpr.getRHS();
      int64_t rhs = llvm::cast<AffineConstantExpr>(rhsExpr).getValue();

      // The dividend must be evenly divisible by the divisor.
      if (rhs != 0 && lhs % rhs != 0) {
        return op->emitError()
               << "hyperparameter \"" << key.getName() << "\" has dividend ("
               << lhs << ") that is not evenly divisible by the divisor ("
               << rhs << ")";
      }
    }

    WalkResult walkResult = op->walk([&](Operation *op) {
      if (llvm::failed(verifyTypeRangeHyperparamUses(
              hyperparams, op->getResultTypes(), usedSymbols,
              [&]() { return op->emitOpError() << "result "; }))) {
        return WalkResult::interrupt();
      }

      for (Region &region : op->getRegions()) {
        // Can't use llvm::enumerate because of nested lambda capture defect.
        unsigned blockNo = 0;
        for (Block &block : region) {
          if (llvm::failed(verifyTypeRangeHyperparamUses(
                  hyperparams, block.getArgumentTypes(), usedSymbols, [&]() {
                    return op->emitOpError()
                           << "region #" << region.getRegionNumber()
                           << " block #" << blockNo << " argument ";
                  }))) {
            return WalkResult::interrupt();
          }
          ++blockNo;
        }
      }

      for (const NamedAttribute &namedAttr : op->getAttrs()) {
        // Skip the hyperparameters attribute itself: its WaveSymbolAttr keys
        // are definitions, not uses.
        if (namedAttr.getName() == kHyperparameterAttrName)
          continue;
        if (llvm::failed(verifyAttributeHyperparamUses(
                hyperparams, namedAttr, usedSymbols, [&]() {
                  return op->emitOpError()
                         << "attribute " << namedAttr.getName() << " ";
                }))) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return llvm::failure();

    llvm::SmallVector<StringRef> unusedNames;
    for (wave::WaveSymbolAttr key :
         llvm::make_first_range(hyperparams.getMapping())) {
      if (!usedSymbols.contains(key.getName()))
        unusedNames.push_back(key.getName());
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
    if (!llvm::isa<IntegerAttr>(attr.getValue())) {
      return op->emitError() << attr.getName() << " expects an IntegerAttr";
    }
    return llvm::success();
  }

  if (attr.getName() == kWaveConstraintsAttrName) {
    ArrayAttr attrs = llvm::dyn_cast<ArrayAttr>(attr.getValue());
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
    for (Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kWaveConstraintsAttrName)) {
        InFlightDiagnostic diag =
            op->emitError()
            << "defines wave constraints when its ancestor already had";
        diag.attachNote(parent->getLoc()) << "ancestor";
        return diag;
      }
    }

    if (llvm::count_if(attrs, llvm::IsaPred<wave::HardwareConstraintAttr>) >
        1) {
      return op->emitError() << "only one hardware constraint is allowed";
    }

    if (!needsHyperparams) {
      return llvm::success();
    }

    // walk up to find hyperparameters
    wave::WaveHyperparameterAttr hyperparams;
    for (Operation *parent = op; parent != nullptr && !hyperparams;
         parent = parent->getParentOp()) {
      for (NamedAttribute attr : parent->getAttrs()) {
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
