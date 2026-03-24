// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/DataFlowAnalyses.h"

using namespace mlir;
using namespace wave;

namespace mlir::water::test {
#define GEN_PASS_DEF_TESTWAVEDIALECTINFERINDEXEXPRSPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir::water::test

static LogicalResult
overrideInitialization(Operation *top,
                       wave::SetIndexLatticeFn setIndexForValue) {
  auto overrideIndex = [&](Operation *op, ValueRange values,
                           const llvm::Twine &attributeName) -> LogicalResult {
    auto overrides = op->getAttrOfType<ArrayAttr>(attributeName.str());
    if (!overrides)
      return success();
    for (auto [value, attr] : llvm::zip(values, overrides)) {
      if (llvm::isa<UnitAttr>(attr))
        continue;
      if (auto strAttr = llvm::dyn_cast<StringAttr>(attr);
          strAttr && strAttr.getValue() == "<top>") {
        setIndexForValue(value, nullptr, llvm::DenseMap<StringAttr, int32_t>(),
                         nullptr);
        continue;
      }

      // Support formats:
      // 1. DictionaryAttr (index expressions only, default priority, no
      // vectorShape)
      // 2. ArrayAttr [priority, indexExprs] - priority (IntegerAttr) and dict
      // 3. ArrayAttr [indexExprs, vectorShape] - dict and optional vectorShape
      // 4. ArrayAttr [priority, indexExprs, vectorShape] - all three
      // 5. ArrayAttr [priorityDict, indexExprs] - per-key priorities (dict of
      //    IntegerAttr) and dict
      // 6. ArrayAttr [priorityDict, indexExprs, vectorShape] - per-key
      //    priorities, dict, and vectorShape
      DictionaryAttr indexExprs = nullptr;
      llvm::DenseMap<StringAttr, int32_t> priorities;
      bool hasPriorities = false;
      DictionaryAttr vectorShape = nullptr;

      auto setUniformPriority = [&](int32_t priority) {
        hasPriorities = true;
        if (indexExprs) {
          for (NamedAttribute na : indexExprs)
            priorities[na.getName()] = priority;
        }
      };

      auto setPerKeyPriorities = [&](DictionaryAttr priDict) {
        hasPriorities = true;
        for (NamedAttribute na : priDict) {
          if (auto intAttr = llvm::dyn_cast<IntegerAttr>(na.getValue()))
            priorities[na.getName()] = intAttr.getInt();
        }
      };

      if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
        if (arrayAttr.size() == 2) {
          if (auto firstInt = llvm::dyn_cast<IntegerAttr>(arrayAttr[0])) {
            // [priority, dict]
            indexExprs = llvm::dyn_cast<DictionaryAttr>(arrayAttr[1]);
            setUniformPriority(firstInt.getInt());
          } else if (auto firstDict =
                         llvm::dyn_cast<DictionaryAttr>(arrayAttr[0])) {
            auto secondDict = llvm::dyn_cast<DictionaryAttr>(arrayAttr[1]);
            // Distinguish [priDict, indexExprs] from [indexExprs, vecShape]:
            // if first dict has IntegerAttr values, it's per-key priorities.
            if (secondDict && !firstDict.empty() &&
                llvm::all_of(firstDict, [](NamedAttribute na) {
                  return llvm::isa<IntegerAttr>(na.getValue());
                })) {
              indexExprs = secondDict;
              setPerKeyPriorities(firstDict);
            } else {
              // [dict, vectorShape]
              indexExprs = firstDict;
              if (!llvm::isa<UnitAttr>(arrayAttr[1]))
                vectorShape = secondDict;
              if (!vectorShape)
                return op->emitError()
                       << "expected vector shape to be a DictionaryAttr";
            }
          }
        } else if (arrayAttr.size() == 3) {
          if (auto firstInt = llvm::dyn_cast<IntegerAttr>(arrayAttr[0])) {
            // [priority, dict, vectorShape]
            indexExprs = llvm::dyn_cast<DictionaryAttr>(arrayAttr[1]);
            setUniformPriority(firstInt.getInt());
            if (!llvm::isa<UnitAttr>(arrayAttr[2]))
              vectorShape = llvm::dyn_cast<DictionaryAttr>(arrayAttr[2]);
          } else if (auto firstDict =
                         llvm::dyn_cast<DictionaryAttr>(arrayAttr[0])) {
            // [priDict, dict, vectorShape]
            indexExprs = llvm::dyn_cast<DictionaryAttr>(arrayAttr[1]);
            setPerKeyPriorities(firstDict);
            if (!llvm::isa<UnitAttr>(arrayAttr[2]))
              vectorShape = llvm::dyn_cast<DictionaryAttr>(arrayAttr[2]);
          }
        }
      } else {
        indexExprs = llvm::dyn_cast<DictionaryAttr>(attr);
      }

      if (!indexExprs ||
          llvm::any_of(indexExprs.getValue(), [](NamedAttribute attr) {
            return !llvm::isa<wave::WaveIndexMappingAttr>(attr.getValue());
          })) {
        return op->emitError()
               << "expected each element of " << attributeName
               << " to be either a DictionaryAttr or an array "
                  "[priority?, indexExprs, vectorShape?], where indexExprs is "
                  "a DictionaryAttr whose values are WaveIndexMappingAttr";
      }

      if (!hasPriorities)
        setUniformPriority(wave::IndexExprsLatticeStorage::kLowestPriority);

      setIndexForValue(value, indexExprs, std::move(priorities), vectorShape);
    }
    return success();
  };

  WalkResult walkResult = top->walk([&](mlir::Operation *op) {
    if (failed(overrideIndex(op, op->getResults(),
                             "wave_test.override_result_index")))
      return WalkResult::interrupt();

    if (failed(overrideIndex(op, op->getOperands(),
                             "wave_test.override_operand_index")))
      return WalkResult::interrupt();

    for (Region &region : op->getRegions()) {
      for (auto &&[blockNumber, block] : llvm::enumerate(region)) {
        if (failed(overrideIndex(op, block.getArguments(),
                                 "wave_test.override_region" +
                                     llvm::Twine(region.getRegionNumber()) +
                                     "_block" + llvm::Twine(blockNumber) +
                                     "_args_index")))
          return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

class TestWaveDialectInferIndexExprsPass
    : public mlir::water::test::impl::TestWaveDialectInferIndexExprsPassBase<
          TestWaveDialectInferIndexExprsPass> {
public:
  void runOnOperation() override {
    SymbolTableCollection symbolTable;
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);

    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    WaveIndexExprsAnalysisOptions options;
    options.disableBackward = getOperation()->getAttrOfType<UnitAttr>(
                                  "wave_test.disable_backward") != nullptr;
    options.disableForward = getOperation()->getAttrOfType<UnitAttr>(
                                 "wave_test.disable_forward") != nullptr;
    options.overrideInitialization = overrideInitialization;
    wave::DelayedErrorEmitterInfo delayedErrorInfo =
        wave::addWaveIndexExprsAnalyses(solver, symbolTable, options);

    IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](wave::IterateOp iterateOp) { iterateOp.makeIsolated(rewriter); });

    if (failed(wave::runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    if (failed(setWaveIndexExprAnalysisResults(getOperation(), solver,
                                               delayedErrorInfo)))
      return signalPassFailure();

    getOperation()->walk([&](wave::IterateOp iterateOp) {
      iterateOp.makeNonIsolated(rewriter);
    });
  }
};
