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
        setIndexForValue(value, nullptr);
        continue;
      }

      auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
      if (!dict || llvm::any_of(dict.getValue(), [](NamedAttribute attr) {
            return !llvm::isa<wave::WaveIndexMappingAttr>(attr.getValue());
          })) {
        return op->emitError()
               << "expected " << attributeName
               << " to be an array of "
                  "dictionaries with WaveIndexMappingAttr or UnitAttr values";
      }

      setIndexForValue(value, dict);
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
    addWaveIndexExprsAnalyses(solver, symbolTable, options);

    IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](wave::IterateOp iterateOp) { iterateOp.makeIsolated(rewriter); });

    if (failed(wave::runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    if (failed(setWaveIndexExprAnalysisResults(getOperation(), solver)))
      return signalPassFailure();

    getOperation()->walk([&](wave::IterateOp iterateOp) {
      iterateOp.makeNonIsolated(rewriter);
    });
  }
};
