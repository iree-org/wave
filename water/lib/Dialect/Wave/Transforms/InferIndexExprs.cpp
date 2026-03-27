// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/DataFlowAnalyses.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

using namespace mlir;

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINFERINDEXEXPRSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {
class InferIndexExprsPass
    : public wave::impl::WaterWaveInferIndexExprsPassBase<InferIndexExprsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    if (llvm::failed(wave::verifyNormalFormPassPrecondition(
            {wave::WaveNormalForm::FunctionBoundarySpecified,
             wave::WaveNormalForm::OpTypesSpecified},
            getOperation(), getArgument())))
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    getOperation()->walk(
        [&](wave::IterateOp iterateOp) { iterateOp.makeIsolated(rewriter); });

    SymbolTableCollection symbolTable;
    DataFlowConfig config;
    config.setInterprocedural(false);
    DataFlowSolver solver(config);

    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    wave::DelayedErrorEmitterInfo delayedErrorInfo =
        wave::addWaveIndexExprsAnalyses(solver, symbolTable);

    if (llvm::failed(
            wave::runSolverAndCaptureErrors(solver, getOperation(), false)))
      return signalPassFailure();

    if (llvm::failed(wave::setWaveIndexExprAnalysisResults(
            getOperation(), solver, delayedErrorInfo)))
      return signalPassFailure();

    getOperation()->walk([&](wave::IterateOp iterateOp) {
      iterateOp.makeNonIsolated(rewriter);
    });

    if (llvm::failed(wave::setNormalFormPassPostcondition(
            {wave::WaveNormalForm::IndexExprsSpecified}, getOperation())))
      return signalPassFailure();
  }
};
} // namespace
