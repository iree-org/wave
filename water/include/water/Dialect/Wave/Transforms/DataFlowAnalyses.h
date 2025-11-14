// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_DATAFLOWANALYSES_H
#define WATER_DIALECT_WAVE_TRANSFORMS_DATAFLOWANALYSES_H

#include "llvm/ADT/FunctionExtras.h"

namespace llvm {
struct LogicalResult;
}

namespace mlir {
class DataFlowSolver;
class SymbolTableCollection;
class Operation;
class Value;
class DictionaryAttr;
} // namespace mlir

namespace wave {
using SetIndexLatticeFn =
    llvm::function_ref<void(mlir::Value, mlir::DictionaryAttr)>;
using OverrideInitializationFn = llvm::function_ref<llvm::LogicalResult(
    mlir::Operation *, SetIndexLatticeFn)>;

// Configuration options for forward analysis. Mostly usable for testing
// purposes.
struct WaveIndexExprsAnalysisOptions {
  OverrideInitializationFn overrideInitialization = nullptr;
  bool disableForward = false;
  bool disableBackward = false;
};

// Add analyses for index expression propagation to the solver.
void addWaveIndexExprsAnalyses(mlir::DataFlowSolver &solver,
                               mlir::SymbolTableCollection &symbolTable,
                               WaveIndexExprsAnalysisOptions options = {});

llvm::LogicalResult
setWaveIndexExprAnalysisResults(mlir::Operation *top,
                                const mlir::DataFlowSolver &solver);

// Run the dataflow analyses and capture whether some diagnostics were emitted.
// Only emit a generic diagnostic if no more specific diagnostic was emitted.
// This is usually indicative of some deep internal problem in the dataflow
// solver.
llvm::LogicalResult runSolverAndCaptureErrors(mlir::DataFlowSolver &solver,
                                              mlir::Operation *root,
                                              bool force = false);
} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_DATAFLOWANALYSES_H
