// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Transforms/Passes.h"

#include "water/Analysis/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::water::test {
#define GEN_PASS_DEF_TESTUNIFORMITYANALYSISPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir::water::test

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::water;

static void setWaveUniformityAnalysisResults(Operation *top,
                                             const DataFlowSolver &solver) {
  // Walk all operations and attach uniformity attributes.
  top->walk([&](Operation *op) {
    if (op->getNumResults() == 0)
      return;

    // Check if all results are uniform.
    bool allResultsUniform = true;
    std::optional<uint64_t> subgroupLinearWidth;

    for (Value result : op->getResults()) {
      if (water::isUniform(result, solver)) {
        continue;
      }

      allResultsUniform = false;

      // Check if subgroup linear.
      if (auto width = water::getSubgroupLinearWidth(result, solver)) {
        if (!subgroupLinearWidth) {
          subgroupLinearWidth = width;
        } else if (*subgroupLinearWidth != *width) {
          // Results have different widths, don't attach attribute.
          subgroupLinearWidth = std::nullopt;
          break;
        }
      } else {
        // Not uniform and not subgroup linear.
        subgroupLinearWidth = std::nullopt;
        break;
      }
    }

    // Attach unit attribute if all results are uniform.
    if (allResultsUniform) {
      op->setAttr("wave.uniform", UnitAttr::get(op->getContext()));
    } else if (subgroupLinearWidth) {
      // Attach subgroup linear attribute with width.
      op->setAttr("wave.subgroup_linear",
                  IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                   *subgroupLinearWidth));
    }
  });
}

namespace {
struct TestUniformityAnalysisPass
    : public mlir::water::test::impl::TestUniformityAnalysisPassBase<
          TestUniformityAnalysisPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    water::addWaterUniformityAnalysis(solver);

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    setWaveUniformityAnalysisResults(op, solver);
  }
};
} // namespace
