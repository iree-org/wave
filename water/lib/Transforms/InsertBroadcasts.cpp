// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Analysis/UniformityAnalysis.h"
#include "water/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::water {
#define GEN_PASS_DEF_WATERINSERTBROADCASTSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

struct InsertBroadcastsPass
    : public water::impl::WaterInsertBroadcastsPassBase<InsertBroadcastsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    // Run uniformity analysis.
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    water::addWaterUniformityAnalysis(solver);

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Collect operations that need broadcasts.
    SmallVector<std::pair<Operation *, Value>> insertsNeeded;

    op->walk([&](Operation *currentOp) {
      // Skip broadcast operations.
      if (isa<gpu::SubgroupBroadcastOp>(currentOp))
        return;

      // Check each result.
      for (Value result : currentOp->getResults()) {
        // Skip if result is not uniform.
        if (!water::isUniform(result, solver))
          continue;

        // Check if any operand is non-uniform.
        bool hasNonUniformOperand = false;
        for (Value operand : currentOp->getOperands()) {
          if (!water::isUniform(operand, solver)) {
            hasNonUniformOperand = true;
            break;
          }
        }

        // If we have non-uniform operands but uniform result, insert broadcast.
        if (hasNonUniformOperand)
          insertsNeeded.push_back({currentOp, result});
      }
    });

    // Insert broadcasts.
    OpBuilder builder(&getContext());
    for (auto [opToInsertAfter, value] : insertsNeeded) {
      builder.setInsertionPointAfter(opToInsertAfter);

      Location loc = opToInsertAfter->getLoc();
      auto broadcast = gpu::SubgroupBroadcastOp::create(
          builder, loc, value.getType(), value,
          /*id=*/nullptr,
          /*broadcast_type=*/gpu::BroadcastType::first_active_lane);

      // Replace all uses of the original value (except in the broadcast
      // itself).
      value.replaceAllUsesExcept(broadcast.getResult(), broadcast);
    }
  }
};

} // namespace
