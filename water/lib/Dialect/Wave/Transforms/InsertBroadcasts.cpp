// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::dataflow;

#define DEBUG_TYPE "wave-insert-broadcasts"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINSERTBROADCASTSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

struct InsertBroadcastsPass
    : public wave::impl::WaterWaveInsertBroadcastsPassBase<
          InsertBroadcastsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    // Run uniformity analysis.
    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    wave::addWaveUniformityAnalysis(solver);

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
        if (!wave::isUniform(result, solver))
          continue;

        // Check if any operand is non-uniform.
        bool hasNonUniformOperand = false;
        for (Value operand : currentOp->getOperands()) {
          if (!wave::isUniform(operand, solver)) {
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
    OpBuilder builder(op->getContext());
    for (auto [opToInsertAfter, value] : insertsNeeded) {
      builder.setInsertionPointAfter(opToInsertAfter);

      // Create broadcast operation.
      auto broadcast = builder.create<gpu::SubgroupBroadcastOp>(
          opToInsertAfter->getLoc(), value.getType(), value,
          /*id=*/nullptr,
          /*broadcast_type=*/gpu::BroadcastType::first_active_lane);

      // Replace all uses of the original value (except in the broadcast
      // itself).
      value.replaceAllUsesExcept(broadcast.getResult(), broadcast);
    }
  }
};

} // namespace
