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

static bool isSupportedBroadcastType(Type type) {
  if (auto integerType = llvm::dyn_cast<IntegerType>(type))
    return llvm::is_contained({8, 16, 32, 64}, (int)integerType.getWidth());

  if (isa<IndexType, FloatType>(type))
    return true;

  return false;
}

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
    SmallVector<Value> insertsNeeded;

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
        if (hasNonUniformOperand && isSupportedBroadcastType(result.getType()))
          insertsNeeded.push_back(result);
      }
    });

    // Insert broadcasts.
    OpBuilder builder(&getContext());
    for (Value value : insertsNeeded) {
      if (auto opToInsertAfter = value.getDefiningOp()) {
        builder.setInsertionPointAfter(opToInsertAfter);
      } else {
        // Block argument.
        builder.setInsertionPointToStart(value.getParentBlock());
      }

      auto broadcast = gpu::SubgroupBroadcastOp::create(
          builder, value.getLoc(), value.getType(), value,
          /*id=*/nullptr,
          /*broadcast_type=*/gpu::BroadcastType::first_active_lane);

      // Replace all uses of the original value (except in the broadcast
      // itself).
      value.replaceAllUsesExcept(broadcast.getResult(), broadcast);
    }
  }
};

} // namespace
