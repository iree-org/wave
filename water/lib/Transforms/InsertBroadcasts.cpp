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
    return llvm::is_contained({16, 32, 64}, (int)integerType.getWidth());

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

    auto isUniform = [&](Value value) -> bool {
      return water::isUniform(value, solver);
    };
    auto isNonUniform = [&](Value value) -> bool {
      return !water::isUniform(value, solver);
    };

    op->walk([&](Operation *currentOp) {
      if (isa<gpu::SubgroupBroadcastOp>(currentOp))
        return;

      // Check if any operand is non-uniform.
      if (!llvm::any_of(currentOp->getOperands(), isNonUniform))
        return;

      for (Value result : currentOp->getResults()) {
        if (!isSupportedBroadcastType(result.getType()))
          continue;

        if (!isUniform(result))
          continue;

        insertsNeeded.push_back(result);
      }
    });

    // Insert broadcasts.
    OpBuilder builder(&getContext());
    for (Value value : insertsNeeded) {
      builder.setInsertionPointAfterValue(value);

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
