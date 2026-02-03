// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERFUSESCALELOADSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

static bool isElementwise(Operation *op) {
  if (!op)
    return false;

  return OpTrait::hasElementwiseMappableTraits(op) &&
         op->getNumResults() == 1 && op->getNumOperands() == 1;
}

static LLVM::LoadOp findLoadOp(Value value) {
  // Follow elementwise operations until we find a load.
  while (isElementwise(value.getDefiningOp()))
    value = value.getDefiningOp()->getOperand(0);

  return value.getDefiningOp<LLVM::LoadOp>();
}

namespace {
struct WmmaScaleLoadRewriter final : OpRewritePattern<amdgpu::ScaledWMMAOp> {
  WmmaScaleLoadRewriter(MLIRContext *context, unsigned waveSize)
      : OpRewritePattern<amdgpu::ScaledWMMAOp>(context), waveSize(waveSize) {}

  LogicalResult matchAndRewrite(amdgpu::ScaledWMMAOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAFirstScaleLane() != 0 || op.getBFirstScaleLane() != 0)
      return failure();

    LLVM::LoadOp loadA = findLoadOp(op.getScaleA());
    if (!loadA)
      return failure();

    LLVM::LoadOp loadB = findLoadOp(op.getScaleB());
    if (!loadB)
      return failure();

    // Check if loads are from the same base pointer.
    if (loadA.getAddr() == loadB.getAddr())
      return failure();

    // Check loads have the same pointer type.
    if (loadA.getAddr().getType() != loadB.getAddr().getType())
      return failure();

    // Check loads have the same result type.
    if (loadA.getResult().getType() != loadB.getResult().getType())
      return failure();

    // Check loads have the same flags.
    if (loadA.getVolatile_() != loadB.getVolatile_() ||
        loadA.getNontemporal() != loadB.getNontemporal() ||
        loadA.getOrdering() != loadB.getOrdering() ||
        loadA.getSyncscope() != loadB.getSyncscope() ||
        loadA.getAlignment() != loadB.getAlignment() ||
        loadA.getInvariant() != loadB.getInvariant() ||
        loadA.getInvariantGroup() != loadB.getInvariantGroup())
      return failure();

    auto scaleAType = cast<VectorType>(op.getScaleA().getType());
    auto scaleBType = cast<VectorType>(op.getScaleB().getType());

    // Each lane holds 4 scale values (4 bytes = 4 x f8).
    constexpr unsigned scalesPerLane = 4;
    unsigned lanesA = scaleAType.getNumElements() / scalesPerLane;
    unsigned lanesB = scaleBType.getNumElements() / scalesPerLane;

    // Check if both scale vectors can fit within the wave size.
    if (lanesA + lanesB > waveSize)
      return failure();

    // Check that scaleA fits in half the wave (so scaleB can start at lane 16).
    if (lanesA > waveSize / 2)
      return failure();

    // Create fused load: select(lane_id < waveSize/2, ptrA, ptrB).
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(loadA);

    Value laneId =
        gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
    Value halfWave = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(waveSize / 2));
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                      laneId, halfWave);
    Value selectedPtr = arith::SelectOp::create(
        rewriter, loc, cmp, loadA.getAddr(), loadB.getAddr());

    // Create new load with the selected pointer.
    unsigned alignment = loadA.getAlignment() ? *loadA.getAlignment() : 0;
    StringRef syncscope =
        loadA.getSyncscope() ? *loadA.getSyncscope() : StringRef();
    auto fusedLoad = LLVM::LoadOp::create(
        rewriter, loc, loadA.getResult().getType(), selectedPtr, alignment,
        loadA.getVolatile_(), loadA.getNontemporal(), loadA.getInvariant(),
        loadA.getInvariantGroup(), loadA.getOrdering(), syncscope);

    // Replace old loads with the fused load.
    rewriter.replaceOp(loadA, fusedLoad.getResult());
    rewriter.replaceOp(loadB, fusedLoad.getResult());

    // Update wmma op to read scaleB from lane 16.
    rewriter.modifyOpInPlace(op, [&]() {
      op.setBFirstScaleLaneAttr(rewriter.getI32IntegerAttr(waveSize / 2));
    });

    return success();
  }

private:
  unsigned waveSize;
};

struct FuseScaleLoadsPass
    : public water::impl::WaterFuseScaleLoadsPassBase<FuseScaleLoadsPass> {
  using WaterFuseScaleLoadsPassBase::WaterFuseScaleLoadsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<WmmaScaleLoadRewriter>(patterns.getContext(), waveSize);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
