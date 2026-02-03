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
#include "mlir/IR/IRMapping.h"
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

  if (op->getNumResults() != 1 || op->getNumOperands() != 1 || !isPure(op))
    return false;

  // Support llvm.bitcast in addition to elementwise ops.
  if (isa<LLVM::BitcastOp>(op))
    return true;

  return OpTrait::hasElementwiseMappableTraits(op);
}

// Collect elementwise ops between value and its defining load.
// Returns the load op and populates elemOps in reverse order (load -> value).
static LLVM::LoadOp
collectElementwiseChain(Value value, SmallVectorImpl<Operation *> &elemOps) {
  while (isElementwise(value.getDefiningOp())) {
    elemOps.push_back(value.getDefiningOp());
    value = value.getDefiningOp()->getOperand(0);
  }
  std::reverse(elemOps.begin(), elemOps.end());
  return value.getDefiningOp<LLVM::LoadOp>();
}

// Clone elementwise chain, replacing the input with newInput.
static Value cloneElementwiseChain(PatternRewriter &rewriter,
                                   ArrayRef<Operation *> elemOps,
                                   Value newInput) {
  Value current = newInput;
  IRMapping mapping;
  for (Operation *op : elemOps) {
    mapping.map(op->getOperand(0), current);
    Operation *cloned = rewriter.clone(*op, mapping);
    current = cloned->getResult(0);
  }
  return current;
}

namespace {
struct WmmaScaleLoadRewriter final : OpRewritePattern<amdgpu::ScaledWMMAOp> {
  WmmaScaleLoadRewriter(MLIRContext *context, unsigned waveSize)
      : OpRewritePattern<amdgpu::ScaledWMMAOp>(context), waveSize(waveSize) {}

  LogicalResult matchAndRewrite(amdgpu::ScaledWMMAOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAFirstScaleLane() != 0 || op.getBFirstScaleLane() != 0)
      return failure();

    SmallVector<Operation *> elemOpsA, elemOpsB;
    LLVM::LoadOp loadA = collectElementwiseChain(op.getScaleA(), elemOpsA);
    if (!loadA)
      return failure();

    LLVM::LoadOp loadB = collectElementwiseChain(op.getScaleB(), elemOpsB);
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

    auto upperBound = rewriter.getIndexAttr(waveSize);
    Value laneId = gpu::LaneIdOp::create(rewriter, loc, upperBound);
    Value halfWave = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIndexAttr(waveSize / 2));
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                      laneId, halfWave);
    Value selectedPtr = arith::SelectOp::create(
        rewriter, loc, cmp, loadA.getAddr(), loadB.getAddr());

    // Create new load with the selected pointer.
    unsigned alignment = loadA.getAlignment().value_or(0);
    StringRef syncscope =
        loadA.getSyncscope() ? *loadA.getSyncscope() : StringRef();
    auto fusedLoad = LLVM::LoadOp::create(
        rewriter, loc, loadA.getResult().getType(), selectedPtr, alignment,
        loadA.getVolatile_(), loadA.getNontemporal(), loadA.getInvariant(),
        loadA.getInvariantGroup(), loadA.getOrdering(), syncscope);

    // Clone elementwise chains with the fused load as input.
    Value newScaleA =
        cloneElementwiseChain(rewriter, elemOpsA, fusedLoad.getResult());
    Value newScaleB =
        cloneElementwiseChain(rewriter, elemOpsB, fusedLoad.getResult());

    // Update wmma op to use fused scales and read scaleB from lane 16.
    rewriter.modifyOpInPlace(op, [&]() {
      op.getScaleAMutable().assign(newScaleA);
      op.getScaleBMutable().assign(newScaleB);
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
