// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMATERIALIZEREGCOPY
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Materialize register copies by routing memref.load through temporary
/// buffers in virtual register space (memspace 128).
class WaterMaterializeRegCopyPass
    : public water::impl::WaterMaterializeRegCopyBase<
          WaterMaterializeRegCopyPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    // Collect all load operations to transform
    SmallVector<memref::LoadOp> loadsToTransform;
    getOperation()->walk(
        [&](memref::LoadOp loadOp) { loadsToTransform.push_back(loadOp); });

    for (memref::LoadOp loadOp : loadsToTransform) {
      if (failed(materializeRegCopy(rewriter, loadOp)))
        return signalPassFailure();
    }
  }

private:
  /// Transform a single load operation to use register space copy.
  LogicalResult materializeRegCopy(IRRewriter &rewriter,
                                   memref::LoadOp loadOp) {
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPoint(loadOp);

    // Get the source memref and indices
    Value memref = loadOp.getMemRef();
    ValueRange indices = loadOp.getIndices();
    auto memrefType = cast<MemRefType>(memref.getType());
    Type elementType = memrefType.getElementType();

    // Create constants for subview
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (Value index : indices) {
      offsets.push_back(index);
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Create subview of size [1, 1, ..., 1] at the load indices
    auto subviewType =
        memref::SubViewOp::inferResultType(memrefType, offsets, sizes, strides);
    auto subviewMemRefType = cast<MemRefType>(subviewType);
    Value subview = memref::SubViewOp::create(rewriter, loc, subviewMemRefType,
                                              memref, offsets, sizes, strides);

    // Create temporary buffer in virtual register space (memspace 128)
    auto regMemSpace = rewriter.getI32IntegerAttr(128);
    auto tempType =
        MemRefType::get(subviewMemRefType.getShape(), elementType,
                        /*layout=*/MemRefLayoutAttrInterface{}, regMemSpace);
    Value tempAlloca = memref::AllocaOp::create(rewriter, loc, tempType,
                                                /*dynamicSizes=*/ValueRange{},
                                                /*alignment=*/IntegerAttr());

    // Copy from subview to temp register buffer
    memref::CopyOp::create(rewriter, loc, subview, tempAlloca);

    // Create zero indices for loading from temp buffer
    SmallVector<Value> zeroIndices;
    for (unsigned i = 0; i < indices.size(); ++i)
      zeroIndices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));

    // Load from the temporary register buffer
    Value result =
        memref::LoadOp::create(rewriter, loc, tempAlloca, zeroIndices);

    // Replace the original load with the new one
    rewriter.replaceOp(loadOp, result);

    return success();
  }
};

} // namespace
