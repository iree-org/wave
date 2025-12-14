// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
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
    SmallVector<Operation *> loadsToTransform;
    getOperation()->walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
        if (!isInRegisterSpace(cast<MemRefType>(loadOp.getMemRef().getType())))
          loadsToTransform.push_back(op);
      } else if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
        if (!isInRegisterSpace(cast<MemRefType>(loadOp.getBase().getType())))
          loadsToTransform.push_back(op);
      }
    });

    for (Operation *op : loadsToTransform) {
      if (failed(materializeRegCopy(rewriter, op)))
        return signalPassFailure();
    }

    // Hoist allocas out of loops when their loads are yielded
    getOperation()->walk(
        [&](scf::ForOp forOp) { (void)hoistAllocasFromLoop(rewriter, forOp); });
  }

private:
  /// Check if a memref type is in virtual register space (memspace 128).
  static bool isInRegisterSpace(MemRefType memrefType) {
    if (auto memSpace =
            dyn_cast_or_null<IntegerAttr>(memrefType.getMemorySpace()))
      return memSpace.getInt() == 128;
    return false;
  }

  /// Transform a single load operation to use register space copy.
  LogicalResult materializeRegCopy(IRRewriter &rewriter, Operation *op) {
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);

    // Extract memref, indices, and element type from either load type
    Value memref, loadResult;
    ValueRange indices;
    Type elementType;
    SmallVector<int64_t> loadShape;

    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      memref = loadOp.getMemRef();
      indices = loadOp.getIndices();
      loadResult = loadOp.getResult();
      elementType = loadOp.getType();
      loadShape.resize(indices.size(), 1);
    } else if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
      memref = loadOp.getBase();
      indices = loadOp.getIndices();
      loadResult = loadOp.getResult();
      VectorType vecType = loadOp.getVectorType();
      elementType = vecType.getElementType();
      loadShape.resize(indices.size() - vecType.getRank(), 1);
      llvm::append_range(loadShape, vecType.getShape());
    } else {
      return op->emitError("unsupported load operation");
    }

    auto memrefType = cast<MemRefType>(memref.getType());

    // Create subview parameters
    Attribute one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (auto [index, shape] : llvm::zip(indices, loadShape)) {
      offsets.push_back(index);
      sizes.push_back(rewriter.getIndexAttr(shape));
      strides.push_back(one);
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

    // Group uses by block and find the first use in each block
    DenseMap<Block *, Operation *> blockToFirstUse;
    for (OpOperand &use : loadResult.getUses()) {
      Operation *userOp = use.getOwner();
      Block *userBlock = userOp->getBlock();
      auto it = blockToFirstUse.find(userBlock);
      if (it == blockToFirstUse.end() || userOp->isBeforeInBlock(it->second))
        blockToFirstUse[userBlock] = userOp;
    }

    // Create zero indices for loading from temp buffer
    SmallVector<Value> zeroIndices(
        loadShape.size(), arith::ConstantIndexOp::create(rewriter, loc, 0));

    // Create one load per block, right before the first use in that block
    DenseMap<Block *, Value> blockToLoad;
    for (auto &[block, firstUse] : blockToFirstUse) {
      rewriter.setInsertionPoint(firstUse);
      Value load;
      if (isa<memref::LoadOp>(op))
        load = memref::LoadOp::create(rewriter, loc, tempAlloca, zeroIndices);
      else if (auto vecLoadOp = dyn_cast<vector::LoadOp>(op))
        load = vector::LoadOp::create(rewriter, loc, vecLoadOp.getVectorType(),
                                      tempAlloca, zeroIndices);
      blockToLoad[block] = load;
    }

    // Replace uses with the appropriate load for their block
    for (OpOperand &use : llvm::make_early_inc_range(loadResult.getUses())) {
      Block *userBlock = use.getOwner()->getBlock();
      use.set(blockToLoad[userBlock]);
    }

    // Erase the original load
    rewriter.eraseOp(op);
    return success();
  }

  /// Hoist allocas from loops when their loads are yielded.
  void hoistAllocasFromLoop(IRRewriter &rewriter, scf::ForOp loop) {
    auto yieldedValues = loop.getYieldedValuesMutable();
    if (!yieldedValues)
      return;

    auto loopResults = loop.getLoopResults();
    if (!loopResults)
      return;

    auto loopInits = loop.getInitsMutable();

    Block *body = loop.getBody();
    Location loc = loop.getLoc();

    DominanceInfo dom;

    // Find yielded values that come from loads of memspace 128 allocas
    for (auto [idx, yieldedValue, iterArg, init, result] :
         llvm::enumerate(*yieldedValues, loop.getRegionIterArgs(), loopInits,
                         *loopResults)) {
      // Check if this is a load from memspace 128
      Operation *defOp = yieldedValue.get().getDefiningOp();
      if (!defOp)
        continue;

      Value alloca;
      ValueRange loadIndices;
      if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
        alloca = loadOp.getMemRef();
        loadIndices = loadOp.getIndices();
      } else if (auto loadOp = dyn_cast<vector::LoadOp>(defOp)) {
        alloca = loadOp.getBase();
        loadIndices = loadOp.getIndices();
      } else {
        continue;
      }

      if (!loadIndices.empty())
        continue;

      // Check if loading from memspace 128 alloca defined in this loop
      auto allocaOp = alloca.getDefiningOp<memref::AllocaOp>();
      if (!allocaOp)
        continue;
      if (!isInRegisterSpace(cast<MemRefType>(alloca.getType())))
        continue;
      if (!body->findAncestorOpInBlock(*allocaOp))
        continue;

      // If load dominates any use of the iter arg, we can't hoist the alloca
      // because the load would be invalidated by the store.
      bool dominates = false;
      for (Operation *user : iterArg.getUsers()) {
        if (dom.dominates(defOp, user)) {
          dominates = true;
          break;
        }
      }
      if (dominates)
        continue;

      // Hoist the alloca before the loop
      allocaOp->moveBefore(loop);
      rewriter.setInsertionPointAfter(allocaOp);

      // Store the iter arg into the alloca
      if (isa<memref::LoadOp>(defOp)) {
        memref::StoreOp::create(rewriter, loc, init.get(), alloca, loadIndices);
      } else if (auto vectorLoad = dyn_cast<vector::LoadOp>(defOp)) {
        vector::StoreOp::create(rewriter, loc, init.get(), alloca, loadIndices);
      }

      // Create a load after the loop
      rewriter.setInsertionPointAfter(loop);
      Value loadAfterLoop;
      if (isa<memref::LoadOp>(defOp)) {
        loadAfterLoop =
            memref::LoadOp::create(rewriter, loc, alloca, loadIndices);
      } else if (auto vectorLoad = dyn_cast<vector::LoadOp>(defOp)) {
        loadAfterLoop = vector::LoadOp::create(
            rewriter, loc, vectorLoad.getVectorType(), alloca, loadIndices);
      }

      // Replace uses of the loop result with the new load
      result.replaceAllUsesWith(loadAfterLoop);
    }
  }
};

} // namespace
