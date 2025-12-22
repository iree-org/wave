// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

#define GEN_PASS_DEF_LOWERWAVECONTROLFLOWPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// IterateOp
//===----------------------------------------------------------------------===//

/// Lower `wave.iterate` to `scf.for`.
class IterateOpLoweringPattern : public OpConversionPattern<wave::IterateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::IterateOp op, wave::IterateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get hyperparameters from the function.
    func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
      return rewriter.notifyMatchFailure(op, "iterate op not in function");
    }

    auto hyperparamAttr =
        parentFunc->getAttrOfType<wave::WaveHyperparameterAttr>(
            wave::WaveDialect::kHyperparameterAttrName);
    if (!hyperparamAttr) {
      return rewriter.notifyMatchFailure(
          op, "no hyperparameters found in function");
    }

    // Get the iterator symbol (e.g., "K").
    wave::WaveSymbolAttr iteratorSymbol = op.getIterator();
    StringRef symbolName = iteratorSymbol.getName();

    // Look for tiling constraints in function attributes.
    ArrayAttr constraints = parentFunc->getAttrOfType<ArrayAttr>(
        wave::WaveDialect::kWaveConstraintsAttrName);
    if (!constraints) {
      return rewriter.notifyMatchFailure(
          op, "no wave constraints found in function");
    }

    // Get the dimension size (e.g., K = 640) from hyperparameters.
    std::optional<SmallVector<int64_t>> resolvedDims =
        wave::resolveSymbolNames(iteratorSymbol, hyperparamAttr);
    if (!resolvedDims || resolvedDims->size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "iterator symbol not found in hyperparameters");
    }
    int64_t dimSize = resolvedDims->front();

    // Find tiling constraint for this dimension to get tile_size.
    std::optional<int64_t> tileSize;
    for (Attribute constraintAttr : constraints) {
      auto tilingConstraint =
          dyn_cast<wave::TilingConstraintAttr>(constraintAttr);
      if (!tilingConstraint)
        continue;

      wave::WaveSymbolAttr constraintDim = tilingConstraint.getDim();
      if (constraintDim.getName() != symbolName)
        continue;

      wave::WaveExprListAttr tileSizeAttr = tilingConstraint.getTileSize();
      AffineMap tileSizeMap = tileSizeAttr.getMap();
      ArrayRef<Attribute> tileSizeSymbols = tileSizeAttr.getSymbols();

      // Evaluate the tile size using hyperparameters.
      std::optional<SmallVector<int64_t>> evaluatedTileSize =
          wave::evaluateMapWithHyperparams(tileSizeMap, tileSizeSymbols,
                                           hyperparamAttr);
      if (!evaluatedTileSize) {
        return rewriter.notifyMatchFailure(
            op, "failed to evaluate tile size from tiling constraint");
      }
      if (evaluatedTileSize->size() != 1) {
        return rewriter.notifyMatchFailure(op,
                                           "tile size must be single value");
      }
      tileSize = (*evaluatedTileSize)[0];
      break;
    }

    if (!tileSize) {
      return rewriter.notifyMatchFailure(
          op, "no tiling constraint found for iterator symbol");
    }

    // TODO(tyb): we reject non-exact division for now, which should require
    // peeling or padding to be correct.
    // TODO(tyb): make these errors better visible to the caller from python.
    if (*tileSize == 0) {
      return rewriter.notifyMatchFailure(op, "tile size cannot be zero");
    }
    if (dimSize % *tileSize != 0) {
      return op.emitOpError("non-exact division not supported to prevent "
                            "potential out-of-bounds access");
    }
    int64_t numIterations = dimSize / *tileSize;

    // Create loop bounds.
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, numIterations);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    rewriter.setInsertionPoint(op);

    // Create the scf.for loop.
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                             adaptor.getIterArgs());

    // Copy the iterator attribute from wave.iterate to scf.for so that
    // WaveIndexSequenceInterface can still resolve their iterator symbols.
    forOp->setAttr("iterator", iteratorSymbol);

    // Convert the body.
    Block &waveBody = op.getBody().front();
    Block &scfBody = *forOp.getBody();

    // Set up insertion point inside the loop body.
    rewriter.setInsertionPointToStart(&scfBody);

    // Create mapping from old block arguments to new ones.
    IRMapping mapping;

    // Map iter_args.
    // Note: wave.iterate doesn't expose the induction variable, so we skip it.
    for (auto [oldArg, newArg] : llvm::zip_equal(
             waveBody.getArguments(), scfBody.getArguments().drop_front())) {
      mapping.map(oldArg, newArg);
    }

    // Clone all operations except the terminator.
    for (Operation &bodyOp : waveBody.without_terminator()) {
      rewriter.clone(bodyOp, mapping);
    }

    // Convert wave.yield to scf.yield.
    auto yieldOp = cast<wave::YieldOp>(waveBody.getTerminator());
    SmallVector<Value> yieldValues;
    yieldValues.reserve(yieldOp.getValues().size());
    for (Value value : yieldOp.getValues()) {
      yieldValues.push_back(mapping.lookup(value));
    }
    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

    // Replace the original op with the for loop results.
    rewriter.replaceOp(op, forOp.getResults());

    return success();
  }
};

struct LowerWaveControlFlowPass
    : public ::impl::LowerWaveControlFlowPassBase<LowerWaveControlFlowPass> {
  using LowerWaveControlFlowPassBase::LowerWaveControlFlowPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    ConversionTarget target(*ctx);
    target.addLegalDialect<
        // clang-format off
      arith::ArithDialect,
      func::FuncDialect,
      scf::SCFDialect,
      wave::WaveDialect
        // clang-format on
        >();

    // Only mark wave.iterate and wave.yield as illegal.
    target.addIllegalOp<wave::IterateOp, wave::YieldOp>();

    // Mark cloned operations in scf.for body region as legal, as they will be
    // lowered by lower-wave-to-mlir next.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);

    // We don't need a type converter for this pass since we're only
    // lowering control flow and leaving data types unchanged
    patterns.add<IterateOpLoweringPattern>(ctx);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> wave::createLowerWaveControlFlowPass() {
  return std::make_unique<LowerWaveControlFlowPass>();
}
