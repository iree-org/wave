// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "wave-resolve-distributed-allocations"

namespace wave {
#define GEN_PASS_DEF_WATERWAVERESOLVEDISTRIBUTEDALLOCATIONSPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

using namespace mlir;
using namespace wave;

namespace {

struct ResolveDistributedAllocations
    : public wave::impl::WaterWaveResolveDistributedAllocationsPassBase<
          ResolveDistributedAllocations> {

  /// Set the ordered_syms attribute on ReadOp and WriteOp based on their
  /// memory operand's WaveTensorType shape.
  void setOrderedSymsOnReadWriteOps(Operation *root) {
    root->walk([&](ReadOp readOp) {
      if (readOp.getOrderedSymsAttr())
        return;

      auto tensorType = dyn_cast<WaveTensorType>(readOp.getMemory().getType());
      if (!tensorType)
        return;

      readOp.setOrderedSyms(tensorType.getShape());
    });

    root->walk([&](WriteOp writeOp) {
      if (writeOp.getOrderedSymsAttr())
        return;

      auto tensorType = dyn_cast<WaveTensorType>(writeOp.getMemory().getType());
      if (!tensorType)
        return;

      writeOp.setOrderedSyms(tensorType.getShape());
    });
  }

  /// Resolve all allocate operations within the given operation using the
  /// provided type converter. Returns failure if any allocation fails to
  /// resolve.
  LogicalResult resolveAllocations(Operation *root,
                                   WaveTypeConverter &typeConverter) {
    LogicalResult result = success();
    root->walk([&](AllocateOp allocateOp) {
      if (isa<MemRefType>(allocateOp.getResult().getType()))
        return;

      auto tensorType = cast<WaveTensorType>(allocateOp.getResult().getType());

      // Only handle shared memory allocations for now.
      if (tensorType.getAddressSpaceValue() != WaveAddressSpace::Shared)
        return;

      WaveExprListAttr distributedShape = allocateOp.getDistributedShape();
      AffineMap shapeMap = distributedShape.getMap();

      // Apply padding to the last dimension of the distributed shape.
      int64_t padding = allocateOp.getPadding();
      if (padding > 0 && shapeMap.getNumResults() > 0) {
        llvm::SmallVector<AffineExpr> results(shapeMap.getResults());
        AffineExpr &lastDim = results.back();
        lastDim = lastDim + getAffineConstantExpr(padding, &getContext());
        shapeMap =
            AffineMap::get(shapeMap.getNumDims(), shapeMap.getNumSymbols(),
                           results, &getContext());
      }

      Type memrefType = typeConverter.convertTensorFromComponents(
          distributedShape.getSymbols(), shapeMap, tensorType.getElementType(),
          tensorType.getAddressSpaceValue());
      if (!memrefType) {
        allocateOp.emitError("failed to create memref type");
        result = failure();
        return;
      }

      // Update the result type in place.
      allocateOp.getResult().setType(memrefType);
    });
    return result;
  }

  void runOnOperation() override {
    auto *waveDialect = getContext().getLoadedDialect<WaveDialect>();
    WalkResult walkResult =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
          // If we hit an AllocateOp before finding hyperparameters, error.
          if (op->getDialect() == waveDialect && isa<AllocateOp>(op)) {
            op->emitError() << "allocate operation with no hyperparameters "
                               "provided by any ancestor";
            return WalkResult::interrupt();
          }

          auto hyperparam = op->getAttrOfType<WaveHyperparameterAttr>(
              WaveDialect::kHyperparameterAttrName);
          if (!hyperparam)
            return WalkResult::advance();

          // Found hyperparameters, set ordered_syms on read/write ops before
          // type conversion loses the dimension ordering information.
          setOrderedSymsOnReadWriteOps(op);

          // Resolve all allocations in this subtree.
          WaveTypeConverter typeConverter(hyperparam);
          if (failed(resolveAllocations(op, typeConverter)))
            return WalkResult::interrupt();

          // Skip children since we already processed this subtree.
          return WalkResult::skip();
        });

    if (walkResult.wasInterrupted())
      return signalPassFailure();

    if (llvm::failed(wave::setNormalFormPassPostcondition(
            wave::WaveNormalForm::ResolvedAllocations |
                wave::WaveNormalForm::OrderedSymsSpecified,
            getOperation())))
      return signalPassFailure();
  }
};

} // namespace
