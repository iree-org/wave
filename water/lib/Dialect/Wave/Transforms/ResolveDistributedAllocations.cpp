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
      Type memrefType = typeConverter.convertTensorFromComponents(
          distributedShape.getSymbols(), distributedShape.getMap(),
          tensorType.getElementType(), tensorType.getAddressSpaceValue());
      if (!memrefType) {
        allocateOp.emitError("failed to create memref type");
        result = failure();
        return;
      }

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

          // Found hyperparameters, resolve all allocations in this subtree.
          WaveTypeConverter typeConverter(hyperparam);
          if (failed(resolveAllocations(op, typeConverter)))
            return WalkResult::interrupt();

          // Skip children since we already processed this subtree.
          return WalkResult::skip();
        });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
