// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Visitors.h"
#include "mlir/Support/WalkResult.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

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

  /// Resolve all allocate and view operations within the given operation using
  /// the provided type converter. Returns failure if any allocation fails to
  /// resolve.
  LogicalResult resolveMemoryOps(Operation *root,
                                 WaveTypeConverter &typeConverter) {
    WalkResult walkResult = root->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(op)
          .Case<wave::AllocateOp, wave::ViewOp>([&](auto memOp) {
            if (isa<MemRefType>(memOp.getResult().getType()))
              return WalkResult::advance();

            auto tensorType = cast<WaveTensorType>(memOp.getResult().getType());

            // Only handle shared memory for now.
            if (tensorType.getAddressSpaceValue() != WaveAddressSpace::Shared)
              return WalkResult::skip();

            WaveExprListAttr distributedShape = memOp.getDistributedShape();
            Type memrefType = typeConverter.convertTensorFromComponents(
                distributedShape.getSymbols(), distributedShape.getMap(),
                tensorType.getElementType(), tensorType.getAddressSpaceValue());
            if (!memrefType) {
              memOp.emitError("failed to create memref type");
              return WalkResult::interrupt();
            }

            // Update the result type in place.
            memOp.getResult().setType(memrefType);

            return WalkResult::advance();
          })
          .Default([&](Operation *op) { return WalkResult::advance(); });
    });

    return llvm::failure(walkResult.wasInterrupted());
  }

  void runOnOperation() override {
    auto *waveDialect = getContext().getLoadedDialect<WaveDialect>();
    WalkResult walkResult =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
          // If we hit an AllocateOp or ViewOp before finding hyperparameters,
          // error.
          if (op->getDialect() == waveDialect &&
              (isa<AllocateOp>(op) || isa<ViewOp>(op))) {
            op->emitError()
                << "allocate/view operation with no hyperparameters "
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

          // Resolve all allocations and views in this subtree.
          WaveTypeConverter typeConverter(hyperparam);
          if (failed(resolveMemoryOps(op, typeConverter)))
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
