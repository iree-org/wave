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
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
  void runOnOperation() override {
    getOperation()->walk([&](AllocateOp allocateOp) {
      if (isa<MemRefType>(allocateOp.getResult().getType()))
        return;

      auto tensorType = cast<WaveTensorType>(allocateOp.getResult().getType());

      // Only handle shared memory allocations for now.
      if (tensorType.getAddressSpaceValue() != WaveAddressSpace::Shared)
        return;

      WaveHyperparameterAttr hyperparams = getHyperparameters(allocateOp);
      if (!hyperparams) {
        allocateOp.emitError("no hyperparameters found for allocate operation");
        return signalPassFailure();
      }

      WaveExprListAttr distributedShape = allocateOp.getDistributedShape();
      WaveTypeConverter typeConverter(hyperparams);
      Type memrefType = typeConverter.convertTensorFromComponents(
          distributedShape.getSymbols(), distributedShape.getMap(),
          tensorType.getElementType(), tensorType.getAddressSpaceValue());
      if (!memrefType) {
        allocateOp.emitError("failed to create memref type");
        return signalPassFailure();
      }

      allocateOp.getResult().setType(memrefType);
    });
  }
};

} // namespace
