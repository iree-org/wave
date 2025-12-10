// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERALLOCTOALLOCAPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

/// Pass that converts memref.alloc operations to memref.alloca operations.
/// This is required for GPU targets which do not support heap allocations.
class AllocToAllocaPass
    : public water::impl::WaterAllocToAllocaPassBase<AllocToAllocaPass> {
public:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](memref::AllocOp allocOp) {
      rewriter.setInsertionPoint(allocOp);
      auto allocaOp = memref::AllocaOp::create(
          rewriter, allocOp.getLoc(), allocOp.getType(),
          allocOp.getDynamicSizes(), allocOp.getAlignmentAttr());
      rewriter.replaceOp(allocOp, allocaOp.getResult());
    });
  }
};
