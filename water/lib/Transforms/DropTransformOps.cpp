// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERDROPTRANSFORMOPSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

class DropTransformOpsPass
    : public water::impl::WaterDropTransformOpsPassBase<DropTransformOpsPass> {
public:
  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
      if (isa_and_nonnull<transform::TransformDialect>(funcOp->getDialect())) {
        funcOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
