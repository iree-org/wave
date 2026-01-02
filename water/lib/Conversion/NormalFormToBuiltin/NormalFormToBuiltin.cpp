// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Conversion/NormalFormToBuiltin/NormalFormToBuiltin.h"
#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"

#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
#define GEN_PASS_DEF_LOWERAFFINEPASS
#include "water/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

/// Lower `normalform.module` to `buitin.module`.
class ModuleOpLowering : public OpRewritePattern<normalform::ModuleOp> {
public:
  using OpRewritePattern<normalform::ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(normalform::ModuleOp nfModule,
                                PatternRewriter &rewriter) const override {

    ModuleOp builtinModule =
        ModuleOp::create(rewriter, nfModule.getLoc(), nfModule.getName());
    builtinModule->setAttr(normalform::NormalFormDialect::kNormalFormAttrName,
                           nfModule.getNormalFormAttr());
    rewriter.inlineRegionBefore(nfModule.getRegion(), builtinModule.getBody());

    // Remove the terminator block that was automatically added by builder
    rewriter.eraseBlock(&builtinModule.getBodyRegion().back());
    rewriter.eraseOp(nfModule);
    return success();
  }
};

} // namespace

void mlir::populateNormalFormToBuiltinConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ModuleOpLowering>(patterns.getContext());
  // clang-format on
}
