// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/PatternMatch.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace mlir::water::test {
#define GEN_PASS_DEF_TESTWAVEDIALECTFUNCTIONSPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir::water::test

static llvm::LogicalResult testCreateTensor(Operation *op) {
  const static llvm::StringLiteral kFullySpecified = "fully_specified";
  const static llvm::StringLiteral kShape = "shape";
  auto fullySpecifiedAttr = op->getAttrOfType<BoolAttr>(kFullySpecified);
  if (!fullySpecifiedAttr)
    return op->emitOpError()
           << "expected the \"" << kFullySpecified << "\" bool attribute";

  auto shapeAttr = op->getAttrOfType<ArrayAttr>(kShape);
  if (!shapeAttr)
    return op->emitOpError()
           << "expected the \"" << kShape << "\" array attribute";

  llvm::SmallVector<wave::WaveSymbolAttr> shapeComponents;
  for (Attribute a : shapeAttr.getValue()) {
    if (auto symbol = llvm::dyn_cast<FlatSymbolRefAttr>(a)) {
      shapeComponents.emplace_back(
          wave::WaveSymbolAttr::get(op->getContext(), symbol.getValue()));
      continue;
    }
    return op->emitOpError() << "expected elements of the \"" << kShape
                             << "\" attribute to be flat symbol references";
  }

  Type elementType = IndexType::get(op->getContext());
  auto type = wave::WaveTensorType::getChecked(
      [&]() { return op->emitError(); }, op->getContext(),
      llvm::ArrayRef(shapeComponents), fullySpecifiedAttr.getValue(),
      elementType,
      wave::WaveAddressSpaceAttr::get(op->getContext(),
                                      wave::WaveAddressSpace::Unspecified));
  return llvm::failure(type == nullptr);
}

class TestWaveDialectFunctionsPass
    : public mlir::water::test::impl::TestWaveDialectFunctionsPassBase<
          TestWaveDialectFunctionsPass> {
public:
  void runOnOperation() override {
    OperationName createTensorOpName("wave_test.create_tensor", &getContext());
    WalkResult walkResult = getOperation()->walk([&](Operation *op) {
      if (op->getName() == createTensorOpName) {
        if (failed(testCreateTensor(op)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](wave::IterateOp iterateOp) {
      if (iterateOp->getAttrOfType<UnitAttr>("wave_test.make_isolated")) {
        iterateOp.makeIsolated(rewriter);
      } else if (iterateOp->getAttrOfType<UnitAttr>(
                     "wave_test.make_non_isolated")) {
        iterateOp.makeNonIsolated(rewriter);
      }
    });
  };
};
