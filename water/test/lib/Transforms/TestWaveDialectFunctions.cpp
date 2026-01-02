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

static llvm::LogicalResult testCreateIndexMapping(mlir::Operation *op) {
  const static llvm::StringLiteral kSymbols = "symbols";
  const static llvm::StringLiteral kStart = "start";
  const static llvm::StringLiteral kStep = "step";
  const static llvm::StringLiteral kStride = "stride";

  auto symbolsAttr = op->getAttrOfType<mlir::ArrayAttr>(kSymbols);
  if (!symbolsAttr)
    return op->emitOpError()
           << "expected the \"" << kSymbols << "\" array attribute";

  auto startAttr = op->getAttrOfType<mlir::AffineMapAttr>(kStart);
  if (!startAttr)
    return op->emitOpError()
           << "expected the \"" << kStart << "\" affine map attribute";

  auto stepAttr = op->getAttrOfType<mlir::AffineMapAttr>(kStep);
  if (!stepAttr)
    return op->emitOpError()
           << "expected the \"" << kStep << "\" affine map attribute";

  auto strideAttr = op->getAttrOfType<mlir::AffineMapAttr>(kStride);
  if (!strideAttr)
    return op->emitOpError()
           << "expected the \"" << kStride << "\" affine map attribute";

  return llvm::success(wave::WaveIndexMappingAttr::getChecked(
                           op->getLoc(), op->getContext(),
                           symbolsAttr.getValue(), startAttr.getValue(),
                           stepAttr.getValue(),
                           strideAttr.getValue()) != nullptr);
}

class TestWaveDialectFunctionsPass
    : public mlir::water::test::impl::TestWaveDialectFunctionsPassBase<
          TestWaveDialectFunctionsPass> {
public:
  void runOnOperation() override {
    mlir::OperationName createTensorOpName("wave_test.create_tensor",
                                           &getContext());
    mlir::OperationName createIndexMappingOpName(
        "wave_test.create_index_mapping", &getContext());
    mlir::WalkResult walkResult =
        getOperation()->walk([&](mlir::Operation *op) {
          if (op->getName() == createTensorOpName) {
            if (failed(testCreateTensor(op)))
              return mlir::WalkResult::interrupt();
          } else if (op->getName() == createIndexMappingOpName) {
            if (failed(testCreateIndexMapping(op)))
              return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
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
