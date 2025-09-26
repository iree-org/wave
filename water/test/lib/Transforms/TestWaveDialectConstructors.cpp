// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::water::test {
#define GEN_PASS_DEF_TESTWAVEDIALECTCONSTRUCTORSPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir::water::test

static llvm::LogicalResult testCreateTensor(mlir::Operation *op) {
  const static llvm::StringLiteral kFullySpecified = "fully_specified";
  const static llvm::StringLiteral kShape = "shape";
  auto fullySpecifiedAttr = op->getAttrOfType<mlir::BoolAttr>(kFullySpecified);
  if (!fullySpecifiedAttr)
    return op->emitOpError()
           << "expected the \"" << kFullySpecified << "\" bool attribute";

  auto shapeAttr = op->getAttrOfType<mlir::ArrayAttr>(kShape);
  if (!shapeAttr)
    return op->emitOpError()
           << "expected the \"" << kShape << "\" array attribute";

  llvm::SmallVector<wave::WaveSymbolAttr> shapeComponents;
  for (mlir::Attribute a : shapeAttr.getValue()) {
    if (auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(a)) {
      shapeComponents.emplace_back(
          wave::WaveSymbolAttr::get(op->getContext(), symbol.getValue()));
      continue;
    }
    return op->emitOpError() << "expected elements of the \"" << kShape
                             << "\" attribute to be flat symbol references";
  }

  mlir::Type elementType = mlir::IndexType::get(op->getContext());
  auto type = wave::WaveTensorType::getChecked(
      [&]() { return op->emitError(); }, op->getContext(),
      llvm::ArrayRef(shapeComponents), fullySpecifiedAttr.getValue(),
      elementType,
      wave::WaveAddressSpaceAttr::get(op->getContext(),
                                      wave::WaveAddressSpace::Unspecified));
  return llvm::failure(type == nullptr);
}

class TestWaveDialectConstructorPass
    : public mlir::water::test::impl::TestWaveDialectConstructorsPassBase<
          TestWaveDialectConstructorPass> {
public:
  void runOnOperation() override {
    mlir::OperationName createTensorOpName("wave_test.create_tensor",
                                           &getContext());
    mlir::WalkResult walkResult =
        getOperation()->walk([&](mlir::Operation *op) {
          if (op->getName() == createTensorOpName) {
            if (failed(testCreateTensor(op)))
              return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  };
};
