// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "WaterTestDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "WaterTestDialect.cpp.inc"

// Test normal form attribute.
#define GET_ATTRDEF_CLASSES
#include "TestNormalFormAttr.cpp.inc"

void mlir::water::test::WaterTestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "WaterTestDialectOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "TestNormalFormAttr.cpp.inc"
      >();
};

namespace mlir::water::test {
void registerWaterTestDialect(DialectRegistry &registry) {
  registry.insert<WaterTestDialect>();
}
} // namespace mlir::water::test

using namespace mlir::water::test;

//-----------------------------------------------------------------------------
// NoIndexTypesAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult
NoIndexTypesAttr::verifyType(llvm::function_ref<InFlightDiagnostic()> emitError,
                             Type type) const {
  if (!type)
    return llvm::success();

  if (type.isIndex())
    return emitError() << "normal form prohibits index types";

  return llvm::success();
}

//-----------------------------------------------------------------------------
// NoInvalidOpsAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult NoInvalidOpsAttr::verifyOperation(
    llvm::function_ref<InFlightDiagnostic()> emitError, Operation *op) const {
  if (isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(op))
    return emitError() << "normal form prohibits division operations";

  return llvm::success();
}

//-----------------------------------------------------------------------------
// NoInvalidAttrsAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult NoInvalidAttrsAttr::verifyAttribute(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute attr) const {
  if (!attr)
    return llvm::success();

  if (auto strAttr = llvm::dyn_cast<StringAttr>(attr)) {
    if (strAttr.getValue() == "invalid")
      return emitError()
             << "normal form prohibits 'invalid' string attribute values";
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// NoForbiddenSymbolsAttr interface implementations.
//-----------------------------------------------------------------------------

llvm::LogicalResult NoForbiddenSymbolsAttr::verifyAttribute(
    llvm::function_ref<InFlightDiagnostic()> emitError, Attribute attr) const {
  if (!attr)
    return llvm::success();

  if (auto symbolAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(attr)) {
    if (symbolAttr.getName() == "forbidden")
      return emitError() << "normal form prohibits 'forbidden' symbol in types";
  }

  return llvm::success();
}

//-----------------------------------------------------------------------------
// WaveFailPropagationOp implementations.
//-----------------------------------------------------------------------------

llvm::FailureOr<ChangeResult> WaveFailPropagationOp::propagateForward(
    llvm::ArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getForward()) {
    errs << "intentionally failed to propagate forward";
    return failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      operandTypes, resultTypes, "operands", "results", errs);
}

llvm::FailureOr<ChangeResult> WaveFailPropagationOp::propagateBackward(
    llvm::MutableArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getBackward()) {
    errs << "intentionally failed to propagate backward";
    return failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      resultTypes, operandTypes, "results", "operands", errs);
}

LogicalResult WaveFailPropagationOp::finalizeTypeInference() {
  return success();
}

#define GET_OP_CLASSES
#include "WaterTestDialectOps.cpp.inc"
