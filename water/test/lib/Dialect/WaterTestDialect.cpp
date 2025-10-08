// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "WaterTestDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "WaterTestDialect.cpp.inc"

void mlir::water::test::WaterTestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "WaterTestDialectOps.cpp.inc"
      >();
};

namespace mlir::water::test {
void registerWaterTestDialect(DialectRegistry &registry) {
  registry.insert<WaterTestDialect>();
}
} // namespace mlir::water::test

using namespace mlir::water::test;

llvm::FailureOr<mlir::ChangeResult> WaveFailPropagationOp::propagateForward(
    llvm::ArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::MutableArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getForward()) {
    errs << "intentionally failed to propagate forward";
    return mlir::failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      operandTypes, resultTypes, "operands", "results", errs);
}

llvm::FailureOr<mlir::ChangeResult> WaveFailPropagationOp::propagateBackward(
    llvm::MutableArrayRef<::wave::WaveTensorType> operandTypes,
    llvm::ArrayRef<::wave::WaveTensorType> resultTypes,
    llvm::raw_ostream &errs) {
  if (getBackward()) {
    errs << "intentionally failed to propagate backward";
    return mlir::failure();
  }
  return wave::detail::identityTypeInferencePropagate(
      resultTypes, operandTypes, "results", "operands", errs);
}

#define GET_OP_CLASSES
#include "WaterTestDialectOps.cpp.inc"
