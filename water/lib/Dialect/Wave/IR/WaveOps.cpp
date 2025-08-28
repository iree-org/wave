// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"

// Update negative indices in the array to positive equivalents given the total
// rank, e.g. -1 and -3 get updated to 3 and 1, respectively, for the rank of 4.
static void updateNegativeIndices(llvm::MutableArrayRef<int> indices,
                                  int rank) {
  for (int &index : indices) {
    if (index < 0)
      index += rank;
  }
}

// Verify that specified dimensions match between LHS and RHS, the lists of
// dimensions are expected to be co-indexed. Emit diagnostic errors and return
// failure when it is not the case.
static mlir::LogicalResult verifyTypesMatchingDimensions(
    mlir::Location loc, llvm::StringRef lhsName, wave::WaveTensorType lhs,
    llvm::ArrayRef<int> lhsDims, llvm::StringRef rhsName,
    wave::WaveTensorType rhs, llvm::ArrayRef<int> rhsDims) {
  // TODO: check whether it is possible to turn this into a trait.

  assert(lhsDims.size() == rhsDims.size() &&
         "expected lhs and rhs dim lists to be co-indexed");

  // Under-specified types are okay everywhere.
  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return mlir::success();

  llvm::SmallVector<int> lhsDimsVec(lhsDims), rhsDimsVec(rhsDims);
  updateNegativeIndices(lhsDimsVec, lhs.getRank());
  updateNegativeIndices(rhsDimsVec, rhs.getRank());
  for (auto &&[lhsDim, rhsDim] : llvm::zip_equal(lhsDimsVec, rhsDimsVec)) {
    wave::WaveSymbolAttr lhsExpr = lhs.getShape()[lhsDim];
    wave::WaveSymbolAttr rhsExpr = rhs.getShape()[rhsDim];
    if (lhsExpr == rhsExpr)
      continue;

    return mlir::emitError(loc)
           << "expected " << lhsName << " dimension #" << lhsDim << " ("
           << lhsExpr << ") to match " << rhsName << " dimension #" << rhsDim
           << " (" << rhsExpr << ")";
  }
  return mlir::success();
}

// Verify that element types of Wave tensors match between LHS and RHS. Emit
// diagnostic errors and return a failure when it is not the case.
static mlir::LogicalResult verifyElementTypesMatch(mlir::Location loc,
                                                   llvm::StringRef lhsName,
                                                   wave::WaveTensorType lhs,
                                                   llvm::StringRef rhsName,
                                                   wave::WaveTensorType rhs) {
  if (lhs.getElementType() == rhs.getElementType())
    return mlir::success();

  return mlir::emitError(loc)
         << "expected " << lhsName << " and " << rhsName
         << " elemental types to match, got " << lhs.getElementType() << ", "
         << rhs.getElementType();
}

mlir::LogicalResult wave::MmaOp::verify() {
  WaveTensorType lhsType = getLhs().getType();
  WaveTensorType rhsType = getRhs().getType();
  WaveTensorType accumulatorType = getAccumulator().getType();

  if (failed(
          verifyElementTypesMatch(getLoc(), "LHS", lhsType, "RHS", rhsType)) ||
      failed(verifyElementTypesMatch(getLoc(), "LHS", lhsType, "accumulator",
                                     accumulatorType)))
    return mlir::failure();

  return mlir::failure(
      verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {1}, "RHS",
                                    rhsType, {0})
          .failed() ||
      verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {0},
                                    "accumulator", accumulatorType, {0})
          .failed() ||
      verifyTypesMatchingDimensions(getLoc(), "RHS", rhsType, {1},
                                    "accumulator", accumulatorType, {1})
          .failed());
}
