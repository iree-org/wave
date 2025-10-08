// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
#define WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H

#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace wave {

// Common verifier for the optional 'index' attribute used by Wave ops.
mlir::LogicalResult verifyWaveIndexMappings(mlir::Operation *op);

// Trait that checks the 'index' attribute using verifyWaveIndexMappings.
template <typename ConcreteType>
class HasWaveIndexMapping
    : public mlir::OpTrait::TraitBase<ConcreteType, HasWaveIndexMapping> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return verifyWaveIndexMappings(op);
  }
};

mlir::ParseResult parseWaveIndexDict(mlir::OpAsmParser &parser,
                                     mlir::DictionaryAttr &out);
void printWaveIndexDict(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                        mlir::DictionaryAttr dict);

namespace detail {
// Propagate shape information from `from` tensor types to `to` tensor types.
// Expects all fully-specified tensor types to have the same shape, prints an
// error message to `errs` otherwise. Update all under-specified tensor types in
// `to` to be fully specified with the shapes extracted from from. For
// fully-specified types in `to`, check if their shapes match those in `from`
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If there was an error
// due to mismatching/irreconcilable types and an error was printed, returns
// `failure`. Otherwise returns an change indicator.
llvm::FailureOr<mlir::ChangeResult>
identityTypeInferencePropagate(llvm::ArrayRef<wave::WaveTensorType> from,
                               llvm::MutableArrayRef<wave::WaveTensorType> to,
                               llvm::StringRef fromName, llvm::StringRef toName,
                               llvm::raw_ostream &errs);
llvm::FailureOr<mlir::ChangeResult>
propagateShapeInformation(wave::WaveTensorType from, wave::WaveTensorType &to,
                          llvm::StringRef fromName, llvm::StringRef toName,
                          llvm::raw_ostream &errs);

// Check whether the `from` and `to` tensor types have reconcilable shapes and
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If types are
// reconcilable, returns an indicator whether the `to` type will have to be
// updated.
llvm::FailureOr<mlir::ChangeResult>
checkPropagateShapeConflict(wave::WaveTensorType from, wave::WaveTensorType to,
                            llvm::StringRef fromName, llvm::StringRef toName,
                            llvm::raw_ostream &errs);

} // namespace detail

// A trait providing an implementation of the WaveInferTypeOpInterface where
// shapes are propagated from all operands to all results and back as is.
template <typename OpTy>
class IdentityTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, IdentityTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        operandTypes, resultTypes, "operands", "results", errs);
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        resultTypes, operandTypes, "results", "operands", errs);
  }
};

// A trait providing an implementation of the WaveInferTypeOpInterface where no
// shape propagation is needed. E.g. for operations that only have operands and
// no results.
template <typename OpTy>
class NoOpTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, NoOpTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }
};

// Verify that element types of Wave tensors match between LHS and RHS. Emit
// diagnostic errors and return a failure when it is not the case.
namespace detail {
llvm::LogicalResult verifyElementTypesMatch(std::optional<mlir::Location> loc,
                                            llvm::StringRef lhsName,
                                            wave::WaveTensorType lhs,
                                            llvm::StringRef rhsName,
                                            wave::WaveTensorType rhs);

// Verify if two types are compatible:
//   - their symbolic shapes are either equal or at least one of them is
//     underspecified;
//   - their address spaces are either equal or at least one of them is
//     underspecified.
// When it is not the case, return failure and optionally report an error if a
// location is provided.
llvm::LogicalResult verifyTypesCompatible(
    wave::WaveTensorType lhs, wave::WaveTensorType rhs,
    bool includeAddressSpace,
    std::optional<mlir::Location> errorLocation = std::nullopt,
    llvm::StringRef lhsName = "", llvm::StringRef rhsName = "");

// Verify that specified dimensions match between LHS and RHS, the lists of
// dimensions are expected to be co-indexed. Emit diagnostic errors and
// return failure when it is not the case.
llvm::LogicalResult
verifyTypesMatchingDimensions(std::optional<mlir::Location> loc,
                              llvm::StringRef lhsName, wave::WaveTensorType lhs,
                              llvm::ArrayRef<int> lhsDims,
                              llvm::StringRef rhsName, wave::WaveTensorType rhs,
                              llvm::ArrayRef<int> rhsDims);

// Verification logic for the compatible-operands traits. Succeeds if all wave
// tensor-typed operands and results have compatible shapes and, if the
// corresponding flag is set, compatible address spaces.
llvm::LogicalResult
verifyCompatibleOperandsAndResultsOpTrait(mlir::Operation *op,
                                          bool includeAddressSpace);
}; // namespace detail

template <typename OpTy>
class CompatibleOperandsAndResultsOpTrait
    : public mlir::OpTrait::TraitBase<OpTy,
                                      CompatibleOperandsAndResultsOpTrait> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyCompatibleOperandsAndResultsOpTrait(
        op, /*includeAddressSpace=*/true);
  }
};

template <typename OpTy>
class CompatibleOperandsAndResultsIgnoreSpaceOpTrait
    : public mlir::OpTrait::TraitBase<
          OpTy, CompatibleOperandsAndResultsIgnoreSpaceOpTrait> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    return detail::verifyCompatibleOperandsAndResultsOpTrait(
        op, /*includeAddressSpace=*/false);
  }
};
} // namespace wave

#include "water/Dialect/Wave/IR/WaveOpInterfaces.h.inc"

#endif // WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
