// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
#define WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace wave {

class WaveTensorType;

//-----------------------------------------------------------------------------
// HasWaveIndexMapping trait
//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface and implementation traits
//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------
// WaveElementsPerThreadOpInterface
//-----------------------------------------------------------------------------

// Lattice for propagating the "elements per thread" value across wave dialect
// operations. In addition to the bottom and top states, it can represent a
// concrete state manifest as an integer value. The JOIN function is defined by
// the following table:
//
// JOIN       top         concrete        bottom
// top        top         top             top
// concrete   top         concrete|top*   concrete
// bottom     top         concrete        bottom
//   * if two concrete values are equal, their JOIN is equal to each of them,
//     otherwise it is considered a propagation conflict and results in the top
//     state.
class ElementsPerThreadLatticeValue {
public:
  // Usage as lattice requires value/storage classes to be default- and
  // copy-constructible, as well as copy-assignable.
  ElementsPerThreadLatticeValue() : value(kBottomTag) {}
  ElementsPerThreadLatticeValue(const ElementsPerThreadLatticeValue &) =
      default;
  ElementsPerThreadLatticeValue &
  operator=(const ElementsPerThreadLatticeValue &other) = default;

  // Create a lattice value in the concrete state.
  explicit ElementsPerThreadLatticeValue(uint64_t value) : value(value) {
    assert(value != kTopTag && "please use top() instead");
    assert(value != kBottomTag && "please use bottom() instead");
  }

  // Create a lattice value in the bottom state.
  static ElementsPerThreadLatticeValue bottom() {
    ElementsPerThreadLatticeValue result(0);
    result.value = kBottomTag;
    return result;
  }

  // Create a lattice value in the top state.
  static ElementsPerThreadLatticeValue top() {
    ElementsPerThreadLatticeValue result(0);
    result.value = kTopTag;
    return result;
  }

  // Usage as lattice requires value/storage class instances to be comparable.
  bool operator==(const ElementsPerThreadLatticeValue &other) const {
    return value == other.value;
  }
  bool operator!=(const ElementsPerThreadLatticeValue &other) const {
    return !operator==(other);
  }

  // Examine the state of the lattice object.
  bool isBottom() const { return value == kBottomTag; }
  bool isTop() const { return value == kTopTag; }
  uint64_t getValue() const {
    assert(!isBottom() && !isTop());
    return value;
  }

  // JOIN two lattice objects and return the result.
  static ElementsPerThreadLatticeValue
  join(const ElementsPerThreadLatticeValue &lhs,
       const ElementsPerThreadLatticeValue &rhs);

  // XXX: backward analysis calls `meet` instead of `join`, but it isn't related
  // to the direction of the analysis. Just defer to join.
  static ElementsPerThreadLatticeValue
  meet(const ElementsPerThreadLatticeValue &lhs,
       const ElementsPerThreadLatticeValue &rhs) {
    return join(lhs, rhs);
  }

  // Forcibly assign the current value of the lattice. This MUST NOT be used in
  // the transfer functions as it may be moving the instance back on the lattice
  // and therefore breaking the analysis convergence guarantees due to
  // non-monotonicity. This is useful during forceful initialization to override
  // the quirk of the dataflow framework using the same function
  // (`setToEntry/ExitState`) to both initialize the analysis and to indicate
  // failure to analyze. Those functions can keep setting the lattice to the top
  // state.
  void unsafeSet(const ElementsPerThreadLatticeValue &value) {
    this->value = value.value;
  }

  // Print the lattice value to the given output stream.
  void print(llvm::raw_ostream &os) const;

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

private:
  uint64_t value;
  constexpr const static uint64_t kBottomTag =
      std::numeric_limits<uint64_t>::max() - 1;
  constexpr const static uint64_t kTopTag =
      std::numeric_limits<uint64_t>::max();
};

namespace detail {

// Propagate elements per thread lattice values from the list in `from` to the
// list in `to`. The lattice values in the `from` list are expected to be
// compatible, i.e., their JOIN should not result in the top value unless one of
// them is already the lattice top. Report errors about conflicts to the error
// stream provided and return failure. Otherwise return whether any of the `to`
// lattices was updated.
llvm::FailureOr<mlir::ChangeResult> identityElementsPerThreadPropagate(
    llvm::ArrayRef<ElementsPerThreadLatticeValue> from,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs);

// Propagate elements per thread lattice values from `from` to the
// `mutableValues` list and check their compatibility with those in the
// `immutableValues` list i.e., their JOIN should not result in the top value
// unless one of them is already the lattice top. Report errors about conflicts
// to the error stream provided and return failure. Otherwise return whether any
// of the `mutableValues` lattices was updated.
llvm::FailureOr<mlir::ChangeResult>
checkAndPropagateElementsPerThreadFromConstant(
    const ElementsPerThreadLatticeValue &from,
    llvm::ArrayRef<ElementsPerThreadLatticeValue> immutableValues,
    llvm::MutableArrayRef<ElementsPerThreadLatticeValue> mutableValues,
    llvm::StringRef fromName, llvm::StringRef immutableName,
    llvm::StringRef mutableName, llvm::raw_ostream &errs);

} // namespace detail

// Trait implementing the methods of the WaveElementsPerThreadOpInterface based
// on the `elements_per_thead` attribute.
template <typename OpTy>
class AttrBasedElementsPerThreadOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, AttrBasedElementsPerThreadOpTrait> {
public:
  // Propagate `elements_per_thread` value to results.
  llvm::FailureOr<mlir::ChangeResult> propagateElementsPerThreadForward(
      llvm::ArrayRef<ElementsPerThreadLatticeValue> operandTypes,
      llvm::MutableArrayRef<ElementsPerThreadLatticeValue> resultTypes,
      llvm::raw_ostream &errs) {
    std::optional<int64_t> elementsPerThread =
        llvm::cast<OpTy>(this->getOperation()).getElementsPerThread();
    if (!elementsPerThread)
      return mlir::ChangeResult::NoChange;

    return detail::checkAndPropagateElementsPerThreadFromConstant(
        ElementsPerThreadLatticeValue(*elementsPerThread), operandTypes,
        resultTypes, "elements_per_thread attribute", "operand", "result",
        errs);
  }

  // Propagate `elements_per_thread` value to operands.
  llvm::FailureOr<mlir::ChangeResult> propagateElementsPerThreadBackward(
      llvm::MutableArrayRef<ElementsPerThreadLatticeValue> operandTypes,
      llvm::ArrayRef<ElementsPerThreadLatticeValue> resultTypes,
      llvm::raw_ostream &errs) {
    std::optional<int64_t> elementsPerThread =
        llvm::cast<OpTy>(this->getOperation()).getElementsPerThread();
    if (!elementsPerThread)
      return mlir::ChangeResult::NoChange;

    return detail::checkAndPropagateElementsPerThreadFromConstant(
        ElementsPerThreadLatticeValue(*elementsPerThread), resultTypes,
        operandTypes, "elements_per_thread attribute", "result", "operand",
        errs);
  }
};

// Trait implementing the methods of the WaveElementsPerThreadOpInterface with
// information flowing between operands and results.
template <typename OpTy>
class IdentityElementsPerThreadOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, IdentityElementsPerThreadOpTrait> {
public:
  // Propagate from operands to results.
  llvm::FailureOr<mlir::ChangeResult> propagateElementsPerThreadForward(
      llvm::ArrayRef<ElementsPerThreadLatticeValue> operandTypes,
      llvm::MutableArrayRef<ElementsPerThreadLatticeValue> resultTypes,
      llvm::raw_ostream &errs) {
    return wave::detail::identityElementsPerThreadPropagate(
        operandTypes, resultTypes, "operands", "results", errs);
  }

  // Propagate from results to operands.
  llvm::FailureOr<mlir::ChangeResult> propagateElementsPerThreadBackward(
      llvm::MutableArrayRef<ElementsPerThreadLatticeValue> operandTypes,
      llvm::ArrayRef<ElementsPerThreadLatticeValue> resultTypes,
      llvm::raw_ostream &errs) {
    return wave::detail::identityElementsPerThreadPropagate(
        resultTypes, operandTypes, "results", "operands", errs);
  }
};

// Trait indicating to the elements per thread propagation analysis that the
// operation intentionally does not use or affect elements per thread.
template <typename OpTy>
class NoOpElementsPerThreadOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, NoOpElementsPerThreadOpTrait> {};

} // namespace wave

#include "water/Dialect/Wave/IR/WaveOpInterfaces.h.inc"

#endif // WATER_DIALECT_WAVE_IR_WAVEINTERFACES_H
