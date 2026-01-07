// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include <cstdint>

namespace mlir::water {

/// Lattice state codes, mutually exclusive.
enum class InUseLatticeState : uint8_t {
  Uninitialized,
  Used,
  Unused,
  Undecidable
};

/// Storage for the lattice value, the structure is a canonical diamond:
///
/// ```
///       Undecidable
///        /        \
///     Used        Unused
///        \        /
///      Uninitialized
/// ```
class InUseLatticeStorage {
public:
  /// Lattice values are created in the bottom state.
  InUseLatticeStorage() : state(InUseLatticeState::Uninitialized) {}
  InUseLatticeStorage(const InUseLatticeState &state) : state(state) {}

  /// Joins two lattice values, potentially moving towards the top of the
  /// lattice. Returns the new value.
  static InUseLatticeStorage join(const InUseLatticeStorage &lhs,
                                  const InUseLatticeStorage &rhs) {
    if (lhs.state == rhs.state)
      return lhs.state;

    if (lhs.state == InUseLatticeState::Uninitialized)
      return rhs.state;

    if (rhs.state == InUseLatticeState::Uninitialized)
      return lhs.state;

    return InUseLatticeState::Undecidable;
  }

  /// Compares two lattice values for equality.
  bool operator==(const InUseLatticeStorage &other) const {
    return state == other.state;
  }

  /// DANGEROUS direct setter of the internal lattice state. This does not
  /// guarantee monotonicity, and MUST NOT be used inside dataflow propagation
  /// as its convergence is provided by monotonicity. Only for use by experts as
  /// a last resort.
  void unsafeSet(const InUseLatticeState &state) { this->state = state; }

  /// Prints the lattice value into the provided stream.
  // TODO: surface into the upstream documentation that this is mandatory.
  llvm::raw_ostream &print(llvm::raw_ostream &os) const {
    switch (state) {
    case InUseLatticeState::Uninitialized:
      return os << "uninitialized";
    case InUseLatticeState::Undecidable:
      return os << "undecidable";
    case InUseLatticeState::Unused:
      return os << "unused";
    case InUseLatticeState::Used:
      return os << "used";
    }
    llvm_unreachable("unsupport enum case");
  }

  /// Returns the lattice bottom value.
  static InUseLatticeStorage bottom() {
    return InUseLatticeState::Uninitialized;
  }

  /// Returns the lattice top value.
  static InUseLatticeStorage top() { return InUseLatticeState::Undecidable; }

  /// Returns the lattice enum state.
  const InUseLatticeState &get() const { return state; }

private:
  InUseLatticeState state;
};

/// The lattice indicating whether a value is used, e.g. for address computation
/// or control flow. The lattice itself does not indicate that the value is used
/// for.
class InUseLattice : public mlir::dataflow::Lattice<InUseLatticeStorage> {
public:
  using Lattice::Lattice;
};

/// Dataflow analysis indicating whether a value is used in address computation
/// or in control flow, and therefore must be present in the speculative
/// out-of-bounds checker. Propagates such known uses backwards in the dataflow
/// sense.
class InUseAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<InUseLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  /// Directly override the great-grandparent initialization function to
  /// graciously handle the non-interprocedural mode: operands of the function
  /// return are considered unused for control flow instead of defaulting to
  /// unanalyzable. This is impossible to express otherwise.
  LogicalResult initialize(Operation *top) override;

  /// This is called when an unanalyzable situation happens (and may also be
  /// called during initialization, bizarrely). Set the lattice to the top
  /// state to indicate the unanalyzable state.
  void setToExitState(InUseLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InUseLatticeStorage::top()));
  }

  /// Directly override the great-grandparent visitation function to graciously
  /// handle the non-interprocedural mode: operands of the function return are
  /// considered unused for control flow instead of defaulting to unanalyzable.
  /// This is impossible to express otherwise. All other cases are deferred to
  /// `visitOperation` and friends below using the default logic.
  LogicalResult visit(ProgramPoint *point) override;

  /// Main visitation function, updates operand lattices based on result
  /// lattices and operation-specific information.
  LogicalResult visitOperation(Operation *op, ArrayRef<InUseLattice *> operands,
                               ArrayRef<const InUseLattice *> results) override;

  /// Calls are not supported yet.
  void visitCallOperand(OpOperand &) override { assert(false && "no calls"); }

  /// Visitation function along branching and region control flow.
  void visitBranchOperand(OpOperand &opOperand) override;

  /// Visit the non-forwarded arguments of a region, such as the
  /// induction variables of a loop.
  void
  visitNonControlFlowArguments(RegionSuccessor & /*successor*/,
                               ArrayRef<BlockArgument> /*arguments*/) override {
    // nothing
  }
};

} // namespace mlir::water
