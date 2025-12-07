// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::water {
#define GEN_PASS_DEF_WATERINSERTWAITCNT
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

/// Shared pending operations list for structural sharing
struct PendingOperations {
  SmallVector<Operation *> ops;
};

/// Lattice state tracking pending asynchronous operations
class WaitcntState : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WaitcntState)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &rhsState = static_cast<const WaitcntState &>(rhs);

    // If rhs has no pending ops, no change needed
    if (rhsState.isEmpty())
      return ChangeResult::NoChange;

    // If we have no pending ops, just share rhs's state
    if (isEmpty()) {
      pendingOps = rhsState.pendingOps;
      return ChangeResult::Change;
    }

    // Conservative union: merge all pending operations
    bool changed = false;
    for (Operation *op : rhsState.getPendingOps()) {
      if (!contains(op)) {
        addPendingOp(op);
        changed = true;
      }
    }

    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    os << "WaitcntState: " << size() << " pending ops";
  }

  /// Add a pending operation (copy-on-write)
  void addPendingOp(Operation *op) {
    auto newPending = std::make_shared<PendingOperations>();
    if (pendingOps)
      newPending->ops = pendingOps->ops;
    newPending->ops.push_back(op);
    pendingOps = newPending;
  }

  /// Get pending operations (read-only)
  ArrayRef<Operation *> getPendingOps() const {
    return pendingOps ? ArrayRef<Operation *>(pendingOps->ops)
                      : ArrayRef<Operation *>();
  }

  /// Check if this operation is in the pending list
  bool contains(Operation *op) const {
    return pendingOps && llvm::is_contained(pendingOps->ops, op);
  }

  /// Check if empty
  bool isEmpty() const { return !pendingOps || pendingOps->ops.empty(); }

  /// Get size
  size_t size() const { return pendingOps ? pendingOps->ops.size() : 0; }

  /// Initialize to empty state
  void clear() { pendingOps = std::make_shared<PendingOperations>(); }

private:
  /// Pending asynchronous operations (vector loads/stores)
  std::shared_ptr<PendingOperations> pendingOps;
};

/// Dense forward dataflow analysis for waitcnt insertion
class WaitcntAnalysis : public DenseForwardDataFlowAnalysis<WaitcntState> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void setToEntryState(WaitcntState *lattice) override {
    // Entry state: empty pending operations
    lattice->clear();
  }

  LogicalResult visitOperation(Operation *op, const WaitcntState &before,
                               WaitcntState *after) override {
    // Start with the state before this operation
    *after = before;

    // Check if this is an async memory operation (vector load/store)
    if (isa<vector::LoadOp, vector::StoreOp>(op)) {
      // Add this operation to the pending list
      after->addPendingOp(op);
    }

    // Check if this operation uses a value produced by a pending operation
    // If so, we need to insert a waitcnt before this operation
    // (We'll handle actual insertion in a separate pass over the IR)

    return success();
  }
};

/// Pass that inserts wait/synchronization instructions for asynchronous
/// memory operations. This is analogous to LLVM's SIInsertWaitcnts pass.
class WaterInsertWaitcntPass
    : public water::impl::WaterInsertWaitcntBase<WaterInsertWaitcntPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();

    // Set up the dataflow solver
    DataFlowSolver solver;
    solver.load<WaitcntAnalysis>();

    // Run the analysis
    if (failed(solver.initializeAndRun(op))) {
      signalPassFailure();
      return;
    }

    // TODO: Use the analysis results to insert waitcnt instructions
    // For now, just run the analysis to test the framework
  }
};

} // namespace
