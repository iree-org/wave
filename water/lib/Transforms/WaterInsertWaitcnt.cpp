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

/// Waitcnt requirement for synchronization
struct WaitcntRequirement {
  std::optional<unsigned> vmcnt;   // Vector memory operations counter
  std::optional<unsigned> lgkmcnt; // LDS/GDS operations counter

  bool hasRequirement() const {
    return vmcnt.has_value() || lgkmcnt.has_value();
  }

  /// Merge with another requirement (take minimum for conservative join)
  /// Returns true if this requirement changed
  bool merge(const WaitcntRequirement &other) {
    bool changed = false;

    // Take minimum of each counter (lower value = more restrictive)
    if (other.vmcnt.has_value()) {
      if (!vmcnt.has_value() || *other.vmcnt < *vmcnt) {
        vmcnt = other.vmcnt;
        changed = true;
      }
    }
    if (other.lgkmcnt.has_value()) {
      if (!lgkmcnt.has_value() || *other.lgkmcnt < *lgkmcnt) {
        lgkmcnt = other.lgkmcnt;
        changed = true;
      }
    }

    return changed;
  }

  std::optional<unsigned> getLoadCnt() const { return vmcnt; }
  std::optional<unsigned> getStoreCnt() const { return vmcnt; }
  std::optional<unsigned> getDsCnt() const { return lgkmcnt; }
};

/// Lattice state tracking pending asynchronous operations
class WaitcntState : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WaitcntState)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &rhsState = static_cast<const WaitcntState &>(rhs);
    bool changed = false;

    // Merge pending operations
    if (!rhsState.isEmpty()) {
      if (isEmpty()) {
        pendingOps = rhsState.pendingOps;
        changed = true;
      } else {
        // Conservative union: merge all pending operations
        for (Operation *op : rhsState.getPendingOps()) {
          if (!contains(op)) {
            addPendingOp(op);
            changed = true;
          }
        }
      }
    }

    // Merge requirements (take minimum for conservative join)
    if (requirement.merge(rhsState.requirement))
      changed = true;

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
  void clear() {
    pendingOps = std::make_shared<PendingOperations>();
    requirement = WaitcntRequirement();
  }

  /// Set the required waitcnt values
  void setRequirement(const WaitcntRequirement &req) { requirement = req; }

  /// Get the required waitcnt values
  const WaitcntRequirement &getRequirement() const { return requirement; }

  /// Check if there's a waitcnt requirement
  bool hasRequirement() const { return requirement.hasRequirement(); }

  /// Check if a value depends on pending operations and compute required wait
  std::optional<WaitcntRequirement> checkRequirement(Value val) const {
    if (!pendingOps || pendingOps->ops.empty())
      return std::nullopt;

    // Check if val is produced by any pending operation
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return std::nullopt;

    // Find the operation in the pending list
    auto it = llvm::find(pendingOps->ops, defOp);
    if (it == pendingOps->ops.end())
      return std::nullopt;

    // Compute distance from the end of the list
    size_t distanceFromEnd = std::distance(it, pendingOps->ops.end()) - 1;

    WaitcntRequirement req;
    // For vector loads/stores, use vmcnt
    if (isa<vector::LoadOp, vector::StoreOp>(defOp))
      req.vmcnt = distanceFromEnd;

    return req;
  }

private:
  /// Pending asynchronous operations (vector loads/stores)
  std::shared_ptr<PendingOperations> pendingOps;

  /// Required waitcnt before this state (for inserting actual waitcnt ops)
  WaitcntRequirement requirement;
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

    // Always reset requirement - it should not propagate from previous state
    after->setRequirement(WaitcntRequirement());

    // Check if any operands depend on pending operations
    WaitcntRequirement opRequirement;
    for (Value operand : op->getOperands()) {
      if (auto req = before.checkRequirement(operand)) {
        // Merge this requirement (take minimum for conservative wait)
        opRequirement.merge(*req);
      }
    }

    // Set the requirement for this operation
    if (opRequirement.hasRequirement())
      after->setRequirement(opRequirement);

    // Check if this is an async memory operation (vector load/store)
    if (isa<vector::LoadOp, vector::StoreOp>(op)) {
      // Add this operation to the pending list
      after->addPendingOp(op);
    }

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

    // Insert waitcnt operations based on analysis results
    IRRewriter rewriter(&getContext());
    op->walk([&](Operation *operation) {
      // Query the state before this operation
      const WaitcntState *state = solver.lookupState<WaitcntState>(
          solver.getProgramPointAfter(operation));
      if (!state || !state->hasRequirement())
        return;

      const WaitcntRequirement &req = state->getRequirement();

      auto getAttr = [&](std::optional<unsigned> cnt) -> IntegerAttr {
        if (!cnt.has_value())
          return nullptr;
        return rewriter.getI32IntegerAttr(*cnt);
      };

      // Insert wait operation before this operation
      rewriter.setInsertionPoint(operation);
      amdgpu::MemoryCounterWaitOp::create(
          rewriter, operation->getLoc(), getAttr(req.getLoadCnt()),
          getAttr(req.getStoreCnt()), getAttr(req.getDsCnt()), nullptr);
    });
  }
};

} // namespace
