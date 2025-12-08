// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "water-insert-waitcnt"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::water {
#define GEN_PASS_DEF_WATERINSERTWAITCNT
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

static std::optional<Value> isLoadOp(Operation *op) {
  if (auto load = dyn_cast<vector::LoadOp>(op))
    return load.getBase();

  return std::nullopt;
}

static std::optional<Value> isStoreOp(Operation *op) {
  if (auto store = dyn_cast<vector::StoreOp>(op))
    return store.getBase();

  return std::nullopt;
}

static std::optional<Value> isLoadOrStoreOp(Operation *op) {
  if (auto load = isLoadOp(op))
    return load;
  if (auto store = isStoreOp(op))
    return store;

  return std::nullopt;
}

static bool isWorkgroupAddressSpace(MemRefType type) {
  auto attr = dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
  return attr && attr.getValue() == gpu::AddressSpace::Workgroup;
}

/// Shared pending operations list for structural sharing
struct PendingOperations {
  PendingOperations() = default;
  PendingOperations(SmallVector<Operation *> &&ops) : ops(std::move(ops)) {}
  SmallVector<Operation *> ops;
};

/// Waitcnt requirement for synchronization
struct WaitcntRequirement {
  std::optional<unsigned> load_cnt;
  std::optional<unsigned> ds_cnt;

  bool hasRequirement() const {
    return load_cnt.has_value() || ds_cnt.has_value();
  }

  /// Merge with another requirement (take minimum for conservative join)
  /// Returns true if this requirement changed
  bool merge(const WaitcntRequirement &other) {
    bool changed = false;

    // Take minimum of each counter (lower value = more restrictive)
    if (other.load_cnt.has_value()) {
      if (!load_cnt.has_value() || *other.load_cnt < *load_cnt) {
        load_cnt = other.load_cnt;
        changed = true;
      }
    }
    if (other.ds_cnt.has_value()) {
      if (!ds_cnt.has_value() || *other.ds_cnt < *ds_cnt) {
        ds_cnt = other.ds_cnt;
        changed = true;
      }
    }

    return changed;
  }

  std::optional<unsigned> getLoadCnt() const { return load_cnt; }
  std::optional<unsigned> getStoreCnt() const { return std::nullopt; }
  std::optional<unsigned> getDsCnt() const { return ds_cnt; }

  bool isSameCounterType(const WaitcntRequirement &other) const {
    return load_cnt.has_value() == other.load_cnt.has_value() ||
           ds_cnt.has_value() == other.ds_cnt.has_value();
  }

  static WaitcntRequirement getOperationRequirement(Operation *op, bool zero) {
    WaitcntRequirement req;
    if (std::optional<Value> base = isLoadOrStoreOp(op)) {
      auto memrefType = cast<MemRefType>(base->getType());
      if (isWorkgroupAddressSpace(memrefType)) {
        req.ds_cnt = zero ? 0 : 1;
      } else {
        req.load_cnt = zero ? 0 : 1;
      }
    }
    return req;
  }

  WaitcntRequirement operator+(const WaitcntRequirement &other) const {
    WaitcntRequirement result;
    if (load_cnt || other.load_cnt)
      result.load_cnt = load_cnt.value_or(0) + other.load_cnt.value_or(0);
    if (ds_cnt || other.ds_cnt)
      result.ds_cnt = ds_cnt.value_or(0) + other.ds_cnt.value_or(0);
    return result;
  }

  bool operator>(const WaitcntRequirement &other) const {
    return load_cnt.value_or(0) > other.load_cnt.value_or(0) ||
           ds_cnt.value_or(0) > other.ds_cnt.value_or(0);
  }

  void print(raw_ostream &os) const {
    os << "WaitcntRequirement: load_cnt=" << load_cnt << " ds_cnt=" << ds_cnt;
  }
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const WaitcntRequirement &result) {
  result.print(os);
  return os;
}

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
    os << "WaitcntState: " << size() << " pending ops [";
    if (pendingOps) {
      llvm::interleaveComma(pendingOps->ops, os,
                            [&](Operation *op) { os << *op; });
    }
    os << "], requirement: " << requirement;
  }

  /// Add a pending operation (copy-on-write)
  void addPendingOp(Operation *op) {
    cow();
    pendingOps->ops.push_back(op);
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
  ChangeResult reset() {
    if (!pendingOps && !requirement.hasRequirement())
      return ChangeResult::NoChange;

    pendingOps.reset();
    requirement = {};
    return ChangeResult::Change;
  }

  /// Set the required waitcnt values
  void setRequirement(const WaitcntRequirement &req) {
    requirement = req;
    SmallVector<Operation *> newPending;
    WaitcntRequirement runningRequirement;
    for (Operation *op : llvm::reverse(pendingOps->ops)) {
      WaitcntRequirement opReq =
          WaitcntRequirement::getOperationRequirement(op, false);
      runningRequirement = runningRequirement + opReq;
      if (runningRequirement > requirement)
        continue;

      newPending.push_back(op);
    }
    if (newPending.size() == pendingOps->ops.size())
      return;

    std::reverse(newPending.begin(), newPending.end());
    pendingOps = std::make_shared<PendingOperations>(std::move(newPending));
  }

  void resetRequirement() { requirement = {}; }

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

    // Search from the back to find the most recent dependency
    auto req = WaitcntRequirement::getOperationRequirement(defOp, true);
    bool found = false;
    for (Operation *op : llvm::reverse(pendingOps->ops)) {
      if (op == defOp) {
        found = true;
        break;
      }
      auto opReq = WaitcntRequirement::getOperationRequirement(op, false);
      if (!req.isSameCounterType(opReq))
        continue;

      req = req + opReq;
    }

    if (!found)
      return std::nullopt;

    return req;
  }

  /// Check for memory dependencies (RAW, WAR, WAW)
  std::optional<WaitcntRequirement>
  checkMemoryDependency(Operation *op, AliasAnalysis &aliasAnalysis) const {
    if (!pendingOps || pendingOps->ops.empty())
      return std::nullopt;

    // Check if this is a load or store operation
    std::optional<Value> currentBase = isLoadOrStoreOp(op);
    if (!currentBase)
      return std::nullopt;

    bool isCurrentLoad = isLoadOp(op).has_value();
    bool isCurrentStore = isStoreOp(op).has_value();

    // Search from the back to find the most recent dependency
    for (Operation *pendingOp : llvm::reverse(pendingOps->ops)) {
      std::optional<Value> pendingBase = isLoadOrStoreOp(pendingOp);
      if (!pendingBase)
        continue;

      if (aliasAnalysis.alias(*currentBase, *pendingBase).isNo())
        continue;

      bool isPendingLoad = isLoadOp(pendingOp).has_value();
      bool isPendingStore = isStoreOp(pendingOp).has_value();

      // Check for dependencies:
      // RAW: current load after pending store
      // WAR: current store after pending load
      // WAW: current store after pending store
      bool hasRAW = isCurrentLoad && isPendingStore;
      bool hasWAR = isCurrentStore && isPendingLoad;
      bool hasWAW = isCurrentStore && isPendingStore;

      if (hasRAW || hasWAR || hasWAW) {
        // Found dependency - compute requirement by counting forward from here
        auto it = llvm::find(pendingOps->ops, pendingOp);
        auto req = WaitcntRequirement::getOperationRequirement(pendingOp, true);
        for (Operation *countOp :
             llvm::make_range(std::next(it), pendingOps->ops.end())) {
          auto opReq =
              WaitcntRequirement::getOperationRequirement(countOp, false);
          if (!req.isSameCounterType(opReq))
            continue;
          req = req + opReq;
        }
        return req;
      }
    }

    return std::nullopt;
  }

private:
  /// Pending asynchronous operations (vector loads/stores)
  std::shared_ptr<PendingOperations> pendingOps;

  /// Required waitcnt before this state (for inserting actual waitcnt ops)
  WaitcntRequirement requirement;

  void cow() {
    if (!pendingOps || pendingOps.use_count() > 1) {
      auto newPending = std::make_shared<PendingOperations>();
      if (pendingOps)
        newPending->ops = pendingOps->ops;

      pendingOps = newPending;
    }
  }
};

/// Dense forward dataflow analysis for waitcnt insertion
class WaitcntAnalysis : public DenseForwardDataFlowAnalysis<WaitcntState> {
public:
  WaitcntAnalysis(DataFlowSolver &solver, AliasAnalysis &aliasAnalysis)
      : DenseForwardDataFlowAnalysis(solver), aliasAnalysis(aliasAnalysis) {}

  void setToEntryState(WaitcntState *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }

  LogicalResult visitOperation(Operation *op, const WaitcntState &before,
                               WaitcntState *after) override {
    LDBG() << "Visiting: " << *op;

    // Start with the state before this operation
    WaitcntState newState = before;

    // Check if any operands depend on pending operations (value dependency)
    WaitcntRequirement opRequirement;
    for (Value operand : op->getOperands()) {
      if (auto req = before.checkRequirement(operand)) {
        // Merge this requirement (take minimum for conservative wait)
        opRequirement.merge(*req);
      }
    }

    // Check for memory dependencies (RAW, WAR, WAW)
    if (auto memReq = before.checkMemoryDependency(op, aliasAnalysis))
      opRequirement.merge(*memReq);

    // Set the requirement for this operation
    if (opRequirement.hasRequirement()) {
      newState.setRequirement(opRequirement);
      LDBG() << "Operation requirement: " << opRequirement;
    } else {
      newState.resetRequirement();
      LDBG() << "No operation requirement";
    }

    // Check if this is an async memory operation (vector load/store)
    if (WaitcntRequirement::getOperationRequirement(op, false)
            .hasRequirement()) {
      // Add this operation to the pending list
      newState.addPendingOp(op);
    }

    LDBG() << "New state: " << newState;
    propagateIfChanged(after, after->join(newState));
    return success();
  }

private:
  AliasAnalysis &aliasAnalysis;
};

/// Pass that inserts wait/synchronization instructions for asynchronous
/// memory operations. This is analogous to LLVM's SIInsertWaitcnts pass.
class WaterInsertWaitcntPass
    : public water::impl::WaterInsertWaitcntBase<WaterInsertWaitcntPass> {
public:
  void runOnOperation() override {
    LDBG() << "Running WaterInsertWaitcntPass";
    Operation *op = getOperation();

    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<WaitcntAnalysis>(aliasAnalysis);

    if (failed(solver.initializeAndRun(op))) {
      signalPassFailure();
      return;
    }

    // Insert waitcnt operations based on analysis results
    IRRewriter rewriter(&getContext());
    op->walk([&](Operation *operation) {
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
