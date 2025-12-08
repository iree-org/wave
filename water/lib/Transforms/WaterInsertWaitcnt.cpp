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

  size_t size() const { return ops.size(); }
  bool empty() const { return ops.empty(); }

  bool hasSameTail(const PendingOperations &other) const {
    for (const auto &[op1, op2] :
         llvm::zip(llvm::reverse(ops), llvm::reverse(other.ops))) {
      if (op1 != op2)
        return false;
    }
    return true;
  }

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
  operator bool() const { return hasRequirement(); }

  void print(raw_ostream &os) const {
    os << "WaitcntRequirement: load_cnt=" << load_cnt << " ds_cnt=" << ds_cnt;
  }
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const WaitcntRequirement &result) {
  result.print(os);
  return os;
}

static bool mayAlias(Value lhs, Value rhs, AliasAnalysis &aliasAnalysis) {
  if (isWorkgroupAddressSpace(cast<MemRefType>(lhs.getType())) !=
      isWorkgroupAddressSpace(cast<MemRefType>(rhs.getType())))
    return false;

  return !aliasAnalysis.alias(lhs, rhs).isNo();
}

/// Lattice state tracking pending asynchronous operations
class WaitcntState : public AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WaitcntState)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &rhsState = static_cast<const WaitcntState &>(rhs);
    bool changed = false;

    SmallVector<std::shared_ptr<PendingOperations>, 4> toAppend;
    // Check if any pending operations has the same subset of operations as the
    // rhs and take the longer one.
    for (auto &rhsPendingOps : rhsState.pendingOpsLists) {
      bool found = false;
      for (auto &pendingOps : pendingOpsLists) {
        if (pendingOps->hasSameTail(*rhsPendingOps)) {
          if (rhsPendingOps->size() > pendingOps->size()) {
            pendingOps = rhsPendingOps;
            changed = true;
          }
          found = true;
          break;
        }
      }
      if (!found)
        toAppend.push_back(rhsPendingOps);
    }

    // If there are any pending operations that don't have the same subset of
    // operations as the rhs, append them to the pending operations lists.
    if (!toAppend.empty()) {
      pendingOpsLists.append(toAppend);
      changed = true;
    }

    // Merge requirements (take minimum for conservative join)
    if (requirement.merge(rhsState.requirement))
      changed = true;

    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    os << "WaitcntState: pending ops [";
    for (auto &pendingOps : pendingOpsLists) {
      os << "[";
      llvm::interleaveComma(pendingOps->ops, os,
                            [&](Operation *op) { os << *op; });
      os << "]";
    }
    os << "], requirement: " << requirement;
  }

  void addPendingOp(Operation *op) {
    if (pendingOpsLists.empty()) {
      pendingOpsLists.push_back(std::make_shared<PendingOperations>());
    } else {
      cow();
    }
    for (auto &pendingOps : pendingOpsLists)
      pendingOps->ops.push_back(op);
  }

  /// Initialize to empty state
  ChangeResult reset() {
    if (pendingOpsLists.empty() && !requirement.hasRequirement())
      return ChangeResult::NoChange;

    pendingOpsLists.clear();
    requirement = {};
    return ChangeResult::Change;
  }

  /// Set the required waitcnt values
  void setRequirement(const WaitcntRequirement &req) {
    requirement = req;
    for (auto &pendingOps : pendingOpsLists) {
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
        continue;

      std::reverse(newPending.begin(), newPending.end());
      pendingOps = std::make_shared<PendingOperations>(std::move(newPending));
    }

    // Remove empty lists
    pendingOpsLists.erase(std::remove_if(pendingOpsLists.begin(),
                                         pendingOpsLists.end(),
                                         [](const auto &pendingOps) {
                                           return pendingOps->empty();
                                         }),
                          pendingOpsLists.end());

    // Merge lists with the same tail (keep the longer one)
    for (size_t i = 0; i < pendingOpsLists.size(); ++i) {
      for (size_t j = i + 1; j < pendingOpsLists.size();) {
        if (pendingOpsLists[i]->hasSameTail(*pendingOpsLists[j])) {
          if (pendingOpsLists[j]->size() > pendingOpsLists[i]->size()) {
            pendingOpsLists[i] = pendingOpsLists[j];
          }
          pendingOpsLists.erase(pendingOpsLists.begin() + j);
        } else {
          ++j;
        }
      }
    }
  }

  void resetRequirement() { requirement = {}; }

  /// Get the required waitcnt values
  const WaitcntRequirement &getRequirement() const { return requirement; }

  /// Check if there's a waitcnt requirement
  bool hasRequirement() const { return requirement.hasRequirement(); }

  /// Check if a value depends on pending operations and compute required wait
  WaitcntRequirement checkRequirement(Value val) const {
    // Check if val is produced by any pending operation
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return {};

    WaitcntRequirement result;
    for (auto &pendingOps : pendingOpsLists) {
      if (pendingOps->empty())
        continue;

      // Search from the back to find the most recent dependency
      bool found = false;
      auto req = WaitcntRequirement::getOperationRequirement(defOp, true);
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

      if (found)
        result.merge(req);
    }

    return result;
  }

  /// Check for memory dependencies (RAW, WAR, WAW)
  WaitcntRequirement checkMemoryDependency(Operation *op,
                                           AliasAnalysis &aliasAnalysis) const {
    // Check if this is a load or store operation
    std::optional<Value> currentBase = isLoadOrStoreOp(op);
    if (!currentBase)
      return {};

    bool isCurrentLoad = isLoadOp(op).has_value();
    bool isCurrentStore = isStoreOp(op).has_value();

    WaitcntRequirement result;
    for (auto &pendingOps : pendingOpsLists) {
      if (pendingOps->empty())
        continue;

      // Search from the back to find the most recent dependency
      for (Operation *pendingOp : llvm::reverse(pendingOps->ops)) {
        std::optional<Value> pendingBase = isLoadOrStoreOp(pendingOp);
        if (!pendingBase)
          continue;

        if (!mayAlias(*currentBase, *pendingBase, aliasAnalysis))
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
          // Found dependency - compute requirement by counting forward from
          // here
          auto it = llvm::find(pendingOps->ops, pendingOp);
          auto req =
              WaitcntRequirement::getOperationRequirement(pendingOp, true);
          for (Operation *countOp :
               llvm::make_range(std::next(it), pendingOps->ops.end())) {
            auto opReq =
                WaitcntRequirement::getOperationRequirement(countOp, false);
            if (!req.isSameCounterType(opReq))
              continue;
            req = req + opReq;
          }
          result.merge(req);
        }
      }
    }

    return result;
  }

private:
  /// Pending asynchronous operations
  SmallVector<std::shared_ptr<PendingOperations>, 4> pendingOpsLists;

  /// Required waitcnt after this state
  WaitcntRequirement requirement;

  void cow() {
    for (auto &pendingOps : pendingOpsLists) {
      if (pendingOps.use_count() > 1) {
        auto newPending = std::make_shared<PendingOperations>();
        if (pendingOps)
          newPending->ops = pendingOps->ops;
        pendingOps = newPending;
      }
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
    LDBG() << "  Before: " << before;

    // Start with the state before this operation
    WaitcntState newState = before;

    // Check if any operands depend on pending operations (value dependency)
    WaitcntRequirement opRequirement;
    for (Value operand : op->getOperands()) {
      if (auto req = before.checkRequirement(operand)) {
        // Merge this requirement (take minimum for conservative wait)
        opRequirement.merge(req);
      }
    }

    // Check for memory dependencies (RAW, WAR, WAW)
    if (auto memReq = before.checkMemoryDependency(op, aliasAnalysis)) {
      LDBG() << "  Memory dependency: " << memReq;
      opRequirement.merge(memReq);
    }

    // Set the requirement for this operation
    if (opRequirement.hasRequirement()) {
      newState.setRequirement(opRequirement);
      LDBG() << "  Operation requirement: " << opRequirement;
    } else {
      newState.resetRequirement();
      LDBG() << "  No operation requirement";
    }

    // Check if this is an async memory operation (vector load/store)
    if (WaitcntRequirement::getOperationRequirement(op, false)
            .hasRequirement()) {
      // Add this operation to the pending list
      newState.addPendingOp(op);
    }

    LDBG() << "  New state: " << newState;
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
