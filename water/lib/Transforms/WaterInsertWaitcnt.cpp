// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
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
static bool isBarrier(Operation *op) {
  return isa<gpu::BarrierOp>(op) || isa<amdgpu::LDSBarrierOp>(op);
}

/// Try to propagate view operations to the base memref.
static Value propagateViewOps(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();

  return value;
}

/// Collect all underlying values through view and select operations.
static SmallVector<Value> collectUnderlyingValues(Value value) {
  SmallVector<Value> result;
  SmallVector<Value> worklist;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value current = propagateViewOps(worklist.pop_back_val());
    if (auto select = current.getDefiningOp<arith::SelectOp>()) {
      worklist.push_back(select.getTrueValue());
      worklist.push_back(select.getFalseValue());
    } else {
      result.push_back(current);
    }
  }
  return result;
}

/// Check if we need to track the operation for waitcnt requirements.
static bool trackOp(Operation *op) {
  return isa<amdgpu::TensorLoadToLDSOp>(op);
}

static std::optional<Value> propagateTensorDesc(Value value, bool isLoad) {
  auto makeDesc = value.getDefiningOp<amdgpu::MakeDmaDescriptorOp>();
  if (!makeDesc)
    return value;

  value = makeDesc.getBase();
  auto makeBase = value.getDefiningOp<amdgpu::MakeDmaBaseOp>();
  if (!makeBase)
    return value;

  return propagateViewOps(isLoad ? makeBase.getGlobal() : makeBase.getLds());
}

/// Check if the operation is a load operation and return list of base memrefs
/// to track.
static SmallVector<Value> isLoadOp(Operation *op) {
  if (auto load = dyn_cast<vector::LoadOp>(op)) {
    return collectUnderlyingValues(load.getBase());
  } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
    return collectUnderlyingValues(load.getMemref());
  } else if (auto load = dyn_cast<amdgpu::TensorLoadToLDSOp>(op)) {
    if (auto memref = propagateTensorDesc(load.getDesc(), true))
      return collectUnderlyingValues(*memref);
  }
  return {};
}

/// Check if the operation is a store operation and return list of base memrefs
/// to track.
static SmallVector<Value> isStoreOp(Operation *op) {
  SmallVector<Value> result;
  if (auto store = dyn_cast<amdgpu::TensorLoadToLDSOp>(op)) {
    if (auto memref = propagateTensorDesc(store.getDesc(), false))
      result.push_back(*memref);
  }
  return result;
}

template <typename T>
static raw_ostream &print_range(raw_ostream &os, T &&range) {
  llvm::interleaveComma(range, os, [&](const auto &item) { os << item; });
  return os;
}

/// Shared pending operations list for structural sharing
struct PendingOperations {
  using TokenContainer = SmallVector<Value, 2>;

  PendingOperations() = default;
  PendingOperations(SmallVector<Operation *> &&ops,
                    SmallVector<TokenContainer> &&opsTokens)
      : ops(std::move(ops)), opsTokens(std::move(opsTokens)) {}

  TokenContainer &addOp(Operation *op) {
    // Failsafe to prevent infinite list growth.
    if (size() >= 256)
      llvm::report_fatal_error("Pending operations list is too long");

    if (!ops.empty() && isBarrier(op) && isBarrier(ops.back()))
      return opsTokens.back();

    ops.push_back(op);
    auto &back = opsTokens.emplace_back();
    for (Value memref : isStoreOp(op))
      back.push_back(memref);

    for (Value memref : isLoadOp(op))
      back.push_back(memref);

    return back;
  }

  size_t size() const { return ops.size(); }
  bool empty() const { return ops.empty(); }

  auto opsAndTokens() const {
    assert(ops.size() == opsTokens.size() &&
           "ops and opsTokens must have the same size");
    return llvm::zip(ops, opsTokens);
  }

  auto opsAndTokensReverse() const {
    assert(ops.size() == opsTokens.size() &&
           "ops and opsTokens must have the same size");
    return llvm::zip(llvm::reverse(ops), llvm::reverse(opsTokens));
  }

  bool hasSameTail(const PendingOperations &other) const {
    for (const auto &[op1, op2, tok1, tok2] :
         llvm::zip(llvm::reverse(ops), llvm::reverse(other.ops),
                   llvm::reverse(opsTokens), llvm::reverse(other.opsTokens))) {
      if (op1 != op2)
        return false;
      if (tok1 != tok2)
        return false;
    }
    return true;
  }

  void updateTokens(
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &)> updateFunc) {
    for (TokenContainer &tokens : opsTokens) {
      TokenContainer newTok;
      for (Value tok : tokens)
        updateFunc(tok, newTok);

      tokens = std::move(newTok);
    }
  }

  void print(raw_ostream &os) const {
    os << "PendingOperations: ops=[";
    llvm::interleaveComma(opsAndTokens(), os, [&](const auto &opAndTok) {
      os << *std::get<0>(opAndTok) << "|";
      print_range(os, std::get<1>(opAndTok));
    });
    os << "]";
  }

  bool operator==(const PendingOperations &other) const {
    return ops == other.ops && opsTokens == other.opsTokens;
  }

  bool operator!=(const PendingOperations &other) const {
    return !(*this == other);
  }

  SmallVector<Operation *> ops;
  SmallVector<TokenContainer> opsTokens;
};

/// Waitcnt requirement for synchronization
struct WaitcntRequirement {
  std::optional<unsigned> tensor_cnt;

  WaitcntRequirement() = default;

  WaitcntRequirement(amdgpu::MemoryCounterWaitOp waitOp) {
    if (auto tensorCnt = waitOp.getTensorAttr())
      tensor_cnt = tensorCnt.getInt();
  }

  bool hasRequirement() const { return tensor_cnt.has_value(); }

  /// Merge with another requirement (take minimum for conservative join)
  /// Returns true if this requirement changed
  bool merge(const WaitcntRequirement &other) {
    bool changed = false;

    // Take minimum of each counter (lower value = more restrictive)
    if (other.tensor_cnt.has_value()) {
      if (!tensor_cnt.has_value() || *other.tensor_cnt < *tensor_cnt) {
        tensor_cnt = other.tensor_cnt;
        changed = true;
      }
    }

    return changed;
  }

  std::optional<unsigned> getTensorCnt() const { return tensor_cnt; }

  bool isSameCounterType(const WaitcntRequirement &other) const {
    return tensor_cnt.has_value() == other.tensor_cnt.has_value();
  }

  static WaitcntRequirement getOperationRequirement(Operation *op, bool zero) {
    WaitcntRequirement req;
    if (isa<amdgpu::TensorLoadToLDSOp, memref::LoadOp, vector::LoadOp>(op))
      req.tensor_cnt = zero ? 0 : 1;

    return req;
  }

  WaitcntRequirement operator+(const WaitcntRequirement &other) const {
    WaitcntRequirement result;
    if (tensor_cnt || other.tensor_cnt)
      result.tensor_cnt = tensor_cnt.value_or(0) + other.tensor_cnt.value_or(0);
    return result;
  }

  bool operator>(const WaitcntRequirement &other) const {
    if (tensor_cnt && other.tensor_cnt && *tensor_cnt > *other.tensor_cnt)
      return true;
    return false;
  }
  operator bool() const { return hasRequirement(); }

  void print(raw_ostream &os) const {
    os << "WaitcntRequirement: tensor_cnt=" << tensor_cnt;
  }
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const WaitcntRequirement &result) {
  result.print(os);
  return os;
}

static bool mayAlias(Value lhs, Value rhs, ArrayRef<Value> tokens) {
  auto memref1 = cast<MemRefType>(lhs.getType());
  auto memref2 = cast<MemRefType>(rhs.getType());
  if (memref1.getMemorySpace() != memref2.getMemorySpace())
    return false;

  return llvm::is_contained(tokens, lhs);
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

    if (changed)
      resetPendingOpsSet();

    // Merge requirements (take minimum for conservative join)
    if (requirement.merge(rhsState.requirement))
      changed = true;

    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ChangeResult merge(const WaitcntState &rhs) {
    bool changed = false;

    if (pendingOpsLists.size() != rhs.pendingOpsLists.size()) {
      changed = true;
    } else {
      for (auto [listSrc, listDst] :
           llvm::zip(pendingOpsLists, rhs.pendingOpsLists)) {
        if (*listSrc != *listDst) {
          changed = true;
          break;
        }
      }
    }

    if (changed) {
      pendingOpsLists = rhs.pendingOpsLists;
      resetPendingOpsSet();
    }

    if (requirement.merge(rhs.requirement))
      changed = true;
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    os << "WaitcntState: pending ops [";
    for (auto &pendingOps : pendingOpsLists) {
      os << "\n   [";
      pendingOps->print(os);
      os << "]";
    }
    os << "\n   ], requirement: " << requirement;
  }

  void addPendingOp(Operation *op) {
    if (pendingOpsLists.empty()) {
      pendingOpsLists.push_back(std::make_shared<PendingOperations>());
    } else {
      cow();
    }
    for (auto &pendingOps : pendingOpsLists) {
      auto &tokens = pendingOps->addOp(op);
      for (Value token : tokens)
        pendingOpsTokens.insert(token);
    }

    pendingOpsSet.insert(op);
  }

  /// Initialize to empty state
  ChangeResult reset() {
    if (pendingOpsLists.empty() && !requirement.hasRequirement())
      return ChangeResult::NoChange;

    pendingOpsLists.clear();
    requirement = {};
    resetPendingOpsSet();
    return ChangeResult::Change;
  }

  /// Set the required waitcnt values
  void setRequirement(const WaitcntRequirement &req) {
    requirement = req;
    for (auto &pendingOps : pendingOpsLists) {
      SmallVector<Operation *> newPending;
      SmallVector<PendingOperations::TokenContainer> newPendingTokens;
      WaitcntRequirement runningRequirement;
      for (const auto &[op, tok] : llvm::reverse(pendingOps->opsAndTokens())) {
        WaitcntRequirement opReq =
            WaitcntRequirement::getOperationRequirement(op, false);
        runningRequirement = runningRequirement + opReq;
        if (runningRequirement > requirement)
          continue;

        newPending.push_back(op);
        newPendingTokens.push_back(tok);
      }
      if (newPending.size() == pendingOps->size())
        continue;

      std::reverse(newPending.begin(), newPending.end());
      std::reverse(newPendingTokens.begin(), newPendingTokens.end());
      pendingOps = std::make_shared<PendingOperations>(
          std::move(newPending), std::move(newPendingTokens));
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

    resetPendingOpsSet();
  }

  void updateTokens(
      llvm::function_ref<void(Value, SmallVectorImpl<Value> &)> updateFunc) {
    for (auto &pendingOps : pendingOpsLists)
      pendingOps->updateTokens(updateFunc);
  }

  void resetRequirement() { requirement = {}; }

  /// Get the required waitcnt values
  const WaitcntRequirement &getRequirement() const { return requirement; }

  /// Check if there's a waitcnt requirement
  bool hasRequirement() const { return requirement.hasRequirement(); }

  /// Check for memory dependencies (RAW, WAR, WAW)  and compute required wait
  WaitcntRequirement
  checkMemoryDependency(Operation *op,
                        llvm::SmallSetVector<Operation *, 4> &barriers) const {
    auto checkMemref = [&](Value memref, bool isCurrentLoad,
                           bool isCurrentStore) -> WaitcntRequirement {
      WaitcntRequirement result;
      if (!isPendingOp(memref))
        return result;

      for (auto &pendingOps : pendingOpsLists) {
        if (pendingOps->empty())
          continue;

        Operation *barrier = nullptr;

        // Search from the back to find the most recent dependency
        for (const auto &[pendingOpVar, pendingTokensVar] :
             pendingOps->opsAndTokensReverse()) {

          if (!barrier && isBarrier(pendingOpVar))
            barrier = pendingOpVar;

          // We canot capture structured bindings into lambda, thanks C++.
          auto &pendingTokens = pendingTokensVar;
          auto &pendingOp = pendingOpVar;
          auto checkPendingMemref =
              [&](Value pendingMemref, bool isPendingLoad,
                  bool isPendingStore) -> WaitcntRequirement {
            WaitcntRequirement pendingResult;
            if (!mayAlias(memref, pendingMemref, pendingTokens))
              return pendingResult;

            // Check for dependencies:
            // RAW: current load after pending store
            // WAR: current store after pending load
            // WAW: current store after pending store
            // We don't care about WAW dependencies for now.
            bool hasRAW = isCurrentLoad && isPendingStore;
            bool hasWAR = isCurrentStore && isPendingLoad;
            bool hasWAW = false; // isCurrentStore && isPendingStore;

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
              pendingResult.merge(req);
            }
            if (pendingResult.hasRequirement() && barrier)
              barriers.insert(barrier);

            return pendingResult;
          };
          for (Value loadBase : isLoadOp(pendingOp))
            result.merge(checkPendingMemref(loadBase, true, false));
          for (Value storeBase : isStoreOp(pendingOp))
            result.merge(checkPendingMemref(storeBase, false, true));
        }
      }

      return result;
    };
    // TODO: atomics will have both load and store flags set
    WaitcntRequirement result;
    for (Value loadBase : isLoadOp(op))
      result.merge(checkMemref(loadBase, true, false));
    for (Value storeBase : isStoreOp(op))
      result.merge(checkMemref(storeBase, false, true));
    return result;
  }

private:
  /// Pending asynchronous operations.
  SmallVector<std::shared_ptr<PendingOperations>, 4> pendingOpsLists;

  /// Required waitcnt after this state.
  WaitcntRequirement requirement;

  /// Cached sets of pending operations and tokens for quick lookup.
  mutable llvm::SmallDenseSet<Operation *> pendingOpsSet;
  mutable llvm::SmallDenseSet<Value> pendingOpsTokens;

  /// Copy on write for pending operations lists.
  void cow() {
    for (auto &pendingOps : pendingOpsLists) {
      if (pendingOps.use_count() > 1) {
        auto newPending = std::make_shared<PendingOperations>();
        if (pendingOps)
          *newPending = *pendingOps;
        pendingOps = std::move(newPending);
      }
    }
  }

  /// Check if the operation or value is in pending operations lists.
  bool isPendingOp(llvm::PointerUnion<Operation *, Value> opOrVal) const {
    if (pendingOpsLists.empty())
      return false;

    // Build the set of pending operations lazily
    bool found = false;
    if (pendingOpsSet.empty()) {
      assert(pendingOpsTokens.empty() && "pendingOpsTokens must be empty");
      Operation *op = dyn_cast<Operation *>(opOrVal);
      Value val = dyn_cast<Value>(opOrVal);
      for (const auto &pendingOps : pendingOpsLists) {
        for (const auto &[pendingOp, pendingTokens] :
             pendingOps->opsAndTokens()) {
          if (pendingOp == op)
            found = true;

          pendingOpsSet.insert(pendingOp);
          for (Value token : pendingTokens) {
            if (token == val)
              found = true;

            pendingOpsTokens.insert(token);
          }
        }
      }
    }

    if (found)
      return true;

    return isa<Operation *>(opOrVal)
               ? pendingOpsSet.contains(cast<Operation *>(opOrVal))
               : pendingOpsTokens.contains(cast<Value>(opOrVal));
  }

  void resetPendingOpsSet() {
    pendingOpsSet.clear();
    pendingOpsTokens.clear();
  }
};

static RegionSuccessor getRegionResults(ArrayRef<RegionSuccessor> successors,
                                        Region *region) {
  for (const auto &successor : successors) {
    if (successor.getSuccessor() == region)
      return successor;
  }
  llvm_unreachable("Region not found, malformed SCF op?");
}

/// Dense forward dataflow analysis for waitcnt insertion
class WaitcntAnalysis : public DenseForwardDataFlowAnalysis<WaitcntState> {
public:
  explicit WaitcntAnalysis(DataFlowSolver &solver)
      : DenseForwardDataFlowAnalysis(solver) {}

  void setToEntryState(WaitcntState *lattice) override {
    propagateIfChanged(lattice, lattice->reset());
  }

  LogicalResult visitOperation(Operation *op, const WaitcntState &before,
                               WaitcntState *after) override {
    LDBG() << "Visiting: " << *op;
    LDBG() << "  Before: " << before;

    // Start with the state before this operation
    WaitcntState newState = before;

    if (isBarrier(op)) {
      LDBG() << "  Barrier: " << *op;
      newState.addPendingOp(op);
      LDBG() << "  New state: " << newState;
      propagateIfChanged(after, after->join(newState));
      return success();
    }

    llvm::SmallSetVector<Operation *, 4> barriers;

    WaitcntRequirement opRequirement = after->getRequirement();

    // Check for memory dependencies (RAW, WAR, WAW)
    if (auto memReq = before.checkMemoryDependency(op, barriers)) {
      LDBG() << "  Memory dependency: " << memReq;
      opRequirement.merge(memReq);
    } else {
      LDBG() << "  No memory dependency";
    }

    if (opRequirement.hasRequirement() && !barriers.empty()) {
      // newState.setRequirement(opRequirement);
      LDBG() << "  Barriers found, requirement: " << opRequirement;
      for (Operation *barrier : barriers) {
        LDBG() << "    " << *barrier;
        WaitcntState *beforeState =
            getOrCreate<WaitcntState>(getProgramPointBefore(barrier));
        WaitcntState *afterState =
            getOrCreate<WaitcntState>(getProgramPointAfter(barrier));
        WaitcntState newBarrierState = *beforeState;
        newBarrierState.setRequirement(opRequirement);
        propagateIfChanged(afterState, afterState->merge(newBarrierState));
      }
      return success();
    }

    // Check if this is an existing memory_counter_wait operation
    if (auto waitOp = dyn_cast<amdgpu::MemoryCounterWaitOp>(op)) {
      LDBG() << "  Existing waitcnt operation: " << *waitOp;
      opRequirement.merge(WaitcntRequirement(waitOp));
    }

    // Set the requirement for this operation
    if (opRequirement.hasRequirement()) {
      newState.setRequirement(opRequirement);
      LDBG() << "  Operation requirement: " << opRequirement;
    } else {
      newState.resetRequirement();
      LDBG() << "  No operation requirement";
    }

    // Check if this is an async memory operation
    if (trackOp(op)) {
      // Add this operation to the pending list
      newState.addPendingOp(op);
    }

    auto changed = after->merge(newState);
    if (changed == ChangeResult::Change) {
      LDBG() << "  New state: " << newState;
    } else {
      LDBG() << "  No change";
    }
    propagateIfChanged(after, changed);
    return success();
  }

  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const WaitcntState &before,
                                            WaitcntState *after) override {
    LDBG() << "Visiting region branch control flow transfer: " << *branch;
    LDBG() << "  Region from: " << regionFrom;
    LDBG() << "  Region to: " << regionTo;
    LDBG() << "  Before: " << before;
    LDBG() << "  After: " << *after;

    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(RegionBranchPoint::parent(), successors);

    auto destSuccessor = [&]() -> RegionSuccessor {
      if (regionTo) {
        Region &region = branch->getRegions()[*regionTo];
        return getRegionResults(successors, &region);
      } else {
        return getRegionResults(successors, nullptr);
      }
    }();
    // Dest values are either nested block args or branch op results.
    ValueRange destValues = destSuccessor.getSuccessorInputs();

    // Map from input values to dest values.
    llvm::SmallDenseMap<Value, Value> valuesMapping;
    if (regionFrom) {
      Region &region = branch->getRegions()[*regionFrom];
      for (Block &block : region) {
        auto term =
            dyn_cast<RegionBranchTerminatorOpInterface>(block.getTerminator());
        if (!term)
          continue;

        ValueRange source =
            term.getMutableSuccessorOperands(destSuccessor).getAsOperandRange();
        for (auto [source, dest] : llvm::zip(source, destValues))
          valuesMapping[source] = dest;
      }
    } else {
      ValueRange source = branch.getEntrySuccessorOperands(destSuccessor);
      for (auto [source, dest] : llvm::zip(source, destValues))
        valuesMapping[source] = dest;
    }

    DominanceInfo dom;

    WaitcntState newState = before;
    auto tokenUpdateFunc = [&](Value value, SmallVectorImpl<Value> &newTokens) {
      // Keep the token if it dominates current op as user can use it directly.
      if (dom.properlyDominates(value, branch))
        newTokens.push_back(value);

      // Add token propagated through region control flow.
      if (Value mappedValue = valuesMapping.lookup(value))
        if (newTokens.empty() || newTokens.back() != mappedValue)
          newTokens.push_back(mappedValue);
    };
    newState.updateTokens(tokenUpdateFunc);

    LDBG() << "  New state: " << newState;

    propagateIfChanged(after, after->join(newState));
  }
};

/// Pass that inserts wait/synchronization instructions for asynchronous
/// memory operations. This is analogous to LLVM's SIInsertWaitcnts pass.
class WaterInsertWaitcntPass
    : public water::impl::WaterInsertWaitcntBase<WaterInsertWaitcntPass> {
public:
  void runOnOperation() override {
    LDBG() << "Running WaterInsertWaitcntPass";
    Operation *op = getOperation();

    DataFlowSolver solver;
    loadBaselineAnalyses(solver);
    solver.load<WaitcntAnalysis>();

    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

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

      // Insert wait operation before the current operation.
      // If the current operation is already a memory_counter_wait operation
      // they will be merged later.
      rewriter.setInsertionPoint(operation);
      amdgpu::MemoryCounterWaitOp::create(rewriter, operation->getLoc(),
                                          nullptr, nullptr, nullptr, nullptr,
                                          getAttr(req.getTensorCnt()));
    });
  }
};

} // namespace
