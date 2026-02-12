// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMMAREORDERPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {

// clang-format off
/// List of all ROCDL WMMA operations that support reuseA/reuseB flags.
#define ROCDL_WMMA_OPS_WITH_REUSE                                              \
    ROCDL::wmma_f32_16x16x4_f32,                                               \
    ROCDL::wmma_f32_16x16x32_bf16,                                             \
    ROCDL::wmma_f32_16x16x32_f16,                                              \
    ROCDL::wmma_f16_16x16x32_f16,                                              \
    ROCDL::wmma_bf16_16x16x32_bf16,                                            \
    ROCDL::wmma_bf16f32_16x16x32_bf16,                                         \
    ROCDL::wmma_f32_16x16x64_fp8_fp8,                                          \
    ROCDL::wmma_f32_16x16x64_fp8_bf8,                                          \
    ROCDL::wmma_f32_16x16x64_bf8_fp8,                                          \
    ROCDL::wmma_f32_16x16x64_bf8_bf8,                                          \
    ROCDL::wmma_f16_16x16x64_fp8_fp8,                                          \
    ROCDL::wmma_f16_16x16x64_fp8_bf8,                                          \
    ROCDL::wmma_f16_16x16x64_bf8_fp8,                                          \
    ROCDL::wmma_f16_16x16x64_bf8_bf8,                                          \
    ROCDL::wmma_f32_16x16x128_fp8_fp8,                                         \
    ROCDL::wmma_f32_16x16x128_fp8_bf8,                                         \
    ROCDL::wmma_f32_16x16x128_bf8_fp8,                                         \
    ROCDL::wmma_f32_16x16x128_bf8_bf8,                                         \
    ROCDL::wmma_f16_16x16x128_fp8_fp8,                                         \
    ROCDL::wmma_f16_16x16x128_fp8_bf8,                                         \
    ROCDL::wmma_f16_16x16x128_bf8_fp8,                                         \
    ROCDL::wmma_f16_16x16x128_bf8_bf8,                                         \
    ROCDL::wmma_i32_16x16x64_iu8,                                              \
    ROCDL::wmma_scale_f32_16x16x128_f8f6f4,                                    \
    ROCDL::wmma_scale16_f32_16x16x128_f8f6f4,                                  \
    ROCDL::wmma_scale_f32_32x16x128_f4,                                        \
    ROCDL::wmma_scale16_f32_32x16x128_f4
// clang-format on

/// Returns true if the operation is a ROCDL matrix multiply intrinsic
/// (WMMA, MFMA, scaled MFMA, or SMFMAC).
static bool isMatrixMultiplyOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("rocdl.wmma") || name.starts_with("rocdl.mfma") ||
         name.starts_with("rocdl.smfmac");
}

/// Helper to get matrix A operand from any matrix multiply op.
/// All ROCDL matrix multiply ops have A as operand 0.
static Value getMatrixA(Operation *op) { return op->getOperand(0); }

/// Helper to get matrix B operand from any matrix multiply op.
/// All ROCDL matrix multiply ops have B as operand 1.
static Value getMatrixB(Operation *op) { return op->getOperand(1); }

/// Helper to get matrix C (accumulator) operand from any matrix multiply op.
/// All ROCDL matrix multiply ops have C as operand 2.
static Value getMatrixC(Operation *op) { return op->getOperand(2); }

/// Sets the reuseA flag on a WMMA operation. No-op for non-WMMA ops.
static void setReuseA(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op).Case<ROCDL_WMMA_OPS_WITH_REUSE>(
      [reuse](auto wmmaOp) { wmmaOp.setReuseA(reuse); });
}

/// Sets the reuseB flag on a WMMA operation. No-op for non-WMMA ops.
static void setReuseB(Operation *op, bool reuse) {
  llvm::TypeSwitch<Operation *>(op).Case<ROCDL_WMMA_OPS_WITH_REUSE>(
      [reuse](auto wmmaOp) { wmmaOp.setReuseB(reuse); });
}

/// Returns true if op is pure (no side effects) and not a matrix multiply.
static bool isPureOp(Operation *op) {
  return !isMatrixMultiplyOp(op) && isPure(op);
}

/// Returns true if op can be moved just before insertionPoint without
/// breaking dominance. All operands must already dominate the insertion point.
static bool canMoveBefore(Operation *op, Operation *insertionPoint) {
  Block *block = insertionPoint->getBlock();
  for (Value operand : op->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def)
      continue; // Block argument — always dominates.
    if (def->getBlock() != block)
      continue; // Different block — assumed to dominate.
    if (!def->isBeforeInBlock(insertionPoint))
      return false;
  }
  return true;
}

/// Returns true if op can be moved just after insertionPoint without
/// breaking dominance. No user in the same block may precede the insertion
/// point.
static bool canMoveAfter(Operation *op, Operation *insertionPoint) {
  Block *block = insertionPoint->getBlock();
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != block)
        continue; // Different block — not a constraint.
      if (user->isBeforeInBlock(insertionPoint) || user == insertionPoint)
        return false;
    }
  }
  return true;
}

class MMAReorderPass
    : public water::impl::WaterMMAReorderPassBase<MMAReorderPass> {
public:
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    // Process each block independently.
    rootOp->walk([&](Block *block) { processBlock(block); });
  }

private:
  /// Process a single basic block. Collects matrix multiply ops together
  /// with interleaved pure ops into extended sequences, hoists/sinks the
  /// pure ops out, then reorders the MMA ops for maximum operand reuse.
  void processBlock(Block *block) {
    SmallVector<Operation *> currentSequence;

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (isMatrixMultiplyOp(&op)) {
        currentSequence.push_back(&op);
      } else if (!currentSequence.empty() && isPureOp(&op)) {
        // Pure op inside a sequence — include it.
        currentSequence.push_back(&op);
      } else if (!currentSequence.empty()) {
        // Non-pure, non-MMA op ends the sequence.
        processExtendedSequence(currentSequence);
        currentSequence.clear();
      }
    }

    if (!currentSequence.empty())
      processExtendedSequence(currentSequence);
  }

  /// Hoist and sink pure ops out of an MMA sequence, then reorder the
  /// remaining consecutive MMA runs.
  static void processExtendedSequence(MutableArrayRef<Operation *> ops) {
    // Trim trailing non-MMA ops — they are after the last MMA and do not
    // contribute to the reordering window.
    while (!ops.empty() && !isMatrixMultiplyOp(ops.back()))
      ops = ops.drop_back();

    // Separate MMA and pure ops.
    SmallVector<Operation *> mmaOps, pureOps;
    for (Operation *op : ops) {
      if (isMatrixMultiplyOp(op))
        mmaOps.push_back(op);
      else
        pureOps.push_back(op);
    }

    if (mmaOps.size() < 2)
      return;

    // Hoist pure ops whose operands all dominate the first MMA.
    // Processing in original order handles chains: an already-hoisted op
    // is now before firstMMA, so dependents see it as dominating.
    Operation *firstMMA = mmaOps.front();
    llvm::SmallDenseSet<Operation *> hoisted;
    for (Operation *pureOp : pureOps) {
      if (canMoveBefore(pureOp, firstMMA)) {
        pureOp->moveBefore(firstMMA);
        hoisted.insert(pureOp);
      }
    }

    // Sink remaining pure ops whose results have no users at-or-before
    // the last MMA. Processing in reverse order handles chains.
    Operation *lastMMA = mmaOps.back();
    for (Operation *pureOp : llvm::reverse(pureOps)) {
      if (hoisted.contains(pureOp))
        continue;
      if (canMoveAfter(pureOp, lastMMA)) {
        pureOp->moveAfter(lastMMA);
        // Update lastMMA to keep sunk ops in original relative order.
        lastMMA = pureOp;
      }
    }

    // Collect consecutive MMA runs between first and last MMA.
    // Any remaining "trapped" pure op splits the run.
    SmallVector<Operation *> consecutiveRun;
    for (auto it = mmaOps.front()->getIterator(),
              end = std::next(mmaOps.back()->getIterator());
         it != end; ++it) {
      if (isMatrixMultiplyOp(&*it)) {
        consecutiveRun.push_back(&*it);
      } else {
        processConsecutiveWMMAOps(consecutiveRun);
        consecutiveRun.clear();
      }
    }
    processConsecutiveWMMAOps(consecutiveRun);
  }

  /// Check if candidate can be scheduled given already scheduled ops.
  /// Returns true if all operands of candidate that are defined by ops
  /// in the sequence have already been scheduled.
  static bool
  canSchedule(Operation *candidate,
              const llvm::SmallSetVector<Operation *, 16> &scheduled,
              const llvm::SmallDenseSet<Operation *> &opsInSequence) {
    for (Value operand : candidate->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && opsInSequence.contains(defOp) && !scheduled.contains(defOp))
        return false;
    }
    return true;
  }

  /// Process a sequence of consecutive matrix multiply operations.
  static void processConsecutiveWMMAOps(MutableArrayRef<Operation *> wmmaOps) {
    if (wmmaOps.size() < 2)
      return;

    llvm::SmallDenseSet<Operation *> opsInSequence(wmmaOps.begin(),
                                                   wmmaOps.end());
    llvm::SmallSetVector<Operation *, 16> scheduled;

    // First op stays in place.
    scheduled.insert(wmmaOps[0]);

    // Greedily select ops that maximize reuse while respecting dependencies.
    for ([[maybe_unused]] auto i : llvm::seq<size_t>(1, wmmaOps.size())) {
      Operation *prev = scheduled.back();
      Value prevA = getMatrixA(prev);
      Value prevB = getMatrixB(prev);

      Operation *bestCandidate = nullptr;
      int bestScore = -1;

      for (Operation *candidate : wmmaOps) {
        if (scheduled.contains(candidate))
          continue;
        if (!canSchedule(candidate, scheduled, opsInSequence))
          continue;

        // Score: +2 for matching A, +2 for matching B, +1 for chaining C.
        int score = 0;
        if (getMatrixA(candidate) == prevA)
          score += 2;
        if (getMatrixB(candidate) == prevB)
          score += 2;
        // Smaller bonus if candidate uses prev's result as its accumulator.
        if (getMatrixC(candidate) == prev->getResult(0))
          score += 1;

        if (score > bestScore) {
          bestScore = score;
          bestCandidate = candidate;
        }
      }

      // Fallback if no best found (pick first available).
      if (!bestCandidate) {
        for (Operation *candidate : wmmaOps) {
          if (!scheduled.contains(candidate) &&
              canSchedule(candidate, scheduled, opsInSequence)) {
            bestCandidate = candidate;
            break;
          }
        }
      }

      scheduled.insert(bestCandidate);
    }

    // Reorder operations in the IR.
    Operation *insertPoint = wmmaOps[0];
    for (Operation *op : scheduled) {
      if (op != insertPoint)
        op->moveAfter(insertPoint);
      insertPoint = op;
    }

    // Set reuse flags based on new order.
    setReuseFlagsForOrder(scheduled.getArrayRef());

    // Insert sched.barrier ops to prevent LLVM from reordering.
    insertSchedBarriers(scheduled.getArrayRef());
  }

  /// Insert rocdl.sched.barrier ops before, between, and after matrix ops.
  static void insertSchedBarriers(ArrayRef<Operation *> wmmaOps) {
    if (wmmaOps.empty())
      return;

    OpBuilder builder(wmmaOps.front()->getContext());

    // Insert barrier before the first op.
    builder.setInsertionPoint(wmmaOps.front());
    ROCDL::SchedBarrier::create(builder, wmmaOps.front()->getLoc(),
                                /*mask=*/0);

    // Insert barrier after each op (covers between ops and after last).
    for (Operation *op : wmmaOps) {
      builder.setInsertionPointAfter(op);
      ROCDL::SchedBarrier::create(builder, op->getLoc(), /*mask=*/0);
    }
  }

  /// Sets reuse flags based on the operation ordering.
  static void setReuseFlagsForOrder(ArrayRef<Operation *> wmmaOps) {
    for (auto i : llvm::seq<size_t>(1, wmmaOps.size())) {
      Operation *curr = wmmaOps[i];
      Operation *prev = wmmaOps[i - 1];

      if (getMatrixA(curr) == getMatrixA(prev))
        setReuseA(prev, true);

      if (getMatrixB(curr) == getMatrixB(prev))
        setReuseB(prev, true);
    }
  }
};

} // namespace
