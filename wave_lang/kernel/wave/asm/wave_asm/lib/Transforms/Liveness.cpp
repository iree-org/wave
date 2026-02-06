// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include <algorithm>

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Region Utilities
//===----------------------------------------------------------------------===//

void collectOpsRecursive(Block &block,
                         llvm::SmallVectorImpl<Operation *> &ops) {
  for (Operation &op : block) {
    ops.push_back(&op);
    for (Region &region : op.getRegions()) {
      for (Block &nestedBlock : region) {
        collectOpsRecursive(nestedBlock, ops);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pressure Computation
//===----------------------------------------------------------------------===//

int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges) {
  if (ranges.empty())
    return 0;

  // Create events: (point, delta, isStart)
  llvm::SmallVector<std::tuple<int64_t, int64_t, bool>> events;

  for (const auto &r : ranges) {
    events.push_back({r.start, r.size, true});     // Start: add
    events.push_back({r.end + 1, -r.size, false}); // End+1: remove
  }

  // Sort by point, then starts before ends
  llvm::sort(events, [](const auto &a, const auto &b) {
    if (std::get<0>(a) != std::get<0>(b))
      return std::get<0>(a) < std::get<0>(b);
    // At same point, process ends before starts to avoid counting both
    return !std::get<2>(a) && std::get<2>(b);
  });

  int64_t currentPressure = 0;
  int64_t maxPressure = 0;

  // Reasonable upper bound for register pressure (no GPU has this many regs)
  constexpr int64_t kMaxReasonablePressure = 1000000;

  for (const auto &[point, delta, isStart] : events) {
    currentPressure += delta;

    // Sanity check for overflow or computation errors
    assert(currentPressure >= 0 &&
           "Negative register pressure - possible overflow or bug");
    assert(currentPressure < kMaxReasonablePressure &&
           "Register pressure exceeds reasonable bounds - possible overflow");

    maxPressure = std::max(maxPressure, currentPressure);
  }

  return maxPressure;
}

//===----------------------------------------------------------------------===//
// Main Liveness Computation (Pure SSA)
//===----------------------------------------------------------------------===//

LivenessInfo computeLiveness(ProgramOp program) {
  LivenessInfo info;

  // Collect all operations in order, recursively walking into regions
  llvm::SmallVector<Operation *> ops;
  collectOpsRecursive(program.getBodyBlock(), ops);

  if (ops.empty())
    return info;

  // Build op-to-index map for range extension
  llvm::DenseMap<Operation *, int64_t> opToIdx;
  for (int64_t idx = 0; idx < static_cast<int64_t>(ops.size()); ++idx) {
    opToIdx[ops[idx]] = idx;
  }

  // Pass 1: Collect def and use points from instructions.
  // Also include block arguments of loop ops as definitions.
  for (int64_t idx = 0; idx < static_cast<int64_t>(ops.size()); ++idx) {
    Operation *op = ops[idx];

    // Process defs: results are definitions
    for (Value def : op->getResults()) {
      if (isVirtualRegType(def.getType())) {
        if (!info.defPoints.contains(def)) {
          info.defPoints[def] = idx;
        }
      }
    }

    // Process uses: operands are uses
    for (Value use : op->getOperands()) {
      if (isVirtualRegType(use.getType())) {
        info.usePoints[use].push_back(idx);
      }
    }

    // Block arguments of while ops are defs at the loop op index
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (isVirtualRegType(arg.getType())) {
            if (!info.defPoints.contains(arg)) {
              info.defPoints[arg] = idx;
            }
          }
        }
      }
    }
  }

  // Pass 2: Build basic live ranges from def/use points
  for (const auto &[value, defPoint] : info.defPoints) {
    LiveRange range;
    range.reg = value;
    range.start = defPoint;

    // Find last use
    auto useIt = info.usePoints.find(value);
    if (useIt != info.usePoints.end() && !useIt->second.empty()) {
      range.end = *std::max_element(useIt->second.begin(), useIt->second.end());
    } else {
      // No uses: range is just the definition point
      range.end = defPoint;
    }

    // For loop op block args, extend live range to cover entire loop body
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (parentOp && isa<LoopOp>(parentOp)) {
        Block *body = blockArg.getOwner();
        Operation *terminator = body->getTerminator();
        if (terminator) {
          auto termIt = opToIdx.find(terminator);
          if (termIt != opToIdx.end()) {
            range.end = std::max(range.end, termIt->second);
          }
        }
      }
    }

    // Get size and alignment from type
    Type ty = value.getType();
    range.size = getRegSize(ty);
    range.alignment = getRegAlignment(ty);
    if (auto regClass = getRegClass(ty)) {
      range.regClass = *regClass;
    }

    info.ranges[value] = range;
  }

  // Pass 2b: Extend live ranges for values used inside loop bodies.
  // Any value used inside a loop body is used on EVERY iteration, so its
  // live range must extend from its definition to the end of the loop body.
  for (const auto &[value, defPoint] : info.defPoints) {
    auto it = info.ranges.find(value);
    if (it == info.ranges.end())
      continue;

    // Check all use points for this value. If any use is inside a loop body,
    // extend the range to cover the entire loop body.
    auto useIt = info.usePoints.find(value);
    if (useIt == info.usePoints.end())
      continue;

    for (int64_t useIdx : useIt->second) {
      if (useIdx >= static_cast<int64_t>(ops.size()))
        continue;
      Operation *useOp = ops[useIdx];

      // Walk up parent chain to find enclosing loop ops
      Operation *parent = useOp->getParentOp();
      while (parent && !isa<ProgramOp>(parent)) {
        if (auto loopOp = dyn_cast<LoopOp>(parent)) {
          // Check if the value is defined inside the loop body
          // (at any nesting depth). Values defined inside are recomputed
          // each iteration and should keep their natural live ranges
          // within the iteration. Only values defined OUTSIDE need
          // extension across the loop.
          bool definedInside = false;
          if (auto defOp = value.getDefiningOp()) {
            // Check if defOp is anywhere inside the loop's region,
            // not just a direct child. This handles values defined
            // inside nested if/loop ops within the loop body.
            definedInside = loopOp->isProperAncestor(defOp);
          } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
            // BlockArguments don't have a defining op. Check if the
            // block argument's parent op is the loop or nested inside it.
            Operation *argParentOp = blockArg.getOwner()->getParentOp();
            definedInside = (argParentOp == loopOp.getOperation()) ||
                            loopOp->isProperAncestor(argParentOp);
          }

          if (!definedInside) {
            // Extend end to cover the entire loop body (value is
            // used every iteration, must survive until loop exits)
            Block &body = loopOp.getBodyBlock();
            Operation *terminator = body.getTerminator();
            if (terminator) {
              auto termIt = opToIdx.find(terminator);
              if (termIt != opToIdx.end()) {
                it->second.end = std::max(it->second.end, termIt->second);
              }
            }
            // Extend start back to the loop op
            auto loopIt = opToIdx.find(loopOp.getOperation());
            if (loopIt != opToIdx.end()) {
              it->second.start = std::min(it->second.start, loopIt->second);
            }
          }
        }
        parent = parent->getParentOp();
      }
    }
  }

  // Note: Pass 3 (CFG-based backward dataflow liveness extension) has been
  // removed. It was needed for the old label-based control flow path where
  // loop back-edges were represented as explicit branch instructions. With
  // region-based control flow (LoopOp/IfOp), Pass 2 and Pass 2b above
  // already handle all necessary live range extensions by directly inspecting
  // the region structure.

  // Pass 4: Categorize ranges by register class and sort by start
  for (const auto &[value, range] : info.ranges) {
    if (range.regClass == RegClass::VGPR) {
      info.vregRanges.push_back(range);
    } else if (range.regClass == RegClass::SGPR) {
      info.sregRanges.push_back(range);
    }
  }

  // Sort by (start, end) for linear scan
  auto sortByStart = [](const LiveRange &a, const LiveRange &b) {
    if (a.start != b.start)
      return a.start < b.start;
    return a.end < b.end;
  };

  llvm::sort(info.vregRanges, sortByStart);
  llvm::sort(info.sregRanges, sortByStart);

  // Pass 5: Compute pressure
  info.maxVRegPressure = computeMaxPressure(info.vregRanges);
  info.maxSRegPressure = computeMaxPressure(info.sregRanges);

  return info;
}

//===----------------------------------------------------------------------===//
// SSA Validation (Pure SSA - simplified)
//===----------------------------------------------------------------------===//

LogicalResult validateSSA(ProgramOp program,
                          llvm::DenseSet<int64_t> /*loopControlSRegs*/,
                          llvm::DenseSet<int64_t> /*accumulatorVRegs*/) {
  // In pure SSA, MLIR already enforces single definition for each Value
  // This function can validate additional constraints if needed
  llvm::DenseSet<Value> definitions;

  for (Operation &op : program.getBodyBlock()) {
    for (Value def : op.getResults()) {
      if (!isVirtualRegType(def.getType()))
        continue;

      auto [it, inserted] = definitions.insert(def);
      if (!inserted) {
        // This shouldn't happen in valid MLIR SSA
        return op.emitOpError()
               << "SSA violation: value defined multiple times.";
      }
    }
  }

  return success();
}

} // namespace waveasm
