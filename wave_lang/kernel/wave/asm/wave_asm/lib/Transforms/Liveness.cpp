// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// Helper: Check if operation is a branch
//===----------------------------------------------------------------------===//

static bool isBranchOp(Operation *op) {
  // Check for branch ops by operation name
  llvm::StringRef name = op->getName().getStringRef();
  return name.contains("s_branch") || name.contains("s_cbranch") ||
         name.contains("s_setpc") || name.contains("s_swappc");
}

static bool isConditionalBranchOp(Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name.contains("s_cbranch");
}

//===----------------------------------------------------------------------===//
// CFG Construction
//===----------------------------------------------------------------------===//

CFG CFG::build(ProgramOp program) {
  CFG cfg;

  // Collect all operations in order
  llvm::SmallVector<Operation *> ops;
  for (Operation &op : program.getBodyBlock()) {
    ops.push_back(&op);
  }

  if (ops.empty())
    return cfg;

  // Pass 1: Find block start points (labels and after branches)
  llvm::SetVector<int64_t> blockStarts;
  blockStarts.insert(0); // First instruction starts a block

  for (int64_t i = 0; i < static_cast<int64_t>(ops.size()); ++i) {
    Operation *op = ops[i];

    // Labels start a new block
    if (isa<LabelOp>(op)) {
      blockStarts.insert(i);
    }

    // Instruction after a branch starts a new block
    if (isBranchOp(op) && i + 1 < static_cast<int64_t>(ops.size())) {
      blockStarts.insert(i + 1);
    }
  }

  // Sort block starts
  llvm::SmallVector<int64_t> sortedStarts(blockStarts.begin(),
                                           blockStarts.end());
  llvm::sort(sortedStarts);

  // Pass 2: Create basic blocks
  for (size_t i = 0; i < sortedStarts.size(); ++i) {
    BasicBlock block;
    block.id = i;
    block.startIdx = sortedStarts[i];
    block.endIdx = (i + 1 < sortedStarts.size())
                       ? sortedStarts[i + 1] - 1
                       : static_cast<int64_t>(ops.size()) - 1;

    // Check if this block starts with a label
    if (auto labelOp = dyn_cast<LabelOp>(ops[block.startIdx])) {
      block.label = labelOp.getName();
    }

    cfg.blocks.push_back(block);
  }

  // Fix up label->block mapping after blocks are in place
  cfg.labelToBlock.clear();
  for (auto &block : cfg.blocks) {
    if (block.label) {
      cfg.labelToBlock[*block.label] = &block;
    }
  }

  // Pass 3: Connect edges based on control flow
  for (size_t i = 0; i < cfg.blocks.size(); ++i) {
    BasicBlock &block = cfg.blocks[i];
    Operation *lastOp = ops[block.endIdx];

    bool isBranch = isBranchOp(lastOp);
    bool isConditional = isConditionalBranchOp(lastOp);

    if (isBranch) {
      // Get target from label operand if present
      // Check both StringAttr and SymbolRefAttr (branches use SymbolRefAttr)
      llvm::StringRef targetLabel;
      if (auto targetAttr = lastOp->getAttrOfType<StringAttr>("target")) {
        targetLabel = targetAttr.getValue();
      } else if (auto targetAttr = lastOp->getAttrOfType<SymbolRefAttr>("target")) {
        targetLabel = targetAttr.getRootReference().getValue();
      }
      if (!targetLabel.empty()) {
        if (BasicBlock *targetBlock = cfg.getBlockForLabel(targetLabel)) {
          block.successors.push_back(targetBlock);
          targetBlock->predecessors.push_back(&block);
        }
      }

      // Conditional branches also fall through
      if (isConditional && i + 1 < cfg.blocks.size()) {
        BasicBlock *nextBlock = &cfg.blocks[i + 1];
        block.successors.push_back(nextBlock);
        nextBlock->predecessors.push_back(&block);
      }
    } else {
      // Non-branch: fall through to next block
      if (i + 1 < cfg.blocks.size()) {
        BasicBlock *nextBlock = &cfg.blocks[i + 1];
        block.successors.push_back(nextBlock);
        nextBlock->predecessors.push_back(&block);
      }
    }
  }

  return cfg;
}

BasicBlock *CFG::getBlockForLabel(llvm::StringRef label) {
  auto it = labelToBlock.find(label);
  return it != labelToBlock.end() ? it->second : nullptr;
}

bool CFG::hasLoops() const {
  // Simple cycle detection: check if any block is reachable from itself
  for (const auto &block : blocks) {
    if (!block.predecessors.empty()) {
      // Check if any predecessor has a higher index (back edge)
      for (const BasicBlock *pred : block.predecessors) {
        if (pred->id >= block.id) {
          return true;
        }
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Local Use/Def Computation (Pure SSA)
//===----------------------------------------------------------------------===//

void computeBlockLocalInfo(BasicBlock &block,
                           llvm::ArrayRef<Operation *> instructions) {
  block.useSet.clear();
  block.defSet.clear();

  for (int64_t i = block.startIdx; i <= block.endIdx; ++i) {
    Operation *op = instructions[i];

    // In pure SSA, operands are uses and results are defs
    // Process uses: add to use set if not already defined in this block
    for (Value use : op->getOperands()) {
      if (isVirtualRegType(use.getType())) {
        if (!block.defSet.contains(use)) {
          block.useSet.insert(use);
        }
      }
    }

    // Process defs: results are definitions
    for (Value def : op->getResults()) {
      if (isVirtualRegType(def.getType())) {
        block.defSet.insert(def);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Backward Dataflow Analysis
//===----------------------------------------------------------------------===//

void computeCFGLiveness(CFG &cfg,
                        llvm::ArrayRef<Operation *> instructions) {
  auto &blocks = cfg.getMutableBlocks();

  // Compute local use/def sets for each block
  for (auto &block : blocks) {
    computeBlockLocalInfo(block, instructions);
  }

  // Iterate to fixed point
  bool changed = true;
  while (changed) {
    changed = false;

    // Process blocks in reverse order (more efficient for backward analysis)
    for (int64_t i = cfg.size() - 1; i >= 0; --i) {
      BasicBlock &block = blocks[i];

      // Compute live_out = union of live_in of all successors
      llvm::DenseSet<Value> newLiveOut;
      for (BasicBlock *succ : block.successors) {
        for (Value v : succ->liveIn) {
          newLiveOut.insert(v);
        }
      }

      // Compute live_in = use union (live_out - def)
      llvm::DenseSet<Value> newLiveIn = block.useSet;
      for (Value v : newLiveOut) {
        if (!block.defSet.contains(v)) {
          newLiveIn.insert(v);
        }
      }

      // Check for changes
      if (newLiveIn != block.liveIn || newLiveOut != block.liveOut) {
        block.liveIn = std::move(newLiveIn);
        block.liveOut = std::move(newLiveOut);
        changed = true;
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

  for (const auto &[point, delta, isStart] : events) {
    currentPressure += delta;
    maxPressure = std::max(maxPressure, currentPressure);
  }

  return maxPressure;
}

//===----------------------------------------------------------------------===//
// Main Liveness Computation (Pure SSA)
//===----------------------------------------------------------------------===//

LivenessInfo computeLiveness(ProgramOp program, bool useCFG) {
  LivenessInfo info;

  // Collect all operations in order
  llvm::SmallVector<Operation *> ops;
  for (Operation &op : program.getBodyBlock()) {
    ops.push_back(&op);
  }

  if (ops.empty())
    return info;

  // Pass 1: Collect def and use points from instructions
  // In pure SSA, results are defs and operands are uses
  for (int64_t idx = 0; idx < static_cast<int64_t>(ops.size()); ++idx) {
    Operation *op = ops[idx];

    // Process defs: results are definitions
    for (Value def : op->getResults()) {
      if (isVirtualRegType(def.getType())) {
        // First definition wins (SSA guarantees single def)
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

    // Get size and alignment from type
    Type ty = value.getType();
    range.size = getRegSize(ty);
    range.alignment = getRegAlignment(ty);
    if (auto regClass = getRegClass(ty)) {
      range.regClass = *regClass;
    }

    info.ranges[value] = range;
  }

  // Pass 3: Extend ranges using CFG analysis (for loops)
  if (useCFG) {
    CFG cfg = CFG::build(program);

    if (cfg.hasLoops()) {
      computeCFGLiveness(cfg, ops);

      // Extend ranges based on CFG liveness
      for (const auto &block : cfg.getBlocks()) {
        // For each value live at block entry, extend its range to cover the
        // block
        for (Value v : block.liveIn) {
          auto it = info.ranges.find(v);
          if (it != info.ranges.end()) {
            // Extend end to at least cover this block
            it->second.end = std::max(it->second.end, block.endIdx);
          }
        }

        // For each value live at block exit, extend its range
        for (Value v : block.liveOut) {
          auto it = info.ranges.find(v);
          if (it != info.ranges.end()) {
            it->second.end = std::max(it->second.end, block.endIdx);
          }
        }
      }
    }
  }

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
