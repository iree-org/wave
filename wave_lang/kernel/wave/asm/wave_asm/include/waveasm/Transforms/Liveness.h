// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_LIVENESS_H
#define WaveASM_TRANSFORMS_LIVENESS_H

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// Live Range
//===----------------------------------------------------------------------===//

/// Represents the lifetime of a virtual register in the instruction stream.
struct LiveRange {
  /// The virtual register value
  mlir::Value reg;

  /// Instruction index of definition
  int64_t start = 0;

  /// Instruction index of last use
  int64_t end = 0;

  /// Number of consecutive registers (for register ranges)
  int64_t size = 1;

  /// Required physical alignment (1, 2, 4, etc.)
  int64_t alignment = 1;

  /// Register class (VGPR or SGPR)
  RegClass regClass = RegClass::VGPR;

  /// Check if this range overlaps with another
  bool overlaps(const LiveRange &other) const {
    return !(end < other.start || other.end < start);
  }

  /// Check if an instruction index is within this range
  bool contains(int64_t point) const {
    return start <= point && point <= end;
  }

  /// Get the duration of this range
  int64_t length() const { return end - start + 1; }
};

//===----------------------------------------------------------------------===//
// Basic Block for CFG
//===----------------------------------------------------------------------===//

/// Represents a basic block in the control flow graph
struct BasicBlock {
  /// Unique block identifier
  int64_t id = 0;

  /// Optional label at the start of this block
  std::optional<llvm::StringRef> label;

  /// First instruction index (inclusive)
  int64_t startIdx = 0;

  /// Last instruction index (inclusive)
  int64_t endIdx = 0;

  /// Successor blocks
  llvm::SmallVector<BasicBlock *> successors;

  /// Predecessor blocks
  llvm::SmallVector<BasicBlock *> predecessors;

  /// Registers used before being defined in this block
  llvm::DenseSet<mlir::Value> useSet;

  /// Registers defined in this block
  llvm::DenseSet<mlir::Value> defSet;

  /// Registers live at block entry
  llvm::DenseSet<mlir::Value> liveIn;

  /// Registers live at block exit
  llvm::DenseSet<mlir::Value> liveOut;
};

//===----------------------------------------------------------------------===//
// Control Flow Graph
//===----------------------------------------------------------------------===//

/// Control flow graph built from labels and branches
class CFG {
public:
  /// Build CFG from a kernel program
  static CFG build(ProgramOp program);

  /// Get all basic blocks
  llvm::ArrayRef<BasicBlock> getBlocks() const {
    return llvm::ArrayRef<BasicBlock>(blocks.begin(), blocks.end());
  }

  /// Get the entry block
  BasicBlock *getEntryBlock() { return blocks.empty() ? nullptr : &blocks[0]; }

  /// Get block for a label
  BasicBlock *getBlockForLabel(llvm::StringRef label);

  /// Check if the CFG contains loops
  bool hasLoops() const;

  /// Get number of blocks
  size_t size() const { return blocks.size(); }

  /// Get mutable blocks for liveness analysis
  llvm::SmallVector<BasicBlock, 16> &getMutableBlocks() { return blocks; }

private:
  llvm::SmallVector<BasicBlock, 16> blocks;
  llvm::DenseMap<llvm::StringRef, BasicBlock *> labelToBlock;
};

//===----------------------------------------------------------------------===//
// Liveness Information
//===----------------------------------------------------------------------===//

/// Complete liveness analysis results
struct LivenessInfo {
  /// Live ranges for all virtual registers
  llvm::DenseMap<mlir::Value, LiveRange> ranges;

  /// VGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> vregRanges;

  /// SGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> sregRanges;

  /// Definition points for each register
  llvm::DenseMap<mlir::Value, int64_t> defPoints;

  /// Use points for each register
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> usePoints;

  /// Maximum VGPR pressure (peak overlapping VGPRs)
  int64_t maxVRegPressure = 0;

  /// Maximum SGPR pressure (peak overlapping SGPRs)
  int64_t maxSRegPressure = 0;

  /// Get live range for a value
  const LiveRange *getRange(mlir::Value value) const {
    auto it = ranges.find(value);
    return it != ranges.end() ? &it->second : nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Liveness Analysis Functions
//===----------------------------------------------------------------------===//

/// Compute liveness information for a kernel program
/// @param program The kernel program to analyze
/// @param useCFG Whether to use CFG-based analysis (needed for loops)
/// @return Complete liveness information
LivenessInfo computeLiveness(ProgramOp program, bool useCFG = true);

/// Compute local use/def sets for a basic block
void computeBlockLocalInfo(BasicBlock &block,
                           llvm::ArrayRef<mlir::Operation *> instructions);

/// Run backward dataflow to compute live_in/live_out sets
/// Iterates until fixed point is reached
void computeCFGLiveness(CFG &cfg,
                        llvm::ArrayRef<mlir::Operation *> instructions);

/// Compute maximum register pressure using sweep algorithm
int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges);

/// Validate that the program is in SSA form
/// Each virtual register should be defined exactly once
/// (with exceptions for loop control and accumulator registers)
mlir::LogicalResult validateSSA(ProgramOp program,
                                 llvm::DenseSet<int64_t> loopControlSRegs = {},
                                 llvm::DenseSet<int64_t> accumulatorVRegs = {});

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_LIVENESS_H
