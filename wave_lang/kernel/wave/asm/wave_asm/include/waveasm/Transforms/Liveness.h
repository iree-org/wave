// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_LIVENESS_H
#define WaveASM_TRANSFORMS_LIVENESS_H

#include "mlir/IR/Value.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
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
  bool contains(int64_t point) const { return start <= point && point <= end; }

  /// Get the duration of this range
  int64_t length() const { return end - start + 1; }

  /// Validate invariants: end >= start, size > 0, alignment > 0
  bool isValid() const { return end >= start && size > 0 && alignment > 0; }

  /// Assert that invariants hold (for debugging)
  void assertValid() const {
    assert(end >= start && "Invalid live range: end < start");
    assert(size > 0 && "Invalid live range: size must be positive");
    assert(alignment > 0 && "Invalid live range: alignment must be positive");
  }
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

  /// AGPR ranges sorted by start point (for linear scan)
  llvm::SmallVector<LiveRange> aregRanges;

  /// Definition points for each register
  llvm::DenseMap<mlir::Value, int64_t> defPoints;

  /// Use points for each register
  llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t>> usePoints;

  /// Maximum VGPR pressure (peak overlapping VGPRs)
  int64_t maxVRegPressure = 0;

  /// Maximum SGPR pressure (peak overlapping SGPRs)
  int64_t maxSRegPressure = 0;

  /// Maximum AGPR pressure (peak overlapping AGPRs)
  int64_t maxARegPressure = 0;

  /// Get live range for a value
  const LiveRange *getRange(mlir::Value value) const {
    auto it = ranges.find(value);
    return it != ranges.end() ? &it->second : nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Region Utilities
//===----------------------------------------------------------------------===//

/// Collect all operations in a block and its nested regions into a flat list.
/// Operations are visited in pre-order: parent first, then children.
/// This is used by liveness analysis, hazard mitigation, and other passes
/// that need a linearized view of the instruction stream.
void collectOpsRecursive(mlir::Block &block,
                         llvm::SmallVectorImpl<mlir::Operation *> &ops);

//===----------------------------------------------------------------------===//
// Liveness Analysis Functions
//===----------------------------------------------------------------------===//

/// Compute liveness information for a kernel program
/// @param program The kernel program to analyze
/// @return Complete liveness information
LivenessInfo computeLiveness(ProgramOp program);

/// Compute maximum register pressure using sweep algorithm
int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges);

/// Compute maximum register pressure and the instruction index where the
/// peak occurs.  If \p peakPoint is non-null, the index is written there.
int64_t computeMaxPressure(llvm::ArrayRef<LiveRange> ranges,
                           int64_t *peakPoint);

/// Dump detailed peak pressure information for diagnostics.
/// Shows which values are live at the peak point, categorized by defining op.
/// Output is gated behind LLVM_DEBUG (use -debug-only=waveasm-liveness).
void dumpPeakPressureInfo(const LivenessInfo &info,
                          llvm::ArrayRef<mlir::Operation *> ops,
                          RegClass regClass);

/// Validate that the program is in SSA form
/// Each virtual register should be defined exactly once
/// (with exceptions for loop control and accumulator registers)
mlir::LogicalResult validateSSA(ProgramOp program,
                                llvm::DenseSet<int64_t> loopControlSRegs = {},
                                llvm::DenseSet<int64_t> accumulatorVRegs = {});

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_LIVENESS_H
