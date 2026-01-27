// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_REGALLOC_H
#define WaveASM_TRANSFORMS_REGALLOC_H

#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Allocation Statistics
//===----------------------------------------------------------------------===//

/// Statistics from register allocation
struct AllocationStats {
  int64_t peakVGPRs = 0;
  int64_t peakSGPRs = 0;
  int64_t totalVRegs = 0;
  int64_t totalSRegs = 0;
  int64_t rangesAllocated = 0;
  int64_t rangesExpired = 0;
};

//===----------------------------------------------------------------------===//
// Physical Mapping (Pure SSA)
//===----------------------------------------------------------------------===//

/// Maps SSA Values to physical register indices
class PhysicalMapping {
public:
  /// Get physical register index for a Value
  int64_t getPhysReg(mlir::Value value) const {
    auto it = valueToPhysReg.find(value);
    return it != valueToPhysReg.end() ? it->second : -1;
  }

  /// Set mapping for a Value
  void setPhysReg(mlir::Value value, int64_t physIdx) {
    valueToPhysReg[value] = physIdx;
  }

  /// Check if a Value is mapped
  bool hasMapping(mlir::Value value) const {
    return valueToPhysReg.contains(value);
  }

  /// Direct access for LinearScanRegAlloc
  llvm::DenseMap<mlir::Value, int64_t> valueToPhysReg;
};

//===----------------------------------------------------------------------===//
// Register Pool
//===----------------------------------------------------------------------===//

/// Pool of available physical registers
class RegPool {
public:
  RegPool(RegClass regClass, int64_t maxRegs,
          const llvm::DenseSet<int64_t> &reserved)
      : regClass(regClass), maxRegs(maxRegs) {
    // Initialize free list with all non-reserved registers
    for (int64_t i = 0; i < maxRegs; ++i) {
      if (!reserved.contains(i)) {
        freeList.push_back(i);
      }
    }
  }

  /// Reserve a specific register (for precoloring)
  void reserve(int64_t reg, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
      int64_t r = reg + i;
      auto it = std::find(freeList.begin(), freeList.end(), r);
      if (it != freeList.end()) {
        freeList.erase(it);
        allocated.insert(r);
      }
    }
    updatePeak();
  }

  /// Allocate a single register
  /// Returns -1 if allocation fails
  int64_t allocSingle() {
    if (freeList.empty())
      return -1;

    int64_t reg = freeList.front();
    freeList.erase(freeList.begin());
    allocated.insert(reg);
    updatePeak();
    return reg;
  }

  /// Allocate a contiguous range of registers with alignment
  /// Returns base register index, or -1 if allocation fails
  int64_t allocRange(int64_t size, int64_t alignment) {
    if (size <= 0)
      return -1;

    llvm::DenseSet<int64_t> freeSet(freeList.begin(), freeList.end());

    for (int64_t candidate : freeList) {
      // Check alignment
      if (candidate % alignment != 0)
        continue;

      // Check if all registers in range are free
      bool allFree = true;
      for (int64_t offset = 0; offset < size; ++offset) {
        int64_t reg = candidate + offset;
        if (reg >= maxRegs || !freeSet.contains(reg)) {
          allFree = false;
          break;
        }
      }

      if (allFree) {
        // Allocate the range
        for (int64_t offset = 0; offset < size; ++offset) {
          int64_t reg = candidate + offset;
          freeList.erase(std::find(freeList.begin(), freeList.end(), reg));
          allocated.insert(reg);
        }
        updatePeak();
        return candidate;
      }
    }

    return -1; // Allocation failed
  }

  /// Free a single register
  void freeSingle(int64_t reg) {
    if (!allocated.contains(reg))
      return;

    allocated.erase(reg);

    // Insert in sorted order
    auto it = std::lower_bound(freeList.begin(), freeList.end(), reg);
    freeList.insert(it, reg);
  }

  /// Free a range of registers
  void freeRange(int64_t base, int64_t size) {
    for (int64_t offset = 0; offset < size; ++offset) {
      freeSingle(base + offset);
    }
  }

  /// Get peak usage
  int64_t getPeakUsage() const { return peak; }

  /// Get current usage
  int64_t getCurrentUsage() const { return allocated.size(); }

  /// Get register class
  RegClass getRegClass() const { return regClass; }

private:
  void updatePeak() {
    peak = std::max(peak, static_cast<int64_t>(allocated.size()));
  }

  RegClass regClass;
  int64_t maxRegs;
  llvm::SmallVector<int64_t> freeList; // Sorted
  llvm::DenseSet<int64_t> allocated;
  int64_t peak = 0;
};

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocator (Pure SSA)
//===----------------------------------------------------------------------===//

/// Linear scan register allocator for pure SSA IR
class LinearScanRegAlloc {
public:
  LinearScanRegAlloc(int64_t maxVGPRs, int64_t maxSGPRs,
                     const llvm::DenseSet<int64_t> &reservedVGPRs,
                     const llvm::DenseSet<int64_t> &reservedSGPRs)
      : maxVGPRs(maxVGPRs), maxSGPRs(maxSGPRs), reservedVGPRs(reservedVGPRs),
        reservedSGPRs(reservedSGPRs) {}

  /// Precolor a Value to a specific physical register (for ABI args)
  void precolorValue(mlir::Value value, int64_t physIdx) {
    precoloredValues[value] = physIdx;
  }

  /// Add a tied operand constraint: result must get same physical reg as operand
  /// Used for MFMA accumulator tying where result overwrites the accumulator
  void addTiedOperand(mlir::Value result, mlir::Value operand) {
    tiedOperands[result] = operand;
  }

  /// Run allocation on a kernel program
  /// Returns the physical mapping and statistics, or failure if allocation fails
  mlir::FailureOr<std::pair<PhysicalMapping, AllocationStats>>
  allocate(ProgramOp program);

private:
  /// Process active ranges, expiring those that end before currentPoint
  void expireRanges(
      llvm::SmallVector<std::tuple<int64_t, LiveRange, int64_t>> &active,
      int64_t currentPoint, RegPool &pool, AllocationStats &stats);

  int64_t maxVGPRs;
  int64_t maxSGPRs;
  llvm::DenseSet<int64_t> reservedVGPRs;
  llvm::DenseSet<int64_t> reservedSGPRs;
  llvm::DenseMap<mlir::Value, int64_t> precoloredValues;
  llvm::DenseMap<mlir::Value, mlir::Value> tiedOperands;  // result -> operand
};

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_REGALLOC_H
