// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/RegAlloc.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

void LinearScanRegAlloc::expireRanges(
    llvm::SmallVector<std::tuple<int64_t, LiveRange, int64_t>> &active,
    int64_t currentPoint, RegPool &pool, AllocationStats &stats) {

  // Remove ranges that ended before currentPoint
  active.erase(
      std::remove_if(active.begin(), active.end(),
                     [&](const auto &entry) {
                       auto [endPoint, range, physReg] = entry;
                       if (endPoint < currentPoint) {
                         pool.freeRange(physReg, range.size);
                         stats.rangesExpired++;
                         return true;
                       }
                       return false;
                     }),
      active.end());
}

//===----------------------------------------------------------------------===//
// Main Allocation Algorithm (Pure SSA)
//===----------------------------------------------------------------------===//

FailureOr<std::pair<PhysicalMapping, AllocationStats>>
LinearScanRegAlloc::allocate(ProgramOp program) {
  PhysicalMapping mapping;
  AllocationStats stats;

  // Step 1: Validate SSA
  if (failed(validateSSA(program))) {
    return program.emitOpError() << "SSA validation failed before allocation";
  }

  // Step 2: Compute liveness
  LivenessInfo liveness = computeLiveness(program, /*useCFG=*/true);

  stats.totalVRegs = liveness.vregRanges.size();
  stats.totalSRegs = liveness.sregRanges.size();

  // Step 3: Create register pools with reserved registers
  RegPool vgprPool(RegClass::VGPR, maxVGPRs, reservedVGPRs);
  RegPool sgprPool(RegClass::SGPR, maxSGPRs, reservedSGPRs);

  // Step 4: Handle precolored values (from ABI args like tid, kernarg)
  for (const auto &[value, physIdx] : precoloredValues) {
    if (isVGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      vgprPool.reserve(physIdx, getRegSize(value.getType()));
    } else if (isSGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      sgprPool.reserve(physIdx, getRegSize(value.getType()));
    }
  }

  // Step 5: Allocate VGPRs using linear scan
  llvm::SmallVector<std::tuple<int64_t, LiveRange, int64_t>> activeVRegs;

  for (const LiveRange &range : liveness.vregRanges) {
    // Skip precolored values
    if (precoloredValues.contains(range.reg)) {
      continue;
    }

    // Expire finished ranges
    expireRanges(activeVRegs, range.start, vgprPool, stats);

    // Check if this value is tied to another (MFMA accumulator case)
    int64_t physReg = -1;
    auto tiedIt = tiedOperands.find(range.reg);
    if (tiedIt != tiedOperands.end()) {
      // Use the same physical register as the tied-to operand
      Value tiedTo = tiedIt->second;
      auto mappingIt = mapping.valueToPhysReg.find(tiedTo);
      if (mappingIt != mapping.valueToPhysReg.end()) {
        physReg = mappingIt->second;
        // Don't add to active list - the tied-to operand is already managing
        // the physical register lifetime
        mapping.valueToPhysReg[range.reg] = physReg;
        stats.rangesAllocated++;
        continue;
      }
      // If tied-to not yet allocated, fall through to normal allocation
    }

    // Allocate physical register(s)
    if (range.size == 1) {
      physReg = vgprPool.allocSingle();
    } else {
      physReg = vgprPool.allocRange(range.size, range.alignment);
    }

    if (physReg < 0) {
      return program.emitOpError()
             << "Failed to allocate VGPR for value. "
             << "Peak pressure: " << liveness.maxVRegPressure
             << ", limit: " << maxVGPRs;
    }

    // Record mapping: Value -> physical register
    mapping.valueToPhysReg[range.reg] = physReg;

    // Add to active list (sorted by end point)
    activeVRegs.push_back({range.end, range, physReg});
    llvm::sort(activeVRegs,
               [](const auto &a, const auto &b) {
                 return std::get<0>(a) < std::get<0>(b);
               });

    stats.rangesAllocated++;
  }

  stats.peakVGPRs = vgprPool.getPeakUsage();

  // Step 6: Allocate SGPRs using linear scan
  llvm::SmallVector<std::tuple<int64_t, LiveRange, int64_t>> activeSRegs;

  for (const LiveRange &range : liveness.sregRanges) {
    // Skip precolored values
    if (precoloredValues.contains(range.reg)) {
      continue;
    }

    // Expire finished ranges
    expireRanges(activeSRegs, range.start, sgprPool, stats);

    // Allocate physical register(s)
    int64_t physReg;
    if (range.size == 1) {
      physReg = sgprPool.allocSingle();
    } else {
      physReg = sgprPool.allocRange(range.size, range.alignment);
    }

    if (physReg < 0) {
      return program.emitOpError()
             << "Failed to allocate SGPR for value. "
             << "Peak pressure: " << liveness.maxSRegPressure
             << ", limit: " << maxSGPRs;
    }

    // Record mapping: Value -> physical register
    mapping.valueToPhysReg[range.reg] = physReg;

    // Add to active list (sorted by end point)
    activeSRegs.push_back({range.end, range, physReg});
    llvm::sort(activeSRegs,
               [](const auto &a, const auto &b) {
                 return std::get<0>(a) < std::get<0>(b);
               });

    stats.rangesAllocated++;
  }

  stats.peakSGPRs = sgprPool.getPeakUsage();

  return std::make_pair(mapping, stats);
}
