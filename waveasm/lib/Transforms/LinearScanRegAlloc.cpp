// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Liveness.h"
#include "waveasm/Transforms/RegAlloc.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// ActiveRange - Represents an allocated range in the active list
//===----------------------------------------------------------------------===//

namespace {

/// Represents an active live range during linear scan register allocation.
/// The active list is sorted by endPoint to efficiently expire ranges.
/// When a range ends (endPoint < currentPoint), its physical register is freed.
struct ActiveRange {
  int64_t endPoint; ///< Program point where this range ends (exclusive)
  LiveRange range;  ///< The original live range (contains Value, start, size)
  int64_t physReg;  ///< Allocated physical register index

  bool operator<(const ActiveRange &other) const {
    return endPoint < other.endPoint;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Expire ranges that ended before currentPoint, returning registers to pool.
static void expireRanges(llvm::SmallVectorImpl<ActiveRange> &active,
                         int64_t currentPoint, RegPool &pool,
                         AllocationStats &stats) {
  active.erase(std::remove_if(active.begin(), active.end(),
                              [&](const ActiveRange &entry) {
                                if (entry.endPoint < currentPoint) {
                                  assert(entry.range.size > 0 &&
                                         "Cannot free zero-sized range");
                                  pool.freeRange(entry.physReg,
                                                 entry.range.size);
                                  stats.rangesExpired++;
                                  return true;
                                }
                                return false;
                              }),
               active.end());
}

/// Insert a new active range while maintaining sorted order by end point.
/// Uses binary search for O(log n) insertion position finding.
static void insertActiveRange(llvm::SmallVectorImpl<ActiveRange> &active,
                              ActiveRange newRange) {
  auto insertPos = std::lower_bound(active.begin(), active.end(), newRange);
  active.insert(insertPos, newRange);
}

/// Try to allocate a physical register from the pool (lowest-first).
static std::optional<int64_t> tryAllocate(RegPool &pool, int64_t size,
                                          int64_t alignment) {
  int64_t physReg =
      (size == 1) ? pool.allocSingle() : pool.allocRange(size, alignment);
  if (physReg < 0)
    return std::nullopt;
  return physReg;
}

//===----------------------------------------------------------------------===//
// Concrete Allocation Strategies
//===----------------------------------------------------------------------===//

std::optional<int64_t>
BidirectionalStrategy::allocate(RegPool &pool, const LiveRange &range,
                                llvm::ArrayRef<LiveRange> allRanges,
                                int64_t maxPressure) {
  if (pool.getRegClass() != RegClass::VGPR || range.size <= 1)
    return std::nullopt;

  int64_t rangeLength = range.end - range.start;
  // Ranges whose length exceeds 75% of the program span are allocated from
  // the top. This targets buffer_load prefetch values (which span almost the
  // entire loop body) while leaving ds_read values (consumed within one half)
  // at the bottom. The ceiling is maxPressure (peak simultaneous VGPRs from
  // liveness), not maxRegs, to avoid allocating into the AGPR region.
  int64_t programEnd = 0;
  for (const auto &lr : allRanges)
    programEnd = std::max(programEnd, lr.end);
  int64_t threshold = (programEnd * thresholdPct) / 100;
  if (rangeLength > threshold) {
    int64_t physReg =
        pool.allocRangeFromTop(range.size, range.alignment, maxPressure);
    if (physReg >= 0)
      return physReg;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Register Class Allocation
//===----------------------------------------------------------------------===//

/// Find the best eviction candidate from the active list.
/// Prefers untied, size-1 ranges with the fewest use sites.
/// Among equal-cost candidates, picks the longest remaining range.
/// Returns the index into `active`, or -1 if no candidate is found.
static int64_t
findEvictionCandidate(const llvm::SmallVectorImpl<ActiveRange> &active,
                      const llvm::DenseMap<Value, Value> &tiedOperands,
                      const LivenessInfo *liveness, int64_t currentPoint) {
  int64_t bestIdx = -1;
  int64_t bestUseCount = std::numeric_limits<int64_t>::max();
  int64_t bestLength = -1;

  for (int64_t i = 0, e = active.size(); i < e; ++i) {
    const ActiveRange &ar = active[i];
    // Only spill size-1 values.
    if (ar.range.size != 1)
      continue;
    // Do not spill tied values.
    if (ar.range.isTied() || tiedOperands.contains(ar.range.reg))
      continue;
    // Must still be alive past the current point.
    if (ar.endPoint <= currentPoint)
      continue;

    int64_t useCount = 0;
    if (liveness) {
      auto it = liveness->usePoints.find(ar.range.reg);
      if (it != liveness->usePoints.end())
        useCount = it->second.size();
    }
    int64_t length = ar.endPoint - currentPoint;

    // Prefer fewer uses; break ties with longer remaining range
    // (evicting a long range frees the register for more time).
    if (useCount < bestUseCount ||
        (useCount == bestUseCount && length > bestLength)) {
      bestIdx = i;
      bestUseCount = useCount;
      bestLength = length;
    }
  }
  return bestIdx;
}

/// Allocate registers for a single register class (VGPR, SGPR, or AGPR).
/// This is the core linear scan algorithm, parameterized by register class.
/// An optional AllocationStrategy is consulted before the default bottom-up
/// allocation; when allocation fails and `altPool` is provided, the allocator
/// evicts an active range to the alternate register class before giving up.
static LogicalResult allocateRegClass(
    ArrayRef<LiveRange> ranges, RegPool &pool, PhysicalMapping &mapping,
    AllocationStats &stats, const llvm::DenseMap<Value, Value> &tiedOperands,
    const llvm::DenseMap<Value, int64_t> &precoloredValues,
    llvm::StringRef regClassName, ProgramOp program, int64_t maxRegs,
    int64_t maxPressure, AllocationStrategy *strategy, RegPool *altPool,
    llvm::SmallVectorImpl<SpillRecord> *spills, const LivenessInfo *liveness) {

  llvm::SmallVector<ActiveRange> active;

  for (const LiveRange &range : ranges) {
    // Skip precolored values - they're already mapped.
    if (precoloredValues.contains(range.reg))
      continue;

    // Expire finished ranges, returning registers to the pool.
    expireRanges(active, range.start, pool, stats);

    std::optional<int64_t> physReg;

    // Check if this value is tied to another value (must share same phys reg).
    // This handles MFMA accumulator tying AND while loop block arg coalescing.
    {
      auto tiedIt = tiedOperands.find(range.reg);
      if (tiedIt != tiedOperands.end()) {
        Value tiedTo = tiedIt->second;
        auto mappingIt = mapping.valueToPhysReg.find(tiedTo);
        if (mappingIt != mapping.valueToPhysReg.end()) {
          physReg = mappingIt->second;
          mapping.valueToPhysReg[range.reg] = *physReg;

          // Extend the physical register's lifetime to cover the tied result.
          bool foundInActive = false;
          for (size_t i = 0; i < active.size(); ++i) {
            if (active[i].physReg == *physReg) {
              foundInActive = true;
              if (range.end > active[i].endPoint) {
                active[i].endPoint = range.end;
                active[i].range = range;
                while (i + 1 < active.size() &&
                       active[i].endPoint > active[i + 1].endPoint) {
                  std::swap(active[i], active[i + 1]);
                  ++i;
                }
              }
              break;
            }
          }

          if (!foundInActive) {
            bool tiedToPrecolored = precoloredValues.contains(tiedTo);
            assert((tiedToPrecolored || pool.isFree(*physReg)) &&
                   "Tied register was re-allocated before re-reservation");
            pool.reserve(*physReg, range.size);
            insertActiveRange(active, {range.end, range, *physReg});
          }

          stats.rangesAllocated++;
          continue;
        }
        // If tied-to not yet allocated, fall through to normal allocation.
      }
    }

    // Consult the pluggable strategy (if any) before the bottom-up fallback.
    if (strategy)
      physReg = strategy->allocate(pool, range, ranges, maxPressure);
    if (!physReg)
      physReg = tryAllocate(pool, range.size, range.alignment);

    // Cross-class eviction: if allocation failed and an alternate pool is
    // available, evict the best candidate from the active list into the
    // alternate class, freeing its register for the incoming range.
    if (!physReg && altPool && altPool->hasFree() && spills) {
      int64_t victimIdx =
          findEvictionCandidate(active, tiedOperands, liveness, range.start);
      if (victimIdx >= 0) {
        ActiveRange victim = active[victimIdx];
        // Allocate a register in the alternate class for the victim.
        int64_t altReg = altPool->allocSingle();
        if (altReg >= 0) {
          // Free the victim's register back to the primary pool.
          pool.freeRange(victim.physReg, victim.range.size);
          active.erase(active.begin() + victimIdx);

          // Record the spill for later op insertion.
          spills->push_back(SpillRecord{victim.range.reg, victim.physReg,
                                        altReg, pool.getRegClass(),
                                        altPool->getRegClass()});

          // Retry allocation for the incoming range.
          physReg = tryAllocate(pool, range.size, range.alignment);
        }
      }
    }

    if (!physReg) {
      InFlightDiagnostic diag = mlir::emitError(range.reg.getLoc())
                                << "Failed to allocate " << regClassName
                                << ": kernel requires " << maxPressure
                                << " but only " << maxRegs << " are available";
      diag.attachNote(range.reg.getLoc())
          << "Register spilling is not supported; reduce register pressure "
             "(e.g., smaller tile sizes, fewer unrolled iterations).";
      return failure();
    }

    // Record mapping: Value -> physical register.
    mapping.valueToPhysReg[range.reg] = *physReg;

    // Add to active list, maintaining sorted order by end point.
    insertActiveRange(active, {range.end, range, *physReg});

    stats.rangesAllocated++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main Allocation Algorithm (Pure SSA)
//===----------------------------------------------------------------------===//

FailureOr<LinearScanRegAlloc::AllocResult>
LinearScanRegAlloc::allocate(ProgramOp program) {
  PhysicalMapping mapping;
  AllocationStats stats;
  llvm::SmallVector<SpillRecord> spills;

  // Step 1: Validate SSA.
  if (failed(validateSSA(program)))
    return program.emitOpError() << "SSA validation failed before allocation";

  // Step 2: Compute liveness (builds tied equivalence classes).
  LivenessInfo liveness = computeLiveness(program);

  // Merge loop tied pairs from liveness into the allocator's tiedOperands.
  for (const auto &[result, operand] : liveness.tiedClasses.tiedPairs) {
    if (!tiedOperands.contains(result))
      tiedOperands[result] = operand;
  }

  stats.totalVRegs = liveness.vregRanges.size();
  stats.totalSRegs = liveness.sregRanges.size();
  stats.totalARegs = liveness.aregRanges.size();

  // Step 3: Create register pools with reserved registers.
  RegPool vgprPool(RegClass::VGPR, maxVGPRs, reservedVGPRs);
  RegPool sgprPool(RegClass::SGPR, maxSGPRs, reservedSGPRs);
  RegPool agprPool(RegClass::AGPR, maxAGPRs, reservedAGPRs);

  // Step 4: Handle precolored values (from ABI args like tid, kernarg).
  for (const auto &[value, physIdx] : precoloredValues) {
    if (isVGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      vgprPool.reserve(physIdx, getRegSize(value.getType()));
    } else if (isAGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      agprPool.reserve(physIdx, getRegSize(value.getType()));
    } else if (isSGPRType(value.getType())) {
      mapping.valueToPhysReg[value] = physIdx;
      sgprPool.reserve(physIdx, getRegSize(value.getType()));
    }
  }

  // Step 5: Allocate VGPRs. On failure, evict to spare AGPRs.
  if (failed(allocateRegClass(liveness.vregRanges, vgprPool, mapping, stats,
                              tiedOperands, precoloredValues, "VGPR", program,
                              maxVGPRs, liveness.maxVRegPressure,
                              vgprStrategy.get(), &agprPool, &spills,
                              &liveness)))
    return failure();
  stats.peakVGPRs = vgprPool.getPeakUsage();

  // Step 6: Allocate SGPRs. On failure, evict to spare VGPRs.
  if (failed(allocateRegClass(liveness.sregRanges, sgprPool, mapping, stats,
                              tiedOperands, precoloredValues, "SGPR", program,
                              maxSGPRs, liveness.maxSRegPressure,
                              /*strategy=*/nullptr, &vgprPool, &spills,
                              &liveness)))
    return failure();
  stats.peakSGPRs = sgprPool.getPeakUsage();

  // Step 7: Allocate AGPRs. On failure, evict to spare VGPRs.
  if (failed(allocateRegClass(liveness.aregRanges, agprPool, mapping, stats,
                              tiedOperands, precoloredValues, "AGPR", program,
                              maxAGPRs, liveness.maxARegPressure,
                              /*strategy=*/nullptr, &vgprPool, &spills,
                              &liveness)))
    return failure();
  stats.peakAGPRs = agprPool.getPeakUsage();

  return AllocResult{std::move(mapping), stats, std::move(spills)};
}
