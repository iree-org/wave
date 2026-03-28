// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// VGPR Compaction Pass
//
// Re-assigns physical VGPRs after register allocation to minimize the peak
// register number. The linear scan allocator assigns VGPRs in instruction
// order (lowest-first), which causes fragmentation when interleaved
// buffer_load (long-lived) and ds_read (short-lived) instructions get
// interleaved register numbers. This pass reassigns them using a
// shortest-first greedy strategy that packs short-lived values into low
// registers, pushing long-lived values to a contiguous high range.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-vgpr-compaction"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMVGPRCOMPACTION
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

struct PhysVGPRRange {
  int64_t physIdx;
  int64_t size;
  int64_t alignment;
  int64_t defPoint;
  int64_t lastUsePoint;
  bool pinned = false; // Precolored — must keep original position

  int64_t length() const { return lastUsePoint - defPoint; }
};

static void collectOps(Block &block,
                       llvm::SmallVectorImpl<Operation *> &ops) {
  for (Operation &op : block) {
    ops.push_back(&op);
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        collectOps(nested, ops);
  }
}

/// Record a PVRegType occurrence. Merges into existing entry if the base
/// index matches, taking max size/alignment and extending the time range.
static void recordPVReg(int64_t baseIdx, int64_t size, int64_t point,
                        bool isDef,
                        llvm::DenseMap<int64_t, int64_t> &defPoints,
                        llvm::DenseMap<int64_t, int64_t> &usePoints,
                        llvm::DenseMap<int64_t, int64_t> &sizes,
                        llvm::DenseMap<int64_t, int64_t> &alignments) {
  int64_t align = (size >= 4) ? 4 : (size >= 2 ? 2 : 1);

  auto it = sizes.find(baseIdx);
  if (it != sizes.end()) {
    it->second = std::max(it->second, size);
    alignments[baseIdx] = std::max(alignments[baseIdx], align);
    if (isDef)
      defPoints[baseIdx] = std::min(defPoints[baseIdx], point);
    else {
      auto uit = usePoints.find(baseIdx);
      if (uit != usePoints.end())
        uit->second = std::max(uit->second, point);
      else
        usePoints[baseIdx] = point;
    }
  } else {
    sizes[baseIdx] = size;
    alignments[baseIdx] = align;
    if (isDef)
      defPoints[baseIdx] = point;
    else
      usePoints[baseIdx] = point;
  }
}

/// For a PVRegType with index X and size S, find the "allocation base":
/// the base index of the multi-register range that contains X.
/// E.g., if the allocator assigned v[92:95] (base=92, size=4), then
/// v93 (index=93, size=1) has allocation base 92.
/// Returns (allocBase, allocSize) or (idx, size) if no containing range.
static std::pair<int64_t, int64_t>
findAllocBase(int64_t idx, int64_t size,
              const llvm::DenseMap<int64_t, int64_t> &knownSizes) {
  // Check if this index IS a known base
  auto it = knownSizes.find(idx);
  if (it != knownSizes.end() && it->second >= size)
    return {idx, it->second};

  // Check if this index falls within a known larger range.
  // Multi-reg ranges are always aligned, so check aligned bases below idx.
  for (int64_t align : {4, 2}) {
    int64_t base = (idx / align) * align;
    if (base == idx)
      continue;
    auto baseIt = knownSizes.find(base);
    if (baseIt != knownSizes.end() && base + baseIt->second > idx)
      return {base, baseIt->second};
  }

  return {idx, size};
}

static void buildPhysRanges(
    ProgramOp program,
    llvm::SmallVectorImpl<PhysVGPRRange> &ranges,
    llvm::DenseMap<int64_t, int64_t> &rangeBaseToIdx) {

  llvm::SmallVector<Operation *> ops;
  collectOps(program.getBodyBlock(), ops);

  llvm::DenseMap<Operation *, int64_t> opToIdx;
  for (int64_t i = 0; i < static_cast<int64_t>(ops.size()); ++i)
    opToIdx[ops[i]] = i;

  llvm::DenseMap<int64_t, int64_t> defPoints, usePoints, sizes, alignments;

  // Track precolored VGPR indices — these must not be remapped.
  llvm::DenseSet<int64_t> pinnedIndices;

  // First pass: collect all multi-register definitions to build the
  // "known allocations" map. This lets us identify sub-element accesses.
  // Also identify precolored VGPRs.
  for (int64_t i = 0; i < static_cast<int64_t>(ops.size()); ++i) {
    Operation *op = ops[i];

    if (isa<PrecoloredVRegOp>(op)) {
      for (Value result : op->getResults()) {
        if (auto pvreg = dyn_cast<PVRegType>(result.getType())) {
          pinnedIndices.insert(pvreg.getIndex());
          for (int64_t k = 0; k < pvreg.getSize(); ++k)
            pinnedIndices.insert(pvreg.getIndex() + k);
        }
      }
    }

    for (Value result : op->getResults()) {
      if (auto pvreg = dyn_cast<PVRegType>(result.getType())) {
        if (pvreg.getSize() > 1) {
          auto &sz = sizes[pvreg.getIndex()];
          sz = std::max(sz, pvreg.getSize());
          int64_t align = (pvreg.getSize() >= 4) ? 4 : 2;
          auto &al = alignments[pvreg.getIndex()];
          al = std::max(al, align);
        }
      }
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (auto pvreg = dyn_cast<PVRegType>(arg.getType())) {
            if (pvreg.getSize() > 1) {
              auto &sz = sizes[pvreg.getIndex()];
              sz = std::max(sz, pvreg.getSize());
              int64_t align = (pvreg.getSize() >= 4) ? 4 : 2;
              auto &al = alignments[pvreg.getIndex()];
              al = std::max(al, align);
            }
          }
        }
      }
    }
  }

  // Second pass: collect def/use points, mapping sub-elements to their base.
  llvm::DenseMap<int64_t, int64_t> knownSizes = sizes;
  defPoints.clear();
  usePoints.clear();
  sizes.clear();
  alignments.clear();

  auto processPVReg = [&](int64_t idx, int64_t size, int64_t point,
                          bool isDef) {
    auto [base, allocSize] = findAllocBase(idx, size, knownSizes);
    recordPVReg(base, allocSize, point, isDef, defPoints, usePoints, sizes,
                alignments);
  };

  for (int64_t i = 0; i < static_cast<int64_t>(ops.size()); ++i) {
    Operation *op = ops[i];

    for (Value result : op->getResults()) {
      if (auto pvreg = dyn_cast<PVRegType>(result.getType()))
        processPVReg(pvreg.getIndex(), pvreg.getSize(), i, true);
    }

    for (Value operand : op->getOperands()) {
      if (auto pvreg = dyn_cast<PVRegType>(operand.getType()))
        processPVReg(pvreg.getIndex(), pvreg.getSize(), i, false);
    }

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (auto pvreg = dyn_cast<PVRegType>(arg.getType()))
            processPVReg(pvreg.getIndex(), pvreg.getSize(), i, true);
        }
      }
    }

    if (isa<LoopOp>(op)) {
      auto loopOp = cast<LoopOp>(op);
      Block &body = loopOp.getBodyBlock();
      Operation *terminator = body.getTerminator();
      int64_t termIdx = terminator ? opToIdx.lookup(terminator) : i;
      for (BlockArgument arg : body.getArguments()) {
        if (auto pvreg = dyn_cast<PVRegType>(arg.getType()))
          processPVReg(pvreg.getIndex(), pvreg.getSize(), termIdx, false);
      }
    }
  }

  // Extend live ranges for values used inside loop bodies.
  // Any value used inside a loop at some program point P is actually
  // live from its definition until the end of the loop body (not just P),
  // because the loop re-executes and the value must be available on
  // every iteration.
  for (int64_t i = 0; i < static_cast<int64_t>(ops.size()); ++i) {
    Operation *op = ops[i];
    if (!isa<LoopOp>(op))
      continue;
    auto loopOp = cast<LoopOp>(op);
    Block &body = loopOp.getBodyBlock();
    Operation *terminator = body.getTerminator();
    int64_t loopStart = i;
    int64_t loopEnd = terminator ? opToIdx.lookup(terminator) : i;

    // For each operand used inside the loop body, extend its range
    // to cover the entire loop.
    body.walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        if (auto pvreg = dyn_cast<PVRegType>(operand.getType())) {
          auto [base, allocSize] = findAllocBase(pvreg.getIndex(),
                                                 pvreg.getSize(), knownSizes);
          auto defIt = defPoints.find(base);
          if (defIt != defPoints.end() && defIt->second < loopStart) {
            // Value defined before the loop but used inside — extend to
            // end of loop body.
            auto &usePt = usePoints[base];
            usePt = std::max(usePt, loopEnd);
          }
        }
      }
    });
  }

  for (const auto &[base, defPt] : defPoints) {
    auto useIt = usePoints.find(base);
    int64_t usePt = (useIt != usePoints.end()) ? useIt->second : defPt;
    PhysVGPRRange r;
    r.physIdx = base;
    r.size = sizes.lookup(base);
    r.alignment = alignments.lookup(base);
    if (r.size == 0)
      r.size = 1;
    if (r.alignment == 0)
      r.alignment = 1;
    r.defPoint = defPt;
    r.lastUsePoint = usePt;
    r.pinned = pinnedIndices.contains(base);
    rangeBaseToIdx[base] = ranges.size();
    ranges.push_back(r);
  }
}

static bool overlaps(const PhysVGPRRange &a, const PhysVGPRRange &b) {
  return a.defPoint <= b.lastUsePoint && b.defPoint <= a.lastUsePoint;
}

static llvm::DenseMap<int64_t, int64_t>
computeCompaction(llvm::SmallVectorImpl<PhysVGPRRange> &ranges,
                  int64_t maxRegs) {
  llvm::SmallVector<int64_t> order(ranges.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](int64_t a, int64_t b) {
    if (ranges[a].length() != ranges[b].length())
      return ranges[a].length() < ranges[b].length();
    return ranges[a].defPoint < ranges[b].defPoint;
  });

  llvm::DenseMap<int64_t, int64_t> oldToNew;
  llvm::SmallVector<int64_t> newAssignment(ranges.size(), -1);

  // Pin precolored ranges to their original positions first.
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (ranges[i].pinned) {
      newAssignment[i] = ranges[i].physIdx;
      oldToNew[ranges[i].physIdx] = ranges[i].physIdx;
    }
  }

  for (int64_t orderIdx : order) {
    if (ranges[orderIdx].pinned)
      continue;

    PhysVGPRRange &r = ranges[orderIdx];
    int64_t sz = r.size;
    int64_t align = r.alignment;

    // v15 is the scratch VGPR for literal materialization in the assembly
    // emitter (AssemblyEmitter.h KernelGenerator::kScratchVGPR). Must stay
    // excluded so compaction never places a live value there.
    constexpr int64_t kScratchVGPR = 15;
    llvm::BitVector occupied(maxRegs, false);
    if (kScratchVGPR < maxRegs)
      occupied.set(kScratchVGPR);
    for (size_t j = 0; j < ranges.size(); ++j) {
      if (newAssignment[j] < 0)
        continue;
      if (overlaps(r, ranges[j])) {
        int64_t base = newAssignment[j];
        for (int64_t k = 0; k < ranges[j].size; ++k)
          if (base + k < maxRegs)
            occupied.set(base + k);
      }
    }

    int64_t chosen = -1;
    for (int64_t c = 0; c + sz <= maxRegs; c += align) {
      bool free = true;
      for (int64_t k = 0; k < sz; ++k) {
        if (occupied.test(c + k)) {
          free = false;
          break;
        }
      }
      if (free) {
        chosen = c;
        break;
      }
    }

    if (chosen >= 0) {
      newAssignment[orderIdx] = chosen;
      oldToNew[r.physIdx] = chosen;
    } else {
      newAssignment[orderIdx] = r.physIdx;
      oldToNew[r.physIdx] = r.physIdx;
    }
  }

  return oldToNew;
}

/// Remap a PVRegType index using the base-to-base mapping.
/// For sub-elements (e.g., v93 within v[92:95]), computes the offset
/// from the old base and applies it to the new base.
static int64_t remapIndex(int64_t oldIdx, int64_t size,
                          const llvm::DenseMap<int64_t, int64_t> &oldToNew,
                          const llvm::SmallVectorImpl<PhysVGPRRange> &ranges) {
  // Direct lookup: this index IS a base
  auto it = oldToNew.find(oldIdx);
  if (it != oldToNew.end())
    return it->second;

  // Sub-element: find the containing base range
  for (const auto &r : ranges) {
    if (oldIdx >= r.physIdx && oldIdx < r.physIdx + r.size) {
      auto baseIt = oldToNew.find(r.physIdx);
      if (baseIt != oldToNew.end())
        return baseIt->second + (oldIdx - r.physIdx);
    }
  }

  return oldIdx;
}

static void applyRemapping(ProgramOp program,
                           const llvm::DenseMap<int64_t, int64_t> &oldToNew,
                           const llvm::SmallVectorImpl<PhysVGPRRange> &ranges) {
  auto remap = [&](Type ty) -> Type {
    auto pvreg = dyn_cast<PVRegType>(ty);
    if (!pvreg)
      return ty;
    int64_t newIdx =
        remapIndex(pvreg.getIndex(), pvreg.getSize(), oldToNew, ranges);
    if (newIdx == pvreg.getIndex())
      return ty;
    return PVRegType::get(ty.getContext(), newIdx, pvreg.getSize());
  };

  program.walk([&](Operation *op) {
    if (isa<ProgramOp>(op))
      return;

    for (Value result : op->getResults()) {
      Type newTy = remap(result.getType());
      if (newTy != result.getType())
        result.setType(newTy);
    }

    if (auto condOp = dyn_cast<ConditionOp>(op)) {
      if (auto attr = condOp->getAttrOfType<DenseI64ArrayAttr>(
              "_iterArgPhysRegs")) {
        auto vals = attr.asArrayRef();
        llvm::SmallVector<int64_t> newVals(vals.begin(), vals.end());
        bool anyChanged = false;
        for (size_t i = 0; i < newVals.size(); ++i) {
          if (newVals[i] < 0)
            continue;
          if (i < condOp.getIterArgs().size()) {
            Type ty = condOp.getIterArgs()[i].getType();
            if (isa<PVRegType>(ty)) {
              int64_t newIdx =
                  remapIndex(newVals[i], 1, oldToNew, ranges);
              if (newIdx != newVals[i]) {
                newVals[i] = newIdx;
                anyChanged = true;
              }
            }
          }
        }
        if (anyChanged) {
          condOp->setAttr("_iterArgPhysRegs",
                          DenseI64ArrayAttr::get(op->getContext(), newVals));
        }
      }
    }
  });

  program.walk([&](LoopOp loopOp) {
    Block &body = loopOp.getBodyBlock();
    for (BlockArgument arg : body.getArguments()) {
      Type newTy = remap(arg.getType());
      if (newTy != arg.getType())
        arg.setType(newTy);
    }
    for (Value result : loopOp->getResults()) {
      Type newTy = remap(result.getType());
      if (newTy != result.getType())
        result.setType(newTy);
    }
  });
}

struct WAVEASMVGPRCompaction
    : waveasm::impl::WAVEASMVGPRCompactionBase<WAVEASMVGPRCompaction> {

  void runOnOperation() override {
    auto moduleOp = getOperation();

    moduleOp->walk([&](ProgramOp program) {
      llvm::SmallVector<PhysVGPRRange> ranges;
      llvm::DenseMap<int64_t, int64_t> rangeBaseToIdx;

      buildPhysRanges(program, ranges, rangeBaseToIdx);

      if (ranges.empty())
        return;

      int64_t maxBefore = 0;
      for (const auto &r : ranges)
        maxBefore = std::max(maxBefore, r.physIdx + r.size);

      auto oldToNew = computeCompaction(ranges, /*maxRegs=*/512);

      int64_t maxAfter = 0;
      for (const auto &r : ranges) {
        auto it = oldToNew.find(r.physIdx);
        int64_t newIdx = (it != oldToNew.end()) ? it->second : r.physIdx;
        maxAfter = std::max(maxAfter, newIdx + r.size);
      }

      bool anyChange = false;
      for (const auto &[old, newIdx] : oldToNew) {
        if (old != newIdx) {
          anyChange = true;
          break;
        }
      }

      if (!anyChange)
        return;

      LLVM_DEBUG(llvm::dbgs() << "VGPR compaction: " << maxBefore << " -> "
                              << maxAfter << " (saved "
                              << (maxBefore - maxAfter) << ")\n");

      applyRemapping(program, oldToNew, ranges);
    });
  }
};

} // namespace
