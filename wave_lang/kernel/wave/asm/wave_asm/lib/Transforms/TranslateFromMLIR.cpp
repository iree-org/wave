// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Transforms/TranslateFromMLIR.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <cctype>
#include <sstream>

#define DEBUG_TYPE "waveasm-translate"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// TranslationContext Implementation
//===----------------------------------------------------------------------===//

TranslationContext::TranslationContext(OpBuilder &builder, ProgramOp program,
                                       TargetAttrInterface target)
    : builder(builder), program(program), target(target) {}

VRegType TranslationContext::createVRegType(int64_t size, int64_t alignment) {
  return VRegType::get(builder.getContext(), size, alignment);
}

SRegType TranslationContext::createSRegType(int64_t size, int64_t alignment) {
  return SRegType::get(builder.getContext(), size, alignment);
}

ImmType TranslationContext::createImmType(int64_t value) {
  return ImmType::get(builder.getContext(), value);
}

LabelOp TranslationContext::emitLabel(StringRef name) {
  return LabelOp::create(builder, builder.getUnknownLoc(), name);
}

CommentOp TranslationContext::emitComment(StringRef text) {
  return CommentOp::create(builder, builder.getUnknownLoc(), text);
}

RawOp TranslationContext::emitRaw(StringRef code) {
  return RawOp::create(builder, builder.getUnknownLoc(), code);
}

Block::iterator TranslationContext::getInsertionPoint() {
  return builder.getInsertionPoint();
}

std::optional<Value> TranslationContext::getCachedExpr(
    StringRef opName, ValueRange operands, ArrayRef<int64_t> constants) const {
  // Build a cache key string
  std::string key;
  llvm::raw_string_ostream os(key);
  os << opName;
  for (Value v : operands) {
    os << "_" << v.getAsOpaquePointer();
  }
  for (int64_t c : constants) {
    os << "_c" << c;
  }

  auto it = exprCache.find(os.str());
  if (it != exprCache.end())
    return it->second;
  return std::nullopt;
}

void TranslationContext::cacheExpr(StringRef opName, ValueRange operands,
                                    ArrayRef<int64_t> constants, Value result) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << opName;
  for (Value v : operands) {
    os << "_" << v.getAsOpaquePointer();
  }
  for (int64_t c : constants) {
    os << "_c" << c;
  }
  exprCache[os.str()] = result;
}

std::string TranslationContext::generateLabel(StringRef prefix) {
  return (prefix + "_" + std::to_string(labelCounter++)).str();
}

void TranslationContext::queueSRDSetup(Value memref, int64_t argIndex,
                                        int64_t bufferSize) {
  // Check if this memref already has an SRD assigned (avoid duplicates)
  if (srdIndexMap.count(memref) > 0) {
    // Already processed - just update size if larger
    updateSRDBufferSize(memref, bufferSize);
    return;
  }

  // Allocate an SRD slot (4 consecutive SGPRs)
  int64_t srdBase = getNextSRDIndex();

  PendingSRD pending;
  pending.memref = memref;
  pending.argIndex = argIndex;
  pending.bufferSize = bufferSize;
  pending.srdBaseIndex = srdBase;

  pendingSRDs.push_back(pending);
  setSRDIndex(memref, srdBase);
}

void TranslationContext::emitSRDPrologue() {
  if (srdPrologueEmitted || pendingSRDs.empty())
    return;

  srdPrologueEmitted = true;
  auto loc = builder.getUnknownLoc();

  // Check if this is a gfx95* target (requires preload pattern with branch+alignment)
  bool isGFX95 = llvm::isa<GFX950TargetAttr>(target);

  // Recompute SRD base indices now that we know the total number of args
  // SRDs must start after: user SGPRs + system SGPRs (workgroup IDs)
  int64_t userSgprCount = 2;  // kernarg ptr
  if (isGFX95) {
    userSgprCount += pendingSRDs.size() * 2;  // preloaded args
  }
  int64_t systemSgprCount = 3;  // workgroup_id_x, y, z
  int64_t srdStartIndex = (userSgprCount + systemSgprCount + 3) & ~3;  // Align to 4

  // Update pending SRDs with correct indices and update srdIndexMap
  for (size_t i = 0; i < pendingSRDs.size(); ++i) {
    int64_t newSrdBase = srdStartIndex + i * 4;
    pendingSRDs[i].srdBaseIndex = newSrdBase;
    srdIndexMap[pendingSRDs[i].memref] = newSrdBase;
  }

  // Emit comment for prologue
  CommentOp::create(builder, loc, "SRD setup prologue");

  if (isGFX95) {
    // GFX95* path: Use preload pattern with intermediate locations and s_mov_b64 copies
    // This matches the Python backend behavior for gfx950.
    //
    // Step 1: Load base addresses into preload locations s[2:3], s[4:5], etc.
    for (const auto &pending : pendingSRDs) {
      int64_t loadBase = 2 + pending.argIndex * 2;
      int64_t kernargOffset = pending.argIndex * 8;

      std::string loadStr = "s_load_dwordx2 s[" + std::to_string(loadBase) + ":" +
                            std::to_string(loadBase + 1) + "], s[0:1], 0x" +
                            llvm::utohexstr(kernargOffset);
      RawOp::create(builder, loc, loadStr);
    }

    // Step 2: Wait for all scalar loads to complete
    auto i32Type = builder.getI32Type();
    auto lgkmcntAttr = IntegerAttr::get(i32Type, 0);
    S_WAITCNT::create(builder, loc, /*vmcnt=*/IntegerAttr{}, lgkmcntAttr, /*expcnt=*/IntegerAttr{});

    // Step 2.5: Branch to aligned entry point (gfx95* requirement)
    std::string kernelName = program.getSymName().str();
    std::string mainLabel = ".L_" + kernelName + "_main";

    RawOp::create(builder, loc, "s_branch " + mainLabel);
    RawOp::create(builder, loc, ".p2align 8");
    RawOp::create(builder, loc, mainLabel + ":");

    // Step 3: Copy from preload locations to SRD positions and fill size/stride
    for (size_t i = 0; i < pendingSRDs.size(); ++i) {
      const auto &pending = pendingSRDs[i];
      int64_t srdBase = pending.srdBaseIndex;
      int64_t preloadBase = 2 + pending.argIndex * 2;

      auto srdType = createSRegType(4, 4);
      auto srdReg = PrecoloredSRegOp::create(builder, loc, srdType, srdBase, 4);

      // Copy base address with s_mov_b64
      std::string movB64Str = "s_mov_b64 s[" + std::to_string(srdBase) + ":" +
                              std::to_string(srdBase + 1) + "], s[" +
                              std::to_string(preloadBase) + ":" +
                              std::to_string(preloadBase + 1) + "]";
      RawOp::create(builder, loc, movB64Str);

      // Fill size and stride
      std::string movSizeStr = "s_mov_b32 s" + std::to_string(srdBase + 2) + ", 0x" +
                               llvm::utohexstr(pending.bufferSize);
      RawOp::create(builder, loc, movSizeStr);

      std::string movStrideStr = "s_mov_b32 s" + std::to_string(srdBase + 3) + ", 0x20000";
      RawOp::create(builder, loc, movStrideStr);

      mapper.mapValue(pending.memref, srdReg);
    }
  } else {
    // Non-GFX95* path (e.g., gfx942): Load directly into SRD positions
    // This eliminates the s_mov_b64 copies by loading args directly into the
    // SRD base addresses (SRD[0:1]), then only filling size/stride with s_mov_b32.
    //
    // Step 1: Load base addresses directly into SRD[0:1] positions
    for (const auto &pending : pendingSRDs) {
      int64_t srdBase = pending.srdBaseIndex;
      int64_t kernargOffset = pending.argIndex * 8;

      // Load directly into SRD base: s[srdBase:srdBase+1]
      std::string loadStr = "s_load_dwordx2 s[" + std::to_string(srdBase) + ":" +
                            std::to_string(srdBase + 1) + "], s[0:1], 0x" +
                            llvm::utohexstr(kernargOffset);
      RawOp::create(builder, loc, loadStr);
    }

    // Step 2: Wait for all scalar loads to complete
    auto i32Type = builder.getI32Type();
    auto lgkmcntAttr = IntegerAttr::get(i32Type, 0);
    S_WAITCNT::create(builder, loc, /*vmcnt=*/IntegerAttr{}, lgkmcntAttr, /*expcnt=*/IntegerAttr{});

    // Step 3: Fill SRD[2:3] with size and stride (no s_mov_b64 copies needed!)
    for (size_t i = 0; i < pendingSRDs.size(); ++i) {
      const auto &pending = pendingSRDs[i];
      int64_t srdBase = pending.srdBaseIndex;

      auto srdType = createSRegType(4, 4);
      auto srdReg = PrecoloredSRegOp::create(builder, loc, srdType, srdBase, 4);

      // Fill size
      std::string movSizeStr = "s_mov_b32 s" + std::to_string(srdBase + 2) + ", 0x" +
                               llvm::utohexstr(pending.bufferSize);
      RawOp::create(builder, loc, movSizeStr);

      // Fill stride descriptor
      std::string movStrideStr = "s_mov_b32 s" + std::to_string(srdBase + 3) + ", 0x20000";
      RawOp::create(builder, loc, movStrideStr);

      mapper.mapValue(pending.memref, srdReg);
    }
  }

  CommentOp::create(builder, loc, "End SRD setup");
}

std::optional<int64_t> TranslationContext::getSRDIndex(Value memref) const {
  auto it = srdIndexMap.find(memref);
  if (it != srdIndexMap.end())
    return it->second;
  return std::nullopt;
}

void TranslationContext::setSRDIndex(Value memref, int64_t srdBaseIndex) {
  srdIndexMap[memref] = srdBaseIndex;
}

int64_t TranslationContext::getNextSRDIndex() {
  // Note: This is called during queueSRDSetup, but we don't know the
  // total number of kernel args yet. We return a placeholder that will
  // be recomputed in emitSRDPrologue.
  // For now, just use a simple incrementing scheme starting from 0.
  // The actual SGPR indices will be computed in emitSRDPrologue.
  int64_t idx = nextSRDIndex < 0 ? 0 : nextSRDIndex;
  nextSRDIndex = idx + 4;  // Each SRD uses 4 consecutive SGPRs
  return idx;
}

void TranslationContext::updateSRDBufferSize(Value memref, int64_t bufferSize) {
  // Find the pending SRD for this memref and update its buffer size
  for (auto &pending : pendingSRDs) {
    if (pending.memref == memref) {
      // Only update if the new size is larger (more specific)
      if (bufferSize > pending.bufferSize) {
        pending.bufferSize = bufferSize;
      }
      return;
    }
  }
}

} // namespace waveasm

using namespace waveasm;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Compute buffer size from memref type
static int64_t computeBufferSizeFromMemRef(MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic)
      dim = 1;  // Conservative estimate for dynamic dims
    numElements *= dim;
  }
  int64_t elementBytes = memrefType.getElementTypeBitWidth() / 8;
  if (elementBytes == 0)
    elementBytes = 1;  // Minimum 1 byte
  return numElements * elementBytes;
}

/// Get or create a mapped VGPR value for an MLIR value
[[maybe_unused]]
static Value getOrCreateVReg(Value mlirValue, TranslationContext &ctx, int64_t size = 1) {
  auto &mapper = ctx.getMapper();

  // Check if already mapped
  if (auto mapped = mapper.getMapped(mlirValue))
    return *mapped;

  // Create a new VGPR - this will be a placeholder that gets resolved
  // In a real implementation, we'd need more sophisticated tracking
  [[maybe_unused]] auto vregType = ctx.createVRegType(size, size > 1 ? size : 1);
  auto loc = mlirValue.getLoc();

  // For block arguments (like function parameters), create a precolored reg
  if (auto blockArg = dyn_cast<BlockArgument>(mlirValue)) {
    // Function arguments are passed via SGPRs (buffer descriptors)
    // This is a simplification - real ABI handling is more complex
    auto sregType = ctx.createSRegType(2, 2);  // 64-bit pointer
    auto sreg = PrecoloredSRegOp::create(ctx.getBuilder(),
        loc, sregType, blockArg.getArgNumber() * 2, 2);
    mapper.mapValue(mlirValue, sreg);
    return sreg;
  }

  return mlirValue;  // Return as-is for now, handler will create proper op
}

/// Check if a memref has LDS address space
bool isLDSMemRef(MemRefType memrefType) {
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace)
    return false;

  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
    return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  return false;
}

/// Get element size in bytes
int64_t getElementBytes(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 4;
}

/// Get vector size in bytes
int64_t getVectorBytes(VectorType vecType) {
  return vecType.getNumElements() * getElementBytes(vecType.getElementType());
}

/// Check if value is power of 2
bool isPowerOf2(int64_t val) {
  return val > 0 && (val & (val - 1)) == 0;
}

/// Get log2 of power of 2
int64_t log2(int64_t val) {
  int64_t result = 0;
  while (val > 1) {
    val >>= 1;
    result++;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// GPU Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle gpu.thread_id - emit v_mbcnt for single-wave or use v_bfe_u32 for multi-wave
LogicalResult handleGPUThreadId(Operation *op, TranslationContext &ctx) {
  auto threadIdOp = cast<gpu::ThreadIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  Value result;

  // Get the dimension being requested
  gpu::Dimension dim = threadIdOp.getDimension();

  // Check if this is a multi-wave kernel
  // For multi-wave, the hardware provides flat_workitem_id in v0
  // For single-wave, we compute lane_id using v_mbcnt
  if (ctx.isMultiWaveKernel()) {
    // Multi-wave: hardware provides flat workitem ID in v0
    // The flat workitem ID is packed as: tid_x + tid_y * wg_x + tid_z * wg_x * wg_y
    // We use v_bfe_u32 to extract the individual components
    // Following Python's approach: 10 bits per dimension (supports up to 1024 per dim)

    // Mark that this kernel uses workitem ID (set amdhsa_system_vgpr_workitem_id)
    ctx.setUsesWorkitemId(true);

    // Get flat workitem ID from v0
    auto flatWorkitemId = PrecoloredVRegOp::create(builder, loc, vregType, 0, 1);

    // Determine bit offset and width based on dimension
    // Python uses 10 bits per dimension:
    // - tid_x: bits 0-9 (offset=0, width=10)
    // - tid_y: bits 10-19 (offset=10, width=10)
    // - tid_z: bits 20-29 (offset=20, width=10)
    int64_t bitOffset = 0;
    int64_t bitWidth = 10;

    switch (dim) {
      case gpu::Dimension::x:
        bitOffset = 0;
        break;
      case gpu::Dimension::y:
        bitOffset = 10;
        break;
      case gpu::Dimension::z:
        bitOffset = 20;
        break;
    }

    // Create constants for offset and width
    auto immOffset = ctx.createImmType(bitOffset);
    auto offsetConst = ConstantOp::create(builder, loc, immOffset, bitOffset);

    auto immWidth = ctx.createImmType(bitWidth);
    auto widthConst = ConstantOp::create(builder, loc, immWidth, bitWidth);

    // v_bfe_u32 dst, src, offset, width - extract bits [offset, offset+width-1]
    result = V_BFE_U32::create(builder, loc, vregType, flatWorkitemId, offsetConst, widthConst);
  } else {
    // Single-wave: compute lane ID using v_mbcnt
    // Note: for single-wave, tid_x == lane_id, and tid_y/tid_z are always 0

    if (dim == gpu::Dimension::x) {
      // Create constant -1 for the mask
      auto immType = ctx.createImmType(-1);
      auto maskConst = ConstantOp::create(builder, loc, immType, -1);

      // Create constant 0
      auto immZero = ctx.createImmType(0);
      auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);

      // v_mbcnt_lo_u32_b32 - count bits in lower 32 lanes
      auto mbcntLo = V_MBCNT_LO_U32_B32::create(builder, loc, vregType, maskConst, zeroConst);

      // v_mbcnt_hi_u32_b32 - count bits in upper 32 lanes, add to low
      auto mbcntHi = V_MBCNT_HI_U32_B32::create(builder, loc, vregType, maskConst, mbcntLo);

      result = mbcntHi;
    } else {
      // For single-wave, tid_y and tid_z are always 0
      auto immZero = ctx.createImmType(0);
      result = ConstantOp::create(builder, loc, immZero, 0);
    }

    // Note: for single-wave, we don't set usesWorkitemId since we use v_mbcnt
    // The kernel descriptor should have amdhsa_system_vgpr_workitem_id = 0
  }

  // Map the result
  ctx.getMapper().mapValue(threadIdOp.getResult(), result);

  // Track upper bound for affine simplification
  // The upper_bound attribute tells us the range of this thread ID
  if (auto upperBoundAttr = threadIdOp.getUpperBoundAttr()) {
    int64_t upperBound = upperBoundAttr.getInt();
    // Track this thread ID value with its upper bound
    ctx.setThreadIdUpperBound(threadIdOp.getResult(), upperBound);

    // Also set bit range for OR vs ADD optimization
    // Thread ID is in range [0, upperBound-1], so we need bits 0..log2(upperBound-1)
    ctx.setBitRange(result, BitRange::fromMaxValue(upperBound - 1));
  } else {
    // Default for 64-lane wavefront: [0, 63] needs 6 bits
    ctx.setBitRange(result, BitRange(0, 5));
  }

  return success();
}

/// Handle gpu.block_id - block ID comes from system SGPRs
LogicalResult handleGPUBlockId(Operation *op, TranslationContext &ctx) {
  auto blockIdOp = cast<gpu::BlockIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Block IDs are passed in system SGPRs via ABI
  // System SGPRs come after user SGPRs (kernarg ptr + preloaded args)
  gpu::Dimension dim = blockIdOp.getDimension();
  int dimIndex = 0;
  switch (dim) {
    case gpu::Dimension::x:
      dimIndex = 0;
      ctx.setUsesWorkgroupIdX(true);
      break;
    case gpu::Dimension::y:
      dimIndex = 1;
      ctx.setUsesWorkgroupIdY(true);
      break;
    case gpu::Dimension::z:
      dimIndex = 2;
      ctx.setUsesWorkgroupIdZ(true);
      break;
  }

  int64_t sgprIndex = ctx.getWorkgroupIdSgprIndex(dimIndex);

  auto sregType = ctx.createSRegType();
  auto blockId = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(blockIdOp.getResult(), blockId);
  return success();
}

/// Handle gpu.barrier - emit s_barrier
LogicalResult handleGPUBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  S_BARRIER::create(builder, loc);
  return success();
}

/// Handle gpu.block_dim - workgroup dimensions from kernel dispatch packet
LogicalResult handleGPUBlockDim(Operation *op, TranslationContext &ctx) {
  auto blockDimOp = cast<gpu::BlockDimOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Block dimensions are available from the dispatch packet or as kernel args
  // For now, use precolored SGPRs at standard ABI locations
  int64_t sgprIndex = 0;
  switch (blockDimOp.getDimension()) {
    case gpu::Dimension::x: sgprIndex = 6; break;   // workgroup_size_x
    case gpu::Dimension::y: sgprIndex = 7; break;   // workgroup_size_y
    case gpu::Dimension::z: sgprIndex = 8; break;   // workgroup_size_z
  }

  auto sregType = ctx.createSRegType();
  auto blockDim = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(blockDimOp.getResult(), blockDim);
  return success();
}

/// Handle gpu.grid_dim - grid dimensions from kernel dispatch packet
LogicalResult handleGPUGridDim(Operation *op, TranslationContext &ctx) {
  auto gridDimOp = cast<gpu::GridDimOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Grid dimensions come from dispatch packet
  int64_t sgprIndex = 0;
  switch (gridDimOp.getDimension()) {
    case gpu::Dimension::x: sgprIndex = 9; break;   // grid_size_x
    case gpu::Dimension::y: sgprIndex = 10; break;  // grid_size_y
    case gpu::Dimension::z: sgprIndex = 11; break;  // grid_size_z
  }

  auto sregType = ctx.createSRegType();
  auto gridDim = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);

  ctx.getMapper().mapValue(gridDimOp.getResult(), gridDim);
  return success();
}

/// Handle gpu.lane_id - lane within wavefront (same as thread_id for wave64)
LogicalResult handleGPULaneId(Operation *op, TranslationContext &ctx) {
  auto laneIdOp = cast<gpu::LaneIdOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  // Lane ID is the thread's position within the wavefront
  // Use v_mbcnt_lo + v_mbcnt_hi pattern
  auto immType = ctx.createImmType(-1);
  auto maskConst = ConstantOp::create(builder, loc, immType, -1);

  auto immZero = ctx.createImmType(0);
  auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);

  auto mbcntLo = V_MBCNT_LO_U32_B32::create(builder, loc, vregType, maskConst, zeroConst);
  auto mbcntHi = V_MBCNT_HI_U32_B32::create(builder, loc, vregType, maskConst, mbcntLo);

  ctx.getMapper().mapValue(laneIdOp.getResult(), mbcntHi);
  return success();
}

/// Handle gpu.subgroup_broadcast - broadcast value from one lane to all
/// This emits v_readlane_b32 to read from a specific lane, or v_readfirstlane_b32
/// for broadcasting from lane 0.
LogicalResult handleGPUSubgroupBroadcast(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // gpu.subgroup_broadcast can have either:
  // - 2 operands: value and lane_id
  // - 1 operand with lane_id as attribute (newer MLIR versions)
  if (op->getNumOperands() < 1) {
    return op->emitError("subgroup_broadcast requires at least a value operand");
  }

  Value srcValue = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(srcValue);
  if (!srcMapped) {
    return op->emitError("source value not mapped");
  }

  // If the source is an immediate (constant), broadcasting is a no-op
  // Just map the result to the same immediate value
  if (isa<ImmType>(srcMapped->getType())) {
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Check for lane_id - either as operand or attribute
  std::optional<Value> laneMapped;
  int64_t laneIdValue = 0;
  bool hasLaneId = false;

  if (op->getNumOperands() >= 2) {
    // Lane ID is operand 1
    Value laneIdx = op->getOperand(1);
    laneMapped = ctx.getMapper().getMapped(laneIdx);
    hasLaneId = true;
  } else if (auto laneAttr = op->getAttrOfType<IntegerAttr>("lane_id")) {
    // Lane ID is an attribute (newer MLIR)
    laneIdValue = laneAttr.getInt();
    hasLaneId = true;
  } else if (auto subgroupIdAttr = op->getAttrOfType<IntegerAttr>("subgroup_id")) {
    // Alternative attribute name
    laneIdValue = subgroupIdAttr.getInt();
    hasLaneId = true;
  }

  auto sregType = ctx.createSRegType();

  // If lane is constant 0, use v_readfirstlane_b32 (simpler, no lane arg)
  bool usesFirstLane = false;
  if (laneMapped) {
    if (auto constOp = laneMapped->getDefiningOp<ConstantOp>()) {
      if (constOp.getValue() == 0) {
        usesFirstLane = true;
      }
    }
  } else if (hasLaneId && laneIdValue == 0) {
    usesFirstLane = true;
  }

  Value result;
  if (usesFirstLane) {
    result = V_READFIRSTLANE_B32::create(builder, loc, sregType, *srcMapped);
  } else {
    // Use v_readlane_b32 with lane index
    if (!laneMapped) {
      auto immType = ctx.createImmType(laneIdValue);
      laneMapped = ConstantOp::create(builder, loc, immType, laneIdValue);
    }
    result = V_READLANE_B32::create(builder, loc, sregType, *srcMapped, *laneMapped);
  }

  ctx.getMapper().mapValue(op->getResult(0), result);
  return success();
}

//===----------------------------------------------------------------------===//
// Arith Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle arith.constant - create immediate or track value
LogicalResult handleArithConstant(Operation *op, TranslationContext &ctx) {
  auto constOp = cast<arith::ConstantOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto value = constOp.getValue();

  // Handle integer constants
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    int64_t intVal = intAttr.getInt();
    auto immType = ctx.createImmType(intVal);
    auto immOp = ConstantOp::create(builder, loc, immType, intVal);
    ctx.getMapper().mapValue(constOp.getResult(), immOp);
    return success();
  }

  // Handle float constants - need to bitcast to int
  if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
    double floatVal = floatAttr.getValueAsDouble();
    // For f32, bitcast to i32
    if (floatAttr.getType().isF32()) {
      float f = static_cast<float>(floatVal);
      int32_t bits;
      memcpy(&bits, &f, sizeof(bits));
      auto immType = ctx.createImmType(bits);
      auto immOp = ConstantOp::create(builder, loc, immType, bits);
      ctx.getMapper().mapValue(constOp.getResult(), immOp);
    }
    return success();
  }

  // Handle dense vector constants (e.g., dense<0.0> for accumulator init)
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
    if (denseAttr.isSplat()) {
      auto splatVal = denseAttr.getSplatValue<Attribute>();
      if (auto floatAttr = dyn_cast<FloatAttr>(splatVal)) {
        double floatVal = floatAttr.getValueAsDouble();
        if (floatVal == 0.0) {
          // Zero accumulator - use literal 0
          auto immType = ctx.createImmType(0);
          auto immOp = ConstantOp::create(builder, loc, immType, 0);
          ctx.getMapper().mapValue(constOp.getResult(), immOp);
        }
      }
    }
    return success();
  }

  return success();
}

/// Handle arith.addi - emit v_add_u32
LogicalResult handleArithAddI(Operation *op, TranslationContext &ctx) {
  auto addOp = cast<arith::AddIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(addOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(addOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto add = V_ADD_U32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(addOp.getResult(), add);
  return success();
}

/// Handle arith.muli - emit v_mul_lo_u32
LogicalResult handleArithMulI(Operation *op, TranslationContext &ctx) {
  auto mulOp = cast<arith::MulIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(mulOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(mulOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto mul = V_MUL_LO_U32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(mulOp.getResult(), mul);
  return success();
}

/// Handle arith.index_cast - typically a no-op on GPU
LogicalResult handleArithIndexCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<arith::IndexCastOp>(op);

  // Index casts are usually no-ops (same register, different type interpretation)
  auto src = ctx.getMapper().getMapped(castOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

/// Handle arith.subi - emit v_sub_u32
LogicalResult handleArithSubI(Operation *op, TranslationContext &ctx) {
  auto subOp = cast<arith::SubIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(subOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(subOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto sub = V_SUB_U32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(subOp.getResult(), sub);
  return success();
}

/// Handle arith.andi - emit v_and_b32
LogicalResult handleArithAndI(Operation *op, TranslationContext &ctx) {
  auto andOp = cast<arith::AndIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(andOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(andOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto andResult = V_AND_B32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(andOp.getResult(), andResult);
  return success();
}

/// Handle arith.ori - emit v_or_b32
LogicalResult handleArithOrI(Operation *op, TranslationContext &ctx) {
  auto orOp = cast<arith::OrIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(orOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(orOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto orResult = V_OR_B32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(orOp.getResult(), orResult);
  return success();
}

/// Handle arith.xori - emit v_xor_b32
LogicalResult handleArithXorI(Operation *op, TranslationContext &ctx) {
  auto xorOp = cast<arith::XOrIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(xorOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(xorOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto xorResult = V_XOR_B32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(xorOp.getResult(), xorResult);
  return success();
}

/// Handle arith.shli - emit v_lshlrev_b32
LogicalResult handleArithShLI(Operation *op, TranslationContext &ctx) {
  auto shlOp = cast<arith::ShLIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(shlOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(shlOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // v_lshlrev has reversed operands (shift amount first)
  auto shl = V_LSHLREV_B32::create(builder, loc, vregType, *rhs, *lhs);
  ctx.getMapper().mapValue(shlOp.getResult(), shl);
  return success();
}

/// Handle arith.shrui - emit v_lshrrev_b32 (unsigned right shift)
LogicalResult handleArithShRUI(Operation *op, TranslationContext &ctx) {
  auto shrOp = cast<arith::ShRUIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(shrOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(shrOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // v_lshrrev has reversed operands (shift amount first)
  auto shr = V_LSHRREV_B32::create(builder, loc, vregType, *rhs, *lhs);
  ctx.getMapper().mapValue(shrOp.getResult(), shr);
  return success();
}

/// Handle arith.shrsi - emit v_ashrrev_i32 (signed right shift)
LogicalResult handleArithShRSI(Operation *op, TranslationContext &ctx) {
  auto shrOp = cast<arith::ShRSIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(shrOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(shrOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // v_ashrrev has reversed operands (shift amount first)
  auto shr = V_ASHRREV_I32::create(builder, loc, vregType, *rhs, *lhs);
  ctx.getMapper().mapValue(shrOp.getResult(), shr);
  return success();
}

/// Handle arith.extui - emit v_mov (zero extension is implicit)
LogicalResult handleArithExtUI(Operation *op, TranslationContext &ctx) {
  auto extOp = cast<arith::ExtUIOp>(op);

  // Zero extension to 32-bit is usually a no-op on GPU
  auto src = ctx.getMapper().getMapped(extOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(extOp.getResult(), *src);
  }
  return success();
}

/// Handle arith.extsi - emit v_bfe (bit field extract for sign extension)
LogicalResult handleArithExtSI(Operation *op, TranslationContext &ctx) {
  auto extOp = cast<arith::ExtSIOp>(op);

  // Sign extension - for small types, may need bit field extract
  // For now, pass through (assuming 32-bit operations)
  auto src = ctx.getMapper().getMapped(extOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(extOp.getResult(), *src);
  }
  return success();
}

/// Handle arith.trunci - emit v_and (truncation via mask)
LogicalResult handleArithTruncI(Operation *op, TranslationContext &ctx) {
  auto truncOp = cast<arith::TruncIOp>(op);

  // Truncation is usually a no-op (just use lower bits)
  auto src = ctx.getMapper().getMapped(truncOp.getIn());
  if (src) {
    ctx.getMapper().mapValue(truncOp.getResult(), *src);
  }
  return success();
}

/// Handle arith.cmpi - emit v_cmp instructions (sets VCC implicitly)
LogicalResult handleArithCmpI(Operation *op, TranslationContext &ctx) {
  auto cmpOp = cast<arith::CmpIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto lhs = ctx.getMapper().getMapped(cmpOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(cmpOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // Emit comparison based on predicate
  // These operations set VCC implicitly (no SSA result)
  switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      V_CMP_EQ_U32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::ne:
      V_CMP_NE_U32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::slt:
      V_CMP_LT_I32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::sle:
      V_CMP_LE_I32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::sgt:
      V_CMP_GT_I32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::sge:
      V_CMP_GE_I32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::ult:
      V_CMP_LT_U32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::ule:
      V_CMP_LE_U32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::ugt:
      V_CMP_GT_U32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpIPredicate::uge:
      V_CMP_GE_U32::create(builder, loc, *lhs, *rhs);
      break;
  }

  // For now, we don't track VCC as a value - the select handler uses it implicitly
  // Map result to a placeholder (the select handler will use VCC)
  auto immOne = ctx.createImmType(1);
  auto one = ConstantOp::create(builder, loc, immOne, 1);
  ctx.getMapper().mapValue(cmpOp.getResult(), one);

  return success();
}

/// Handle arith.select - emit v_cndmask_b32
LogicalResult handleArithSelect(Operation *op, TranslationContext &ctx) {
  auto selectOp = cast<arith::SelectOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  // Note: condition was handled by cmpi which set VCC
  auto trueVal = ctx.getMapper().getMapped(selectOp.getTrueValue());
  auto falseVal = ctx.getMapper().getMapped(selectOp.getFalseValue());

  if (!trueVal || !falseVal) {
    return op->emitError("operands not mapped");
  }

  // Get a zero constant for the VCC operand (VCC is implicitly used)
  auto immZero = ctx.createImmType(0);
  auto zeroConst = ConstantOp::create(builder, loc, immZero, 0);

  // v_cndmask_b32: select true_val if vcc set, false_val otherwise
  // The third operand is VCC (implicitly used from previous v_cmp)
  auto select = V_CNDMASK_B32::create(builder, loc, vregType, *falseVal, *trueVal, zeroConst);
  ctx.getMapper().mapValue(selectOp.getResult(), select);
  return success();
}

/// Helper to check if an MLIR value is a constant power of 2
static std::optional<int64_t> getConstantPowerOf2(Value val, TranslationContext &ctx) {
  // Check if it's a waveasm.constant
  if (auto constOp = val.getDefiningOp<ConstantOp>()) {
    int64_t value = constOp.getValue();
    if (value > 0 && (value & (value - 1)) == 0) {
      // It's a power of 2, return log2
      int64_t log2val = 0;
      while ((1LL << log2val) < value) {
        log2val++;
      }
      return log2val;
    }
  }
  // Check for arith.constant in the original MLIR
  auto *defOp = val.getDefiningOp();
  if (!defOp) return std::nullopt;

  if (auto arithConst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
      int64_t value = intAttr.getInt();
      if (value > 0 && (value & (value - 1)) == 0) {
        int64_t log2val = 0;
        while ((1LL << log2val) < value) {
          log2val++;
        }
        return log2val;
      }
    }
  }
  return std::nullopt;
}

/// Handle arith.divui - emit v_lshrrev for power-of-2, general div otherwise
LogicalResult handleArithDivUI(Operation *op, TranslationContext &ctx) {
  auto divOp = cast<arith::DivUIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(divOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(divOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // Check if RHS is a constant power of 2
  if (auto log2val = getConstantPowerOf2(divOp.getRhs(), ctx)) {
    // Division by 2^n = right shift by n
    auto shiftAmt = ctx.createImmType(*log2val);
    auto shiftConst = ConstantOp::create(builder, loc, shiftAmt, *log2val);
    auto div = V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, *lhs);
    ctx.getMapper().mapValue(divOp.getResult(), div);
    return success();
  }

  // General case: need complex reciprocal sequence
  // For now, emit a placeholder shift (will need full implementation)
  // TODO: Implement proper integer division using reciprocal approximation
  auto div = V_LSHRREV_B32::create(builder, loc, vregType, *rhs, *lhs);
  ctx.getMapper().mapValue(divOp.getResult(), div);
  return success();
}

/// Helper to get constant value from arith.constant
static std::optional<int64_t> getConstantValue(Value val) {
  auto *defOp = val.getDefiningOp();
  if (!defOp) return std::nullopt;

  // Try typed arith::ConstantOp first
  if (auto arithConst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
      return intAttr.getInt();
    }
  }

  // Handle generic form "arith.constant" with attribute "value"
  if (defOp->getName().getStringRef() == "arith.constant") {
    if (auto valueAttr = defOp->getAttr("value")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
        return intAttr.getInt();
      }
    }
  }

  return std::nullopt;
}

/// Handle arith.remui - emit v_and for power-of-2, general rem otherwise
LogicalResult handleArithRemUI(Operation *op, TranslationContext &ctx) {
  auto remOp = cast<arith::RemUIOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(remOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(remOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // Check if RHS is a constant power of 2
  if (auto rhsVal = getConstantValue(remOp.getRhs())) {
    int64_t value = *rhsVal;
    if (value > 0 && (value & (value - 1)) == 0) {
      // Modulo by 2^n = AND with (2^n - 1)
      int64_t mask = value - 1;
      auto maskImm = ctx.createImmType(mask);
      auto maskConst = ConstantOp::create(builder, loc, maskImm, mask);
      auto rem = V_AND_B32::create(builder, loc, vregType, *lhs, maskConst);
      ctx.getMapper().mapValue(remOp.getResult(), rem);
      return success();
    }
  }

  // General case: rem = lhs - (lhs / rhs) * rhs
  // This requires division, so for now emit a placeholder
  // TODO: Implement proper general modulo
  auto rem = V_AND_B32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(remOp.getResult(), rem);
  return success();
}

//===----------------------------------------------------------------------===//
// Floating-Point Arith Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle arith.addf - emit v_add_f32
LogicalResult handleArithAddF(Operation *op, TranslationContext &ctx) {
  auto addOp = cast<arith::AddFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(addOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(addOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto add = V_ADD_F32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(addOp.getResult(), add);
  return success();
}

/// Handle arith.subf - emit v_sub_f32
LogicalResult handleArithSubF(Operation *op, TranslationContext &ctx) {
  auto subOp = cast<arith::SubFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(subOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(subOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto sub = V_SUB_F32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(subOp.getResult(), sub);
  return success();
}

/// Handle arith.mulf - emit v_mul_f32
LogicalResult handleArithMulF(Operation *op, TranslationContext &ctx) {
  auto mulOp = cast<arith::MulFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(mulOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(mulOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  auto mul = V_MUL_F32::create(builder, loc, vregType, *lhs, *rhs);
  ctx.getMapper().mapValue(mulOp.getResult(), mul);
  return success();
}

/// Handle arith.divf - emit v_div (reciprocal + multiply for fast path)
LogicalResult handleArithDivF(Operation *op, TranslationContext &ctx) {
  auto divOp = cast<arith::DivFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto lhs = ctx.getMapper().getMapped(divOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(divOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // Use v_rcp_f32 + v_mul_f32 for fast division
  auto rcp = V_RCP_F32::create(builder, loc, vregType, *rhs);
  auto div = V_MUL_F32::create(builder, loc, vregType, *lhs, rcp);
  ctx.getMapper().mapValue(divOp.getResult(), div);
  return success();
}

/// Handle arith.negf - emit v_mul with -1.0
LogicalResult handleArithNegF(Operation *op, TranslationContext &ctx) {
  auto negOp = cast<arith::NegFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto operand = ctx.getMapper().getMapped(negOp.getOperand());
  if (!operand) {
    return op->emitError("operand not mapped");
  }

  // Negate using XOR with sign bit (0x80000000)
  int32_t signBit = 0x80000000;
  auto signImm = ctx.createImmType(signBit);
  auto signConst = ConstantOp::create(builder, loc, signImm, signBit);
  auto neg = V_XOR_B32::create(builder, loc, vregType, *operand, signConst);
  ctx.getMapper().mapValue(negOp.getResult(), neg);
  return success();
}

/// Handle arith.cmpf - emit v_cmp_f32 instructions
LogicalResult handleArithCmpF(Operation *op, TranslationContext &ctx) {
  auto cmpOp = cast<arith::CmpFOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto lhs = ctx.getMapper().getMapped(cmpOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(cmpOp.getRhs());

  if (!lhs || !rhs) {
    return op->emitError("operands not mapped");
  }

  // Emit comparison based on predicate (sets VCC)
  switch (cmpOp.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      V_CMP_EQ_F32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      V_CMP_NE_F32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      V_CMP_LT_F32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      V_CMP_LE_F32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      V_CMP_GT_F32::create(builder, loc, *lhs, *rhs);
      break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      V_CMP_GE_F32::create(builder, loc, *lhs, *rhs);
      break;
    default:
      // Handle other predicates as unordered
      V_CMP_NE_F32::create(builder, loc, *lhs, *rhs);
      break;
  }

  // Map result to a placeholder (select handler uses VCC implicitly)
  auto immOne = ctx.createImmType(1);
  auto one = ConstantOp::create(builder, loc, immOne, 1);
  ctx.getMapper().mapValue(cmpOp.getResult(), one);

  return success();
}

/// Handle math.fma - emit v_fma_f32
LogicalResult handleMathFma(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  if (op->getNumOperands() < 3) {
    return op->emitError("fma requires 3 operands");
  }

  auto a = ctx.getMapper().getMapped(op->getOperand(0));
  auto b = ctx.getMapper().getMapped(op->getOperand(1));
  auto c = ctx.getMapper().getMapped(op->getOperand(2));

  if (!a || !b || !c) {
    return op->emitError("operands not mapped");
  }

  auto fma = V_FMA_F32::create(builder, loc, vregType, *a, *b, *c);
  ctx.getMapper().mapValue(op->getResult(0), fma);
  return success();
}

//===----------------------------------------------------------------------===//
// Vector Dialect - Additional Handlers
//===----------------------------------------------------------------------===//

/// Handle vector.broadcast - replicate scalar to vector
LogicalResult handleVectorBroadcast(Operation *op, TranslationContext &ctx) {
  auto broadcastOp = cast<vector::BroadcastOp>(op);

  // For GPU, broadcast is typically a no-op (value is already lane-uniform
  // or will be handled by register allocation)
  auto src = ctx.getMapper().getMapped(broadcastOp.getSource());
  if (src) {
    ctx.getMapper().mapValue(broadcastOp.getResult(), *src);
  }
  return success();
}

/// Handle vector.extract - extract scalar from vector
/// Creates a new single-element register at the correct offset
LogicalResult handleVectorExtract(Operation *op, TranslationContext &ctx) {
  auto extractOp = cast<vector::ExtractOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(extractOp.getSource());
  if (!src) {
    return op->emitError("source value not mapped");
  }

  // Get the extraction index (position in the source vector)
  auto staticPos = extractOp.getStaticPosition();
  int64_t index = 0;
  if (!staticPos.empty()) {
    index = staticPos[0];
  }

  // Get the source register type to find the base physical register
  Type srcType = src->getType();
  int64_t baseIdx = 0;

  if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
    // Physical VGPR - extract element at offset
    baseIdx = pvreg.getIndex() + index;
    auto elemType = PVRegType::get(builder.getContext(), baseIdx, 1);
    auto elemReg = PrecoloredVRegOp::create(builder, loc, elemType, baseIdx, 1);
    ctx.getMapper().mapValue(extractOp.getResult(), elemReg);
  } else {
    // Virtual VGPR or other type - use waveasm.extract op
    // This will be lowered to proper register offset during register allocation
    auto elemType = ctx.createVRegType(1, 1);
    auto extractWaveOp = ExtractOp::create(builder, loc, elemType, *src,
                                                    builder.getI64IntegerAttr(index));
    ctx.getMapper().mapValue(extractOp.getResult(), extractWaveOp.getResult());
  }
  return success();
}

/// Handle vector.insert - insert scalar into vector
LogicalResult handleVectorInsert(Operation *op, TranslationContext &ctx) {
  auto insertOp = cast<vector::InsertOp>(op);

  // Pass through the destination (modification happens via register offset)
  auto dest = ctx.getMapper().getMapped(insertOp.getDest());
  if (dest) {
    ctx.getMapper().mapValue(insertOp.getResult(), *dest);
  }
  return success();
}

/// Handle vector.shape_cast - reinterpret vector shape
LogicalResult handleVectorShapeCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<vector::ShapeCastOp>(op);

  // Shape cast is a no-op at the register level
  auto src = ctx.getMapper().getMapped(castOp.getSource());
  if (src) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

/// Handle vector.transfer_read - similar to vector.load
LogicalResult handleVectorTransferRead(Operation *op, TranslationContext &ctx) {
  auto readOp = cast<vector::TransferReadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(readOp.getBase().getType());
  auto vectorType = readOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  int64_t numDwords = (numBytes + 3) / 4;

  auto vregType = ctx.createVRegType(numDwords, numDwords > 1 ? numDwords : 1);

  if (isLDSMemRef(memrefType)) {
    // LDS read
    Value vaddr;
    if (!readOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(readOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    Operation *loadInstr;
    if (numBytes == 8) {
      loadInstr = DS_READ_B64::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes == 16) {
      loadInstr = DS_READ_B128::create(builder, loc, TypeRange{vregType}, vaddr);
    } else {
      loadInstr = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    }
    ctx.getMapper().mapValue(readOp.getResult(), loadInstr->getResult(0));
  } else {
    // Global read - buffer load
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!readOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(readOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    Operation *loadInstr;
    if (numDwords == 1) {
      loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType}, srd, voffset);
    } else if (numDwords == 2) {
      loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{vregType}, srd, voffset);
    } else {
      loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{vregType}, srd, voffset);
    }
    ctx.getMapper().mapValue(readOp.getResult(), loadInstr->getResult(0));
  }

  return success();
}

/// Handle vector.transfer_write - similar to vector.store
LogicalResult handleVectorTransferWrite(Operation *op, TranslationContext &ctx) {
  auto writeOp = cast<vector::TransferWriteOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(writeOp.getBase().getType());
  auto vectorType = writeOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  int64_t numDwords = (numBytes + 3) / 4;

  auto data = ctx.getMapper().getMapped(writeOp.getVector());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS write
    Value vaddr;
    if (!writeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(writeOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    if (numBytes == 8) {
      DS_WRITE_B64::create(builder, loc, *data, vaddr);
    } else if (numBytes == 16) {
      DS_WRITE_B128::create(builder, loc, *data, vaddr);
    } else {
      DS_WRITE_B32::create(builder, loc, *data, vaddr);
    }
  } else {
    // Global write - buffer store
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!writeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(writeOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    if (numDwords == 1) {
      BUFFER_STORE_DWORD::create(builder, loc, *data, srd, voffset);
    } else if (numDwords == 2) {
      BUFFER_STORE_DWORDX2::create(builder, loc, *data, srd, voffset);
    } else {
      BUFFER_STORE_DWORDX4::create(builder, loc, *data, srd, voffset);
    }
  }

  return success();
}

// Note: gpu.subgroup_broadcast handler removed - not available in this MLIR version
// When available, it would emit v_readlane_b32 or v_readfirstlane_b32

/// Handle vector.fma - fused multiply-add
LogicalResult handleVectorFma(Operation *op, TranslationContext &ctx) {
  auto fmaOp = cast<vector::FMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto lhs = ctx.getMapper().getMapped(fmaOp.getLhs());
  auto rhs = ctx.getMapper().getMapped(fmaOp.getRhs());
  auto acc = ctx.getMapper().getMapped(fmaOp.getAcc());

  if (!lhs || !rhs || !acc) {
    return op->emitError("FMA operands not mapped");
  }

  auto resultType = fmaOp.getResult().getType();
  Type elemType;
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(resultType)) {
    elemType = vecType.getElementType();
    numElements = vecType.getNumElements();
  } else {
    elemType = resultType;
  }

  // Create result register
  auto vregType = ctx.createVRegType(numElements, 1);

  Value result;
  if (elemType.isF32()) {
    // v_fma_f32 dst, src0, src1, src2 : dst = src0 * src1 + src2
    result = V_FMA_F32::create(builder, loc, vregType, *lhs, *rhs, *acc);
  } else if (elemType.isF16()) {
    // v_fma_f16 for f16 types
    result = V_FMA_F16::create(builder, loc, vregType, *lhs, *rhs, *acc);
  } else {
    // Fall back to mul + add for other types
    auto mulResult = V_MUL_F32::create(builder, loc, vregType, *lhs, *rhs);
    result = V_ADD_F32::create(builder, loc, vregType, mulResult, *acc);
  }

  ctx.getMapper().mapValue(fmaOp.getResult(), result);
  return success();
}

/// Handle vector.reduction - reduction operations
LogicalResult handleVectorReduction(Operation *op, TranslationContext &ctx) {
  // vector.reduction has operands: vector to reduce, optional accumulator
  if (op->getNumOperands() < 1) {
    return op->emitError("reduction requires vector operand");
  }

  Value vector = op->getOperand(0);
  auto vectorMapped = ctx.getMapper().getMapped(vector);
  if (!vectorMapped) {
    return op->emitError("vector operand not mapped");
  }

  // For now, emit a comment - full reduction requires wave-level operations
  // like DPP or permute instructions
  ctx.emitComment("vector.reduction - wave-level reduction");

  // Simple fallback: just map the first element
  ctx.getMapper().mapValue(op->getResult(0), *vectorMapped);
  return success();
}

/// Handle scf.if - emit conditional execution
LogicalResult handleSCFIf(Operation *op, TranslationContext &ctx) {
  auto ifOp = cast<scf::IfOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get condition
  auto cond = ctx.getMapper().getMapped(ifOp.getCondition());
  if (!cond) {
    return op->emitError("condition not mapped");
  }

  // Create labels for if/else/endif
  std::string baseName = "L_if_" + std::to_string(reinterpret_cast<uintptr_t>(op));
  std::string elseLabel = baseName + "_else";
  std::string endLabel = baseName + "_end";

  // Branch to else block if condition is false
  auto elseLabelRef = SymbolRefAttr::get(builder.getContext(), elseLabel);
  S_CBRANCH_SCC0::create(builder, loc, elseLabelRef);

  // Translate then region
  for (Operation &thenOp : ifOp.getThenRegion().front().without_terminator()) {
    if (failed(translateOperation(&thenOp, ctx))) {
      return failure();
    }
  }

  // Jump to end (skip else block)
  if (!ifOp.getElseRegion().empty()) {
    auto endLabelRef = SymbolRefAttr::get(builder.getContext(), endLabel);
    S_BRANCH::create(builder, loc, endLabelRef);
  }

  // Else label and block
  if (!ifOp.getElseRegion().empty()) {
    LabelOp::create(builder, loc, elseLabel);
    for (Operation &elseOp : ifOp.getElseRegion().front().without_terminator()) {
      if (failed(translateOperation(&elseOp, ctx))) {
        return failure();
      }
    }
  }

  // End label
  LabelOp::create(builder, loc, endLabel);

  return success();
}

/// Handle memref.subview - compute offset
LogicalResult handleMemRefSubView(Operation *op, TranslationContext &ctx) {
  auto subviewOp = cast<memref::SubViewOp>(op);

  // Pass through source (offset computation handled by indices)
  if (auto src = ctx.getMapper().getMapped(subviewOp.getSource())) {
    ctx.getMapper().mapValue(subviewOp.getResult(), *src);
  }
  return success();
}

/// Handle memref.load - emit ds_read or buffer_load
LogicalResult handleMemRefLoad(Operation *op, TranslationContext &ctx) {
  auto loadOp = cast<memref::LoadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = loadOp.getMemRefType();
  auto vregType = ctx.createVRegType();

  if (isLDSMemRef(memrefType)) {
    // LDS load
    Value vaddr;
    if (!loadOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(loadOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    auto readOp = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    ctx.getMapper().mapValue(loadOp.getResult(), readOp.getResult(0));
  } else {
    // Global load
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!loadOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(loadOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    auto loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType}, srd, voffset);
    ctx.getMapper().mapValue(loadOp.getResult(), loadInstr.getResult(0));
  }

  return success();
}

/// Handle memref.store - emit ds_write or buffer_store
LogicalResult handleMemRefStore(Operation *op, TranslationContext &ctx) {
  auto storeOp = cast<memref::StoreOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = storeOp.getMemRefType();

  auto data = ctx.getMapper().getMapped(storeOp.getValueToStore());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS store
    Value vaddr;
    if (!storeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(storeOp.getIndices()[0])) {
        vaddr = *mapped;
      }
    }
    if (!vaddr) {
      auto immType = ctx.createImmType(0);
      vaddr = ConstantOp::create(builder, loc, immType, 0);
    }

    DS_WRITE_B32::create(builder, loc, *data, vaddr);
  } else {
    // Global store
    auto sregType = ctx.createSRegType(4, 4);
    auto srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);

    Value voffset;
    if (!storeOp.getIndices().empty()) {
      if (auto mapped = ctx.getMapper().getMapped(storeOp.getIndices()[0])) {
        voffset = *mapped;
      }
    }
    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    BUFFER_STORE_DWORD::create(builder, loc, *data, srd, voffset);
  }

  return success();
}

/// Handle memref.cast - pass through source
LogicalResult handleMemRefCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<memref::CastOp>(op);

  if (auto src = ctx.getMapper().getMapped(castOp.getSource())) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Affine Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle affine.apply - compile affine expression to arithmetic instructions
LogicalResult handleAffineApply(Operation *op, TranslationContext &ctx) {
  auto applyOp = cast<affine::AffineApplyOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto map = applyOp.getAffineMap();

  // Get the single operand (for single-dimension maps)
  if (applyOp.getOperands().empty()) {
    return op->emitError("affine.apply with no operands");
  }

  Value baseValue;
  if (auto mapped = ctx.getMapper().getMapped(applyOp.getOperands()[0])) {
    baseValue = *mapped;
  } else {
    return op->emitError("operand not mapped");
  }

  // For single result affine maps, analyze the expression
  if (map.getNumResults() != 1) {
    return op->emitError("only single-result affine maps supported");
  }

  AffineExpr expr = map.getResult(0);

  // Get thread ID upper bound for the first operand (used for simplification)
  // If the first operand is a thread ID with known upper bound, we can
  // simplify floor divisions where divisor >= upper_bound to 0
  int64_t threadIdUpperBound = 0;
  if (applyOp.getOperands().size() > 0) {
    threadIdUpperBound = ctx.getThreadIdUpperBound(applyOp.getOperands()[0]);
  }

  // HIGH-LEVEL SIMPLIFICATION: Check if the entire expression simplifies to just
  // the input symbol when floor divisions evaluate to 0
  // Pattern: s0 + (s0 floordiv N) * C where N >= upper_bound
  //       => s0 + 0 * C = s0
  if (threadIdUpperBound > 0) {
    // Check if expression is Add(symbol, Mul(FloorDiv(symbol, N), C))
    // where N >= threadIdUpperBound
    if (auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (addExpr.getKind() == AffineExprKind::Add) {
        // Check if LHS is the symbol and RHS is a Mul containing FloorDiv
        if (isa<AffineSymbolExpr>(addExpr.getLHS())) {
          if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(addExpr.getRHS())) {
            if (mulExpr.getKind() == AffineExprKind::Mul) {
              // Check if LHS of Mul is FloorDiv with divisor >= upperBound
              if (auto floorExpr = dyn_cast<AffineBinaryOpExpr>(mulExpr.getLHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv = dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      // Expression simplifies to just the symbol (s0)
                      // Map result to the thread ID value
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
              // Also check RHS of Mul
              if (auto floorExpr = dyn_cast<AffineBinaryOpExpr>(mulExpr.getRHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv = dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Extract constant addend from top-level expression for buffer store offset:N
  // Pattern: dynamic_expr + constant -> track constant separately
  // This allows using "offset:N" in buffer stores instead of computing in VGPR
  int64_t constAddend = 0;
  AffineExpr exprToCompile = expr;

  if (auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (addExpr.getKind() == AffineExprKind::Add) {
      // Check if RHS is a constant
      if (auto constRhs = dyn_cast<AffineConstantExpr>(addExpr.getRHS())) {
        constAddend = constRhs.getValue();
        exprToCompile = addExpr.getLHS();
      }
      // Also check if LHS is a constant
      else if (auto constLhs = dyn_cast<AffineConstantExpr>(addExpr.getLHS())) {
        constAddend = constLhs.getValue();
        exprToCompile = addExpr.getRHS();
      }
    }
  }

  // Simple pattern matching for common affine expressions
  // Pattern: d0 mod N -> v_and_b32 (when N is power of 2)
  // Pattern: d0 floordiv N -> v_lshrrev_b32 (when N is power of 2)
  // Pattern: d0 * N -> v_lshlrev_b32 (when N is power of 2)

  // Result type that includes bit range tracking for OR optimization
  struct ExprResult {
    Value value;
    BitRange range;
    ExprResult(Value v, BitRange r) : value(v), range(r) {}
  };

  // Helper to emit the compiled expression with bit range tracking
  std::function<ExprResult(AffineExpr)> compileExpr = [&](AffineExpr e) -> ExprResult {
    // Dimension reference
    if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
      if (dimExpr.getPosition() < applyOp.getOperands().size()) {
        Value operand = applyOp.getOperands()[dimExpr.getPosition()];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          // Use tracked bit range if available
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Symbol reference
    if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
      int64_t symIdx = map.getNumDims() + symExpr.getPosition();
      if (symIdx < static_cast<int64_t>(applyOp.getOperands().size())) {
        Value operand = applyOp.getOperands()[symIdx];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Constant
    if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
      int64_t val = constExpr.getValue();
      auto immType = ctx.createImmType(val);
      Value constVal = ConstantOp::create(builder, loc, immType, val);
      return ExprResult(constVal, BitRange::fromConstant(val));
    }

    // Binary expressions
    if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
      ExprResult lhsResult = compileExpr(binExpr.getLHS());
      ExprResult rhsResult = compileExpr(binExpr.getRHS());
      Value lhs = lhsResult.value;
      Value rhs = rhsResult.value;
      BitRange lhsRange = lhsResult.range;
      BitRange rhsRange = rhsResult.range;

      switch (binExpr.getKind()) {
        case AffineExprKind::Add: {
          // Check if bit ranges overlap - if not, use OR instead of ADD
          if (!lhsRange.overlaps(rhsRange)) {
            // Check if either operand is a shift (Mul by power of 2)
            // If so, emit v_lshl_or_b32 directly instead of lshlrev + or
            auto tryFuseShiftOr = [&](AffineExpr shiftExpr, Value orend,
                                       BitRange orendRange) -> std::optional<ExprResult> {
              if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(shiftExpr)) {
                if (mulExpr.getKind() == AffineExprKind::Mul) {
                  // Check for power of 2 multiplier
                  if (auto constRhs = dyn_cast<AffineConstantExpr>(mulExpr.getRHS())) {
                    int64_t val = constRhs.getValue();
                    if (val > 0 && (val & (val - 1)) == 0) {
                      // It's a shift! Emit v_lshl_or_b32 directly
                      int64_t shiftAmount = log2(val);
                      // Get the base value being shifted (compile without the multiply)
                      ExprResult baseResult = compileExpr(mulExpr.getLHS());
                      auto shiftImm = ctx.createImmType(shiftAmount);
                      auto shiftConst = ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                      // v_lshl_or_b32: dst = (src << shift) | orend
                      Value fusedResult = V_LSHL_OR_B32::create(builder, loc, vregType,
                                                                 baseResult.value, shiftConst, orend);
                      BitRange shiftedRange = baseResult.range.shiftLeft(shiftAmount);
                      BitRange resultRange = shiftedRange.merge(orendRange);
                      ctx.setBitRange(fusedResult, resultRange);
                      return ExprResult(fusedResult, resultRange);
                    }
                  }
                  // Also check LHS for constant
                  if (auto constLhs = dyn_cast<AffineConstantExpr>(mulExpr.getLHS())) {
                    int64_t val = constLhs.getValue();
                    if (val > 0 && (val & (val - 1)) == 0) {
                      int64_t shiftAmount = log2(val);
                      ExprResult baseResult = compileExpr(mulExpr.getRHS());
                      auto shiftImm = ctx.createImmType(shiftAmount);
                      auto shiftConst = ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                      Value fusedResult = V_LSHL_OR_B32::create(builder, loc, vregType,
                                                                 baseResult.value, shiftConst, orend);
                      BitRange shiftedRange = baseResult.range.shiftLeft(shiftAmount);
                      BitRange resultRange = shiftedRange.merge(orendRange);
                      ctx.setBitRange(fusedResult, resultRange);
                      return ExprResult(fusedResult, resultRange);
                    }
                  }
                }
              }
              return std::nullopt;
            };

            // Try to fuse: check if LHS is a shift
            if (auto result = tryFuseShiftOr(binExpr.getLHS(), rhs, rhsRange)) {
              return *result;
            }
            // Try to fuse: check if RHS is a shift
            if (auto result = tryFuseShiftOr(binExpr.getRHS(), lhs, lhsRange)) {
              return *result;
            }

            // No fusion possible, emit regular v_or_b32
            Value orResult = V_OR_B32::create(builder, loc, vregType, lhs, rhs);
            BitRange resultRange = lhsRange.merge(rhsRange);
            ctx.setBitRange(orResult, resultRange);
            return ExprResult(orResult, resultRange);
          }
          // Overlapping ranges - must use ADD
          Value addResult = V_ADD_U32::create(builder, loc, vregType, lhs, rhs);
          BitRange resultRange = lhsRange.extendForAdd(rhsRange);
          ctx.setBitRange(addResult, resultRange);
          return ExprResult(addResult, resultRange);
        }

        case AffineExprKind::Mul: {
          // Constant folding: if either operand is constant 0, result is 0
          if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
            if (constLhs.getValue() == 0) {
              auto immZero = ctx.createImmType(0);
              return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                                BitRange(0, 0));
            }
          }
          if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            if (constRhs.getValue() == 0) {
              auto immZero = ctx.createImmType(0);
              return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                                BitRange(0, 0));
            }
            // Check if RHS is constant power of 2 -> use shift
            int64_t val = constRhs.getValue();
            if (isPowerOf2(val)) {
              int64_t shiftAmount = log2(val);
              auto shiftAmt = ctx.createImmType(shiftAmount);
              auto shiftConst = ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
              Value shiftResult = V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, lhs);
              // Shift the bit range left by shiftAmount
              BitRange resultRange = lhsRange.shiftLeft(shiftAmount);
              ctx.setBitRange(shiftResult, resultRange);
              return ExprResult(shiftResult, resultRange);
            }
          }
          // Also check LHS for power of 2 multiply
          if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
            int64_t val = constLhs.getValue();
            if (isPowerOf2(val)) {
              int64_t shiftAmount = log2(val);
              auto shiftAmt = ctx.createImmType(shiftAmount);
              auto shiftConst = ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
              Value shiftResult = V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, rhs);
              BitRange resultRange = rhsRange.shiftLeft(shiftAmount);
              ctx.setBitRange(shiftResult, resultRange);
              return ExprResult(shiftResult, resultRange);
            }
          }
          Value mulResult = V_MUL_LO_U32::create(builder, loc, vregType, lhs, rhs);
          return ExprResult(mulResult, BitRange());  // Conservative: full range
        }

        case AffineExprKind::FloorDiv: {
          // Check if RHS is constant
          if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t divisor = constRhs.getValue();

            // SIMPLIFICATION: If the LHS is a thread ID with upper_bound <= divisor,
            // then floor(tid / divisor) = 0 for all valid thread IDs.
            // Example: tid_x in [0, 63], floor(tid_x / 64) = 0
            if (threadIdUpperBound > 0 && divisor >= threadIdUpperBound) {
              // LHS is in range [0, upper_bound-1], so floor(LHS / divisor) = 0
              auto immZero = ctx.createImmType(0);
              return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                                BitRange(0, 0));
            }

            // Check if divisor is power of 2 -> use right shift
            if (isPowerOf2(divisor)) {
              int64_t shiftAmount = log2(divisor);
              auto shiftAmt = ctx.createImmType(shiftAmount);
              auto shiftConst = ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
              Value shiftResult = V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, lhs);
              // Shift the bit range right by shiftAmount
              BitRange resultRange = lhsRange.shiftRight(shiftAmount);
              ctx.setBitRange(shiftResult, resultRange);
              return ExprResult(shiftResult, resultRange);
            }
          }
          // General floordiv - needs more complex handling
          return ExprResult(lhs, BitRange());  // Conservative
        }

        case AffineExprKind::Mod: {
          // Check if RHS is constant power of 2 -> use AND
          if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t val = constRhs.getValue();
            if (isPowerOf2(val)) {
              auto maskVal = ctx.createImmType(val - 1);
              auto maskConst = ConstantOp::create(builder, loc, maskVal, val - 1);
              Value andResult = V_AND_B32::create(builder, loc, vregType, lhs, maskConst);
              // Result uses bits 0..(log2(val)-1)
              BitRange resultRange = BitRange(0, log2(val) - 1);
              ctx.setBitRange(andResult, resultRange);
              return ExprResult(andResult, resultRange);
            }
          }
          // General mod - needs more complex handling
          return ExprResult(lhs, BitRange());  // Conservative
        }

        default:
          return ExprResult(lhs, BitRange());  // Unsupported, return LHS as fallback
      }
    }

    return ExprResult(baseValue, BitRange());  // Fallback
  };

  ExprResult result = compileExpr(exprToCompile);
  ctx.getMapper().mapValue(applyOp.getResult(), result.value);
  ctx.setBitRange(result.value, result.range);

  // Track the constant addend for buffer store offset:N optimization
  if (constAddend != 0) {
    ctx.setConstOffset(applyOp.getResult(), constAddend);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Vector Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle vector.load - emit buffer_load or ds_read based on address space
/// Splits large loads (> 16 bytes) into multiple buffer_load_dwordx4 instructions
LogicalResult handleVectorLoad(Operation *op, TranslationContext &ctx) {
  auto loadOp = cast<vector::LoadOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(loadOp.getBase().getType());
  auto vectorType = loadOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  int64_t numDwords = (numBytes + 3) / 4;

  if (isLDSMemRef(memrefType)) {
    // LDS load - ds_read_b* with proper byte address computation
    auto vregType = ctx.createVRegType(numDwords, numDwords > 1 ? numDwords : 1);
    auto indices = loadOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Compute vaddr as byte offset from indices and strides
    // Also track instruction offset from constant addends in affine expressions
    Value vaddr;
    int64_t instOffset = 0;  // Instruction offset for ds_read offset:N
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped) continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0) continue;

        // Check if this index has a constant addend from affine.apply
        // This allows using "offset:N" in the instruction instead of VGPR computation
        int64_t constAddend = ctx.getConstOffset(indices[i]);
        if (constAddend != 0) {
          // Add constant * stride to instruction offset
          instOffset += constAddend * strideBytes;
        }

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(), idx, strideImm);
        }

        if (!vaddr) {
          vaddr = dimOffset;
        } else {
          vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr, dimOffset);
        }
      }
    }

    // Fallback: use first index with element size scaling
    if (!vaddr) {
      auto idxMapped = ctx.getMapper().getMapped(indices[0]);
      if (idxMapped) {
        if (elementBytes > 1) {
          int shift = 0;
          int64_t temp = elementBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          vaddr = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, *idxMapped);
        } else {
          vaddr = *idxMapped;
        }
      } else {
        auto immType = ctx.createImmType(0);
        vaddr = ConstantOp::create(builder, loc, immType, 0);
      }
    }

    // Add the LDS base offset from memref.view (if any)
    // This handles cases like memref.view with byte_shift = 640
    if (auto baseOffset = ctx.getLDSBaseOffset(loadOp.getBase())) {
      // Need to move the base offset to a VGPR first since v_add_u32 requires
      // src1 to be a VGPR, and the base offset may be an immediate
      Value baseOffsetVgpr = V_MOV_B32::create(builder, loc, ctx.createVRegType(), *baseOffset);
      vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr, baseOffsetVgpr);
    }

    // Create the DS_READ operation with optional offset attribute
    Operation *readOp;
    if (numBytes == 8) {
      readOp = DS_READ_B64::create(builder, loc, TypeRange{vregType}, vaddr);
    } else if (numBytes == 16) {
      readOp = DS_READ_B128::create(builder, loc, TypeRange{vregType}, vaddr);
    } else {
      readOp = DS_READ_B32::create(builder, loc, TypeRange{vregType}, vaddr);
    }

    // Add offset attribute if we have a non-zero instruction offset
    if (instOffset != 0) {
      readOp->setAttr("offset", builder.getI64IntegerAttr(instOffset));
    }

    ctx.getMapper().mapValue(loadOp.getResult(), readOp->getResult(0));
  } else {
    // Global load - buffer_load_dwordx* with splitting for large vectors

    // Compute voffset as byte offset from indices and strides
    // For memref<...xf16, strided<[stride0, stride1]>>, byte_offset =
    //   (idx0 * stride0 + idx1 * stride1) * element_size
    Value voffset;
    auto indices = loadOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Get strides from the memref type
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    int64_t instOffset = 0;  // Instruction offset for offset:N modifier
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Compute linearized byte offset
      // For a 2D memref with indices [i, j] and strides [s0, s1]:
      // byte_offset = (i * s0 + j * s1) * element_size

      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped) continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0) continue;

        // Check if this index has a constant addend that can be used for instOffset
        // This allows using "offset:N" in the instruction instead of VGPR computation
        int64_t constAddend = ctx.getConstOffset(indices[i]);
        if (constAddend != 0) {
          // Add constant * stride to instruction offset
          instOffset += constAddend * strideBytes;
        }

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(), idx, strideImm);
        }

        if (!voffset) {
          voffset = dimOffset;
        } else {
          // Add to existing offset
          voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(), voffset, dimOffset);
        }
      }
    } else {
      // Fallback: use first index directly if strides not available
      if (!indices.empty()) {
        if (auto mapped = ctx.getMapper().getMapped(indices[0])) {
          voffset = *mapped;
        }
      }
    }

    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    // Get SRD for this memref - look up from binding or use tracked SRD
    Value srd;
    if (auto srdIdx = ctx.getSRDIndex(loadOp.getBase())) {
      auto sregType = ctx.createSRegType(4, 4);
      srd = PrecoloredSRegOp::create(builder, loc, sregType, *srdIdx, 4);
    } else if (auto mapped = ctx.getMapper().getMapped(loadOp.getBase())) {
      srd = *mapped;
    } else {
      // Fallback to default SRD at s[8:11]
      auto sregType = ctx.createSRegType(4, 4);
      srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);
    }

    // Split large loads into multiple buffer_load_dwordx4 (16 bytes each)
    // Use the same voffset for all loads, with instOffset for subsequent chunks
    SmallVector<Value, 4> loadResults;
    int64_t bytesRemaining = numBytes;
    int64_t currentOffset = 0;

    while (bytesRemaining > 0) {
      int64_t loadBytes;
      int64_t loadDwords;

      if (bytesRemaining >= 16) {
        loadBytes = 16;
        loadDwords = 4;
      } else if (bytesRemaining >= 8) {
        loadBytes = 8;
        loadDwords = 2;
      } else {
        loadBytes = 4;
        loadDwords = 1;
      }

      auto loadVregType = ctx.createVRegType(loadDwords, loadDwords > 1 ? loadDwords : 1);

      // Use instOffset attribute instead of computing new voffset
      // This generates "offset:N" modifier in assembly, saving a V_ADD_U32 instruction
      // Combine the base instOffset (from affine constant addends) with currentOffset (for split loads)
      int64_t totalOffset = instOffset + currentOffset;
      Operation *loadInstr;
      if (loadDwords == 4) {
        loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{loadVregType}, srd, voffset, totalOffset);
      } else if (loadDwords == 2) {
        loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{loadVregType}, srd, voffset, totalOffset);
      } else {
        loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{loadVregType}, srd, voffset, totalOffset);
      }

      loadResults.push_back(loadInstr->getResult(0));
      bytesRemaining -= loadBytes;
      currentOffset += loadBytes;
    }

    // Map the first result to the vector.load result
    // For composite results, also register all split results for use in stores
    if (!loadResults.empty()) {
      ctx.getMapper().mapValue(loadOp.getResult(), loadResults[0]);

      // Register all split results for later use in vector.store
      if (loadResults.size() > 1) {
        ctx.registerSplitResults(loadOp.getResult(), loadResults);
      }
    }
  }

  return success();
}

/// Handle vector.store - emit buffer_store or ds_write
/// Splits large stores (> 16 bytes) into multiple buffer_store_dwordx4 instructions
LogicalResult handleVectorStore(Operation *op, TranslationContext &ctx) {
  auto storeOp = cast<vector::StoreOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto memrefType = cast<MemRefType>(storeOp.getBase().getType());
  auto vectorType = storeOp.getVectorType();
  int64_t numBytes = getVectorBytes(vectorType);
  [[maybe_unused]] int64_t numDwords = (numBytes + 3) / 4;

  auto data = ctx.getMapper().getMapped(storeOp.getValueToStore());
  if (!data) {
    return op->emitError("data value not mapped");
  }

  if (isLDSMemRef(memrefType)) {
    // LDS store - ds_write_b* with proper byte address computation
    auto indices = storeOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Compute vaddr as byte offset from indices and strides
    Value vaddr;
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped) continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0) continue;

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(), idx, strideImm);
        }

        if (!vaddr) {
          vaddr = dimOffset;
        } else {
          vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr, dimOffset);
        }
      }
    }

    // Fallback: use first index with element size scaling
    if (!vaddr) {
      auto idxMapped = ctx.getMapper().getMapped(indices[0]);
      if (idxMapped) {
        if (elementBytes > 1) {
          int shift = 0;
          int64_t temp = elementBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          vaddr = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, *idxMapped);
        } else {
          vaddr = *idxMapped;
        }
      } else {
        auto immType = ctx.createImmType(0);
        vaddr = ConstantOp::create(builder, loc, immType, 0);
      }
    }

    // Add the LDS base offset from memref.view (if any)
    // This handles cases like memref.view with byte_shift = 640
    if (auto baseOffset = ctx.getLDSBaseOffset(storeOp.getBase())) {
      // Need to move the base offset to a VGPR first since v_add_u32 requires
      // src1 to be a VGPR, and the base offset may be an immediate
      Value baseOffsetVgpr = V_MOV_B32::create(builder, loc, ctx.createVRegType(), *baseOffset);
      vaddr = V_ADD_U32::create(builder, loc, ctx.createVRegType(), vaddr, baseOffsetVgpr);
    }

    if (numBytes == 8) {
      DS_WRITE_B64::create(builder, loc, *data, vaddr);
    } else if (numBytes == 16) {
      DS_WRITE_B128::create(builder, loc, *data, vaddr);
    } else {
      DS_WRITE_B32::create(builder, loc, *data, vaddr);
    }
  } else {
    // Global store - buffer_store_dwordx* with splitting for large vectors

    // Compute voffset as byte offset from indices and strides
    // For 2D memrefs: offset = idx0 * stride0 * elementBytes + idx1 * stride1 * elementBytes
    Value voffset;
    int64_t instOffset = 0;  // Constant offset for buffer_store offset:N syntax
    auto indices = storeOp.getIndices();
    Type elementType = memrefType.getElementType();
    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

    // Get strides from the memref type
    SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (succeeded(memrefType.getStridesAndOffset(strides, offset))) {
      // Process each index dimension
      for (size_t i = 0; i < indices.size() && i < strides.size(); ++i) {
        auto idxMapped = ctx.getMapper().getMapped(indices[i]);
        if (!idxMapped) continue;

        Value idx = *idxMapped;
        int64_t strideBytes = strides[i] * elementBytes;

        if (strideBytes == 0) continue;

        // Check if this index has a constant addend that can be used for instOffset
        // This allows using "offset:N" in the instruction instead of VGPR computation
        int64_t constAddend = ctx.getConstOffset(indices[i]);
        if (constAddend != 0) {
          // Add constant * stride to instruction offset
          instOffset += constAddend * strideBytes;
        }

        Value dimOffset;
        if (strideBytes == 1) {
          dimOffset = idx;
        } else if ((strideBytes & (strideBytes - 1)) == 0) {
          // Power of 2 - use shift
          int shift = 0;
          int64_t temp = strideBytes;
          while (temp > 1) { shift++; temp >>= 1; }
          auto shiftImm = ConstantOp::create(builder, loc, ctx.createImmType(shift), shift);
          dimOffset = V_LSHLREV_B32::create(builder, loc, ctx.createVRegType(), shiftImm, idx);
        } else {
          // General case - use multiply
          auto strideImm = ConstantOp::create(builder, loc, ctx.createImmType(strideBytes), strideBytes);
          dimOffset = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(), idx, strideImm);
        }

        if (!voffset) {
          voffset = dimOffset;
        } else {
          // Add to existing offset
          voffset = V_ADD_U32::create(builder, loc, ctx.createVRegType(), voffset, dimOffset);
        }
      }
    } else {
      // Fallback: use first index directly if strides not available
      if (!indices.empty()) {
        if (auto mapped = ctx.getMapper().getMapped(indices[0])) {
          voffset = *mapped;
        }
      }
    }

    if (!voffset) {
      auto immType = ctx.createImmType(0);
      voffset = ConstantOp::create(builder, loc, immType, 0);
    }

    // Get SRD for this memref - look up from binding or use tracked SRD
    Value srd;
    if (auto srdIdx = ctx.getSRDIndex(storeOp.getBase())) {
      auto sregType = ctx.createSRegType(4, 4);
      srd = PrecoloredSRegOp::create(builder, loc, sregType, *srdIdx, 4);
    } else if (auto mapped = ctx.getMapper().getMapped(storeOp.getBase())) {
      srd = *mapped;
    } else {
      // Fallback to default SRD at s[8:11]
      auto sregType = ctx.createSRegType(4, 4);
      srd = PrecoloredSRegOp::create(builder, loc, sregType, 8, 4);
    }

    // Check if the source value has split results from a corresponding load
    auto splitResults = ctx.getSplitResults(storeOp.getValueToStore());

    // Split large stores into multiple buffer_store_dwordx4 (16 bytes each)
    // Use the same voffset for all stores, with instOffset for subsequent chunks
    // Add any constant offset from affine expressions to the base offset
    int64_t bytesRemaining = numBytes;
    int64_t currentOffset = instOffset;  // Start with constant offset from affine expressions
    size_t splitIndex = 0;

    while (bytesRemaining > 0) {
      int64_t storeBytes;
      int64_t storeDwords;

      if (bytesRemaining >= 16) {
        storeBytes = 16;
        storeDwords = 4;
      } else if (bytesRemaining >= 8) {
        storeBytes = 8;
        storeDwords = 2;
      } else {
        storeBytes = 4;
        storeDwords = 1;
      }

      // Use the correct split result if available, otherwise use mapped data
      Value storeData = *data;
      if (!splitResults.empty() && splitIndex < splitResults.size()) {
        storeData = splitResults[splitIndex];
      }

      // Use instOffset attribute instead of computing new voffset
      // This generates "offset:N" modifier in assembly, saving a V_ADD_U32 instruction
      if (storeDwords == 4) {
        BUFFER_STORE_DWORDX4::create(builder, loc, storeData, srd, voffset, currentOffset);
      } else if (storeDwords == 2) {
        BUFFER_STORE_DWORDX2::create(builder, loc, storeData, srd, voffset, currentOffset);
      } else {
        BUFFER_STORE_DWORD::create(builder, loc, storeData, srd, voffset, currentOffset);
      }

      bytesRemaining -= storeBytes;
      currentOffset += storeBytes;
      splitIndex++;
    }
  }

  return success();
}

/// Handle vector.extract_strided_slice - extract subset of source registers
/// Creates a register alias at the correct offset for proper assembly emission
LogicalResult handleVectorExtractStridedSlice(Operation *op, TranslationContext &ctx) {
  auto extractOp = cast<vector::ExtractStridedSliceOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  auto src = ctx.getMapper().getMapped(extractOp.getSource());
  if (!src) {
    return op->emitError("source value not mapped");
  }

  // Get the offset from the offsets attribute (for 1D, it's a single value)
  auto offsets = extractOp.getOffsets();
  int64_t offset = 0;
  if (!offsets.empty()) {
    offset = cast<IntegerAttr>(offsets[0]).getInt();
  }

  // Get the size from the sizes attribute (for 1D, it's a single value)
  auto sizes = extractOp.getSizes();
  int64_t size = 1;
  if (!sizes.empty()) {
    size = cast<IntegerAttr>(sizes[0]).getInt();
  }

  // Get the source register type to find the base physical register
  Type srcType = src->getType();

  if (auto pvreg = dyn_cast<PVRegType>(srcType)) {
    // Physical VGPR - extract element(s) at offset
    int64_t baseIdx = pvreg.getIndex() + offset;
    auto elemType = PVRegType::get(builder.getContext(), baseIdx, size);
    auto elemReg = PrecoloredVRegOp::create(builder, loc, elemType, baseIdx, size);
    ctx.getMapper().mapValue(extractOp.getResult(), elemReg);
  } else {
    // Virtual VGPR or other type - use waveasm.extract op
    // This will be lowered to proper register offset during register allocation
    auto elemType = ctx.createVRegType(size, 1);
    auto extractWaveOp = ExtractOp::create(builder, loc, elemType, *src,
                                                    builder.getI64IntegerAttr(offset));
    ctx.getMapper().mapValue(extractOp.getResult(), extractWaveOp.getResult());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AMDGPU Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle amdgpu.lds_barrier - emit s_waitcnt + s_barrier
LogicalResult handleAMDGPULdsBarrier(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Wait for all LDS operations to complete
  S_WAITCNT_LGKMCNT::create(builder, loc, 0);
  // Barrier synchronization
  S_BARRIER::create(builder, loc);

  return success();
}

/// Handle amdgpu.mfma - emit v_mfma instruction
/// Supports all MFMA variants: f16, bf16, i8, f32, f64, fp8/bf8
LogicalResult handleAMDGPUMfma(Operation *op, TranslationContext &ctx) {
  auto mfmaOp = cast<amdgpu::MFMAOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get MFMA dimensions
  int64_t m = mfmaOp.getM();
  int64_t n = mfmaOp.getN();
  int64_t k = mfmaOp.getK();

  // Get operands
  auto srcA = ctx.getMapper().getMapped(mfmaOp.getSourceA());
  auto srcB = ctx.getMapper().getMapped(mfmaOp.getSourceB());
  auto srcC = ctx.getMapper().getMapped(mfmaOp.getDestC());

  if (!srcA || !srcB || !srcC) {
    return op->emitError("MFMA operands not mapped");
  }

  // Determine element type from sourceA
  Type srcAType = mfmaOp.getSourceA().getType();
  Type elemType;
  if (auto vecType = dyn_cast<VectorType>(srcAType)) {
    elemType = vecType.getElementType();
  } else {
    elemType = srcAType;
  }

  // Determine accumulator size based on MFMA variant
  int64_t accSize = 4;  // Default for 16x16xk small variants
  if (m == 32 && n == 32) {
    accSize = 16;  // 32x32xk variants
  } else if (m == 4 && n == 4) {
    accSize = 4;   // 4x4xk variants
  } else if (m == 16 && n == 16 && k == 4) {
    accSize = 16;  // 16x16x4 variants (larger accumulator)
  }

  // For f64 variants, sizes are different
  if (elemType.isF64()) {
    if (m == 16 && n == 16) accSize = 8;
    else if (m == 4 && n == 4) accSize = 2;
  }

  auto vregType = ctx.createVRegType(accSize, 4);  // Quad-aligned

  // Select MFMA instruction based on dimensions and element type
  Value result;

  // Helper to check if element type matches
  auto isF16 = [&]() { return elemType.isF16(); };
  auto isBF16 = [&]() { return elemType.isBF16(); };
  auto isI8 = [&]() { return elemType.isInteger(8); };
  auto isF32 = [&]() { return elemType.isF32(); };
  auto isF64 = [&]() { return elemType.isF64(); };

  // F16 variants
  if (isF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      // gfx950+ (MI350) - v_mfma_f32_16x16x32_f16
      result = V_MFMA_F32_16X16X32_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_F16::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_F16::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    }
  }
  // BF16 variants
  else if (isBF16()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 32) {
      // gfx950+ (MI350) - v_mfma_f32_16x16x32_bf16
      result = V_MFMA_F32_16X16X32_BF16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_F32_32X32X8_BF16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_F32_16X16X4_BF16::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_F32_32X32X4_BF16::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_F32_4X4X4_BF16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X16_BF16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    }
  }
  // I8 variants
  else if (isI8()) {
    if (m == 16 && n == 16 && k == 16) {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 8) {
      result = V_MFMA_I32_32X32X8_I8::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 16 && n == 16 && k == 4) {
      auto largeVregType = ctx.createVRegType(16, 4);
      result = V_MFMA_I32_16X16X4_I8::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 4) {
      auto largeVregType = ctx.createVRegType(32, 4);
      result = V_MFMA_I32_32X32X4_I8::create(builder, loc, largeVregType, *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      result = V_MFMA_I32_4X4X4_I8::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else {
      result = V_MFMA_I32_16X16X16_I8::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    }
  }
  // F32 variants
  else if (isF32()) {
    if (m == 16 && n == 16 && k == 4) {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 32 && n == 32 && k == 2) {
      result = V_MFMA_F32_32X32X2_F32::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 1) {
      result = V_MFMA_F32_4X4X1_F32::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    } else {
      result = V_MFMA_F32_16X16X4_F32::create(builder, loc, vregType, *srcA, *srcB, *srcC);
    }
  }
  // F64 variants
  else if (isF64()) {
    if (m == 16 && n == 16 && k == 4) {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA, *srcB, *srcC);
    } else if (m == 4 && n == 4 && k == 4) {
      auto f64VregType = ctx.createVRegType(2, 2);
      result = V_MFMA_F64_4X4X4_F64::create(builder, loc, f64VregType, *srcA, *srcB, *srcC);
    } else {
      auto f64VregType = ctx.createVRegType(8, 4);
      result = V_MFMA_F64_16X16X4_F64::create(builder, loc, f64VregType, *srcA, *srcB, *srcC);
    }
  }
  // Default to F16 16x16x16
  else {
    result = V_MFMA_F32_16X16X16_F16::create(builder, loc, vregType, *srcA, *srcB, *srcC);
  }

  ctx.getMapper().mapValue(mfmaOp.getDestD(), result);
  return success();
}

//===----------------------------------------------------------------------===//
// MemRef Dialect Handlers
//===----------------------------------------------------------------------===//

/// Handle memref.alloc - track LDS allocation
LogicalResult handleMemRefAlloc(Operation *op, TranslationContext &ctx) {
  auto allocOp = cast<memref::AllocOp>(op);

  // Check if this is an LDS allocation (workgroup address space)
  auto memrefType = allocOp.getResult().getType();
  if (isLDSMemRef(memrefType)) {
    // Compute the allocation size in bytes
    int64_t numElements = 1;
    for (int64_t dim : memrefType.getShape()) {
      if (dim != ShapedType::kDynamic) {
        numElements *= dim;
      }
    }
    int64_t elementBytes = (memrefType.getElementTypeBitWidth() + 7) / 8;
    int64_t allocSize = numElements * elementBytes;

    // Track the total LDS size for the kernel descriptor
    ctx.addLDSSize(allocSize);
  }

  return success();
}

/// Handle memref.view - compute LDS offset
LogicalResult handleMemRefView(Operation *op, TranslationContext &ctx) {
  auto viewOp = cast<memref::ViewOp>(op);

  // Track the byte offset for LDS addressing
  // The byteShift is the base offset into LDS that this view starts at
  if (auto offset = ctx.getMapper().getMapped(viewOp.getByteShift())) {
    // Store the LDS base offset for this memref so it can be added
    // during vector.load/store operations
    ctx.setLDSBaseOffset(viewOp.getResult(), *offset);
  }

  return success();
}

/// Handle memref.reinterpret_cast - track memref identity
LogicalResult handleMemRefReinterpretCast(Operation *op, TranslationContext &ctx) {
  auto castOp = cast<memref::ReinterpretCastOp>(op);

  // Reinterpret cast doesn't change the underlying buffer
  if (auto src = ctx.getMapper().getMapped(castOp.getSource())) {
    ctx.getMapper().mapValue(castOp.getResult(), *src);
  }

  // Propagate LDS base offset from source to result
  // This is needed when memref.view creates an LDS view with a byte offset,
  // and then memref.reinterpret_cast is applied to reshape it
  if (auto ldsOffset = ctx.getLDSBaseOffset(castOp.getSource())) {
    ctx.setLDSBaseOffset(castOp.getResult(), *ldsOffset);
  }

  // The result type often has more specific shape info than the source.
  // Update the SRD buffer size if the result type is larger.
  if (auto memrefType = dyn_cast<MemRefType>(castOp.getResult().getType())) {
    int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
    // Update the SRD for the source memref with the more accurate size
    ctx.updateSRDBufferSize(castOp.getSource(), bufferSize);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SCF Dialect Handlers
//===----------------------------------------------------------------------===//

LogicalResult handleSCFFor(Operation *op, TranslationContext &ctx);
LogicalResult handleSCFIf(Operation *op, TranslationContext &ctx);
LogicalResult handleSCFYield(Operation *op, TranslationContext &ctx);

/// Handle scf.for - emit loop structure with iter_args support
LogicalResult handleSCFFor(Operation *op, TranslationContext &ctx) {
  auto forOp = cast<scf::ForOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Generate unique loop label
  std::string labelName = ctx.generateLabel("L_loop");
  std::string endLabelName = labelName + "_end";

  // Initialize loop counter in a FIXED physical SGPR
  // Using a fixed register ensures the counter value persists across loop iterations
  // Reserve s32+ for loop counters (avoiding s0-s31 which may be used for arguments/SRDs/cache-swizzle)
  // Note: s[24:31] may be used for cache swizzle SRDs in g2s kernels
  int64_t loopCounterPhysReg = 32 + ctx.getLoopDepth();  // Use s32, s33, etc. for nested loops
  auto counterType = PSRegType::get(builder.getContext(), loopCounterPhysReg, 1);

  auto lb = ctx.getMapper().getMapped(forOp.getLowerBound());
  if (!lb) {
    auto immType = ctx.createImmType(0);
    lb = ConstantOp::create(builder, loc, immType, 0);
  }

  auto counter = S_MOV_B32::create(builder, loc, counterType, *lb);
  ctx.getMapper().mapValue(forOp.getInductionVar(), counter);

  // Handle iter_args (loop-carried values)
  // These are values passed from one iteration to the next (e.g., accumulators)
  // For vector-type iter_args (accumulators), we need to:
  // 1. Allocate VREGs before the loop
  // 2. Initialize them with v_mov_b32
  // 3. Map the region arg to those VREGs for in-place accumulation
  SmallVector<Value, 4> iterArgValues;
  for (auto [initArg, regionArg] : llvm::zip(forOp.getInitArgs(),
                                              forOp.getRegionIterArgs())) {
    auto mapped = ctx.getMapper().getMapped(initArg);

    // Check if this is a vector type (likely an accumulator for MFMA)
    Type regionArgType = regionArg.getType();
    if (auto vecType = dyn_cast<VectorType>(regionArgType)) {
      // For vector accumulators, we need to materialize VREGs
      int64_t numElems = vecType.getNumElements();

      // Check if the init value is a constant 0 (common for accumulators)
      bool isZeroInit = false;
      if (auto constOp = initArg.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat()) {
            if (auto floatAttr = dyn_cast<FloatAttr>(denseAttr.getSplatValue<Attribute>())) {
              isZeroInit = (floatAttr.getValueAsDouble() == 0.0);
            }
          }
        }
      }

      if (isZeroInit) {
        // Allocate VREGs for the accumulator and initialize to 0
        // The MFMA instruction will use these registers for in-place accumulation
        auto vregType = ctx.createVRegType(numElems, 4);  // Quad-aligned for MFMA
        auto zeroImm = ctx.createImmType(0);
        auto zero = ConstantOp::create(builder, loc, zeroImm, 0);

        // Create v_mov_b32 to initialize each element of the accumulator
        // Mark with "no_cse" to prevent CSE from merging multiple accumulators
        auto accMovOp = V_MOV_B32::create(builder, loc, vregType, zero);
        accMovOp->setAttr("no_cse", builder.getUnitAttr());
        Value accReg = accMovOp.getResult();

        // Map the region argument to this register
        ctx.getMapper().mapValue(regionArg, accReg);
        iterArgValues.push_back(accReg);
        continue;
      }
    }

    // Default: map to the initial value directly
    if (mapped) {
      ctx.getMapper().mapValue(regionArg, *mapped);
      iterArgValues.push_back(*mapped);
    }
  }

  // Set up loop context for nested operations
  LoopContext loopCtx;
  loopCtx.inductionVar = counter;
  loopCtx.iterArgs = iterArgValues;
  loopCtx.labelName = labelName;
  loopCtx.depth = ctx.getLoopDepth() + 1;
  ctx.pushLoopContext(loopCtx);

  // Clear expression cache at loop entry (loop-variant expressions must be recomputed)
  ctx.clearExprCache();

  // Loop label
  LabelOp::create(builder, loc, labelName);

  // Translate loop body
  for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
    if (failed(translateOperation(&bodyOp, ctx))) {
      ctx.popLoopContext();
      return failure();
    }
  }

  // Handle yield - update iter_args for next iteration
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (auto [yieldedVal, regionArg] : llvm::zip(yieldOp.getOperands(),
                                                   forOp.getRegionIterArgs())) {
      auto mapped = ctx.getMapper().getMapped(yieldedVal);
      if (mapped) {
        // Update the region argument mapping for the next iteration
        ctx.getMapper().mapValue(regionArg, *mapped);
      }
    }
  }

  // Increment counter - use the same physical register type as the counter
  // This ensures the increment writes to the same physical register
  auto step = ctx.getMapper().getMapped(forOp.getStep());
  if (!step) {
    auto immType = ctx.createImmType(1);
    step = ConstantOp::create(builder, loc, immType, 1);
  }

  // Get the counter type from the original counter (it's a physical register type)
  auto counterPhysType = counter.getType();
  auto newCounter = S_ADD_U32::create(builder, loc, counterPhysType, counter, *step);

  // Since we're using physical registers, newCounter is in the same register as counter
  // Update the mapping for any post-loop uses
  ctx.getMapper().mapValue(forOp.getInductionVar(), newCounter);

  // Compare and branch back to loop header
  auto ub = ctx.getMapper().getMapped(forOp.getUpperBound());
  if (ub) {
    S_CMP_LT_U32::create(builder, loc, newCounter, *ub);
    auto labelRef = SymbolRefAttr::get(builder.getContext(), labelName);
    S_CBRANCH_SCC1::create(builder, loc, labelRef);
  }

  // End label for loop exit
  LabelOp::create(builder, loc, endLabelName);

  // Map loop results to final iter_arg values
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (auto [result, yieldedVal] : llvm::zip(forOp.getResults(),
                                                yieldOp.getOperands())) {
      auto mapped = ctx.getMapper().getMapped(yieldedVal);
      if (mapped) {
        ctx.getMapper().mapValue(result, *mapped);
      }
    }
  }

  ctx.popLoopContext();
  return success();
}

/// Handle scf.yield - typically a no-op
LogicalResult handleSCFYield(Operation *op, TranslationContext &ctx) {
  // Yield values are handled by the parent op
  return success();
}

} // namespace

namespace waveasm {

//===----------------------------------------------------------------------===//
// OpHandlerRegistry Implementation
//===----------------------------------------------------------------------===//

OpHandlerRegistry::OpHandlerRegistry() {
  registerDefaultHandlers();
}

void OpHandlerRegistry::registerHandler(StringRef opName, OpHandler handler) {
  handlers[opName] = std::move(handler);
}

std::optional<OpHandler> OpHandlerRegistry::getHandler(StringRef opName) const {
  auto it = handlers.find(opName);
  if (it != handlers.end())
    return it->second;
  return std::nullopt;
}

bool OpHandlerRegistry::hasHandler(StringRef opName) const {
  return handlers.contains(opName);
}

OpHandlerRegistry &OpHandlerRegistry::getDefault() {
  static OpHandlerRegistry registry;
  return registry;
}

//===----------------------------------------------------------------------===//
// IREE/Stream Dialect Handlers (Unregistered Operations)
//===----------------------------------------------------------------------===//

/// Handle iree_input.binding.subspan or stream.binding.subspan
/// These operations map kernel arguments to buffer descriptors (SRDs)
LogicalResult handleBindingSubspan(Operation *op, TranslationContext &ctx) {
  // This operation connects a kernel argument to a memref
  // The result is typically consumed by a memref.reinterpret_cast
  // We track this for SRD setup during load/store operations

  // Get the binding index from the operation
  // The first operand is a !stream.binding which comes from a function argument
  int64_t bindingIdx = 0;
  if (auto bindingAttr = op->getAttrOfType<IntegerAttr>("binding")) {
    bindingIdx = bindingAttr.getInt();
  } else if (op->getNumOperands() > 0) {
    // Get binding index from the first operand (the !stream.binding argument)
    // If it's a block argument, use its argument number
    if (auto blockArg = dyn_cast<BlockArgument>(op->getOperand(0))) {
      bindingIdx = blockArg.getArgNumber();
    }
  }

  // Store the binding info
  ctx.trackBinding(op->getResult(0), bindingIdx);

  // Try to compute buffer size from result type
  int64_t bufferSize = 512;  // Default buffer size
  if (auto memrefType = dyn_cast<MemRefType>(op->getResult(0).getType())) {
    bufferSize = computeBufferSizeFromMemRef(memrefType);
  }

  // Queue SRD setup for this binding
  ctx.queueSRDSetup(op->getResult(0), bindingIdx, bufferSize);

  return success();
}

/// Handle amdgpu.fat_raw_buffer_cast
/// This operation creates a buffer descriptor with cache swizzle info for gather_to_lds
LogicalResult handleFatRawBufferCast(Operation *op, TranslationContext &ctx) {
  // The operation format is:
  //   %result = amdgpu.fat_raw_buffer_cast %source, %offset, %cacheSwizzleStride
  //
  // For gather_to_lds, we need to create a new SRD with cache swizzle bits:
  //   word0: copy from source SRD word0 (base address low)
  //   word1: (source word1 & 0xffff) | 0x40400000 (cache swizzle bits)
  //   word2: 0x7ffffffd (size for gather)
  //   word3: 0x27000 (format with swizzle, instead of standard 0x20000)

  if (op->getNumOperands() < 1) {
    return success();  // No source operand
  }

  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get source SRD
  auto srcMapped = ctx.getMapper().getMapped(op->getOperand(0));
  if (!srcMapped) {
    // Just pass through if no mapping
    return success();
  }

  // Check if there's a cache swizzle stride (operand 2)
  bool hasCacheSwizzle = false;
  int64_t swizzleStride = 0;
  if (op->getNumOperands() >= 3) {
    if (auto swizzleVal = getConstantValue(op->getOperand(2))) {
      swizzleStride = *swizzleVal;
      hasCacheSwizzle = (swizzleStride > 0);
    }
  }

  if (!hasCacheSwizzle) {
    // No cache swizzle - just pass through
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Create a new SRD with cache swizzle bits
  // Allocate a new SRD slot (use SGPRs after the regular SRDs)
  // The Python backend uses s[24:27] for the first swizzle SRD
  int64_t newSrdBase = ctx.getNextSwizzleSRDIndex();

  // Get source SRD base index
  int64_t srcSrdBase = -1;

  // Check if srcMapped is defined by a PrecoloredSRegOp (which has the physical index)
  if (auto defOp = srcMapped->getDefiningOp()) {
    if (defOp->getName().getStringRef() == "waveasm.precolored.sreg") {
      if (auto indexAttr = defOp->getAttrOfType<IntegerAttr>("index")) {
        srcSrdBase = indexAttr.getInt();
      }
    }
  }

  // Fallback: try PSRegType
  if (srcSrdBase < 0) {
    if (auto psreg = dyn_cast<PSRegType>(srcMapped->getType())) {
      srcSrdBase = psreg.getIndex();
    }
  }

  // Fallback: try getSRDIndex
  if (srcSrdBase < 0) {
    if (auto srcSrdIdx = ctx.getSRDIndex(op->getOperand(0))) {
      srcSrdBase = *srcSrdIdx;
    }
  }

  if (srcSrdBase < 0) {
    // Fallback - just pass through
    ctx.getMapper().mapValue(op->getResult(0), *srcMapped);
    return success();
  }

  // Emit the cache swizzle SRD setup:
  //   s_mov_b32 sN, sSrc       ; word0 - base address low
  //   s_and_b32 sN+1, sSrc+1, 0xffff
  //   s_or_b32 sN+1, sN+1, 0x40400000  ; cache swizzle bits
  //   s_mov_b32 sN+2, 0x7ffffffd  ; word2 - size
  //   s_mov_b32 sN+3, 0x27000     ; word3 - format with swizzle

  // word0: copy base address low
  std::string mov0 = "s_mov_b32 s" + std::to_string(newSrdBase) + ", s" +
                     std::to_string(srcSrdBase);
  RawOp::create(builder, loc, mov0);

  // word1: mask and add cache swizzle bits
  std::string and1 = "s_and_b32 s" + std::to_string(newSrdBase + 1) + ", s" +
                     std::to_string(srcSrdBase + 1) + ", 0xffff";
  RawOp::create(builder, loc, and1);

  std::string or1 = "s_or_b32 s" + std::to_string(newSrdBase + 1) + ", s" +
                    std::to_string(newSrdBase + 1) + ", 0x40400000";
  RawOp::create(builder, loc, or1);

  // word2: size for gather operations
  std::string mov2 = "s_mov_b32 s" + std::to_string(newSrdBase + 2) + ", 0x7ffffffd";
  RawOp::create(builder, loc, mov2);

  // word3: format with swizzle (0x27000 instead of 0x20000)
  std::string mov3 = "s_mov_b32 s" + std::to_string(newSrdBase + 3) + ", 0x27000";
  RawOp::create(builder, loc, mov3);

  // Create precolored SREG for the new SRD and map the result
  auto srdType = ctx.createSRegType(4, 4);
  auto newSrd = PrecoloredSRegOp::create(builder, loc, srdType, newSrdBase, 4);
  ctx.getMapper().mapValue(op->getResult(0), newSrd);

  // Also store the swizzle stride for potential later use
  ctx.setCacheSwizzleStride(op->getResult(0), swizzleStride);

  return success();
}

/// Handle amdgpu.gather_to_lds
/// This is the high-level operation for gathering from global memory to LDS
///
/// Operand layout (from MLIR):
///   operand(0) = source buffer (fat_raw_buffer memref)
///   operand(1) = source index (global memory offset) -> voffset for buffer_load
///   operand(2) = LDS destination memref -> used to get LDS base offset
///   operand(3) = LDS row index -> used to compute m0
///   operand(4) = LDS col index -> used to compute m0
///
/// For buffer_load...lds:
///   - voffset provides the global memory offset
///   - m0 provides the LDS destination offset
LogicalResult handleGatherToLds(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // operand(0): source buffer descriptor
  Value srcBuffer = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(srcBuffer);
  if (!srcMapped) {
    return op->emitError("source buffer not mapped");
  }

  // operand(1): source index -> this is the voffset for global memory access
  Value voff;
  if (op->getNumOperands() > 1) {
    Value srcIndex = op->getOperand(1);
    auto srcIndexMapped = ctx.getMapper().getMapped(srcIndex);
    if (srcIndexMapped) {
      voff = *srcIndexMapped;
      // Ensure it's in a VGPR
      if (!isVGPRType(voff.getType())) {
        auto vregType = ctx.createVRegType();
        voff = V_MOV_B32::create(builder, loc, vregType, voff);
      }
    }
  }
  if (!voff) {
    // Fallback: use zero offset
    auto immType = ctx.createImmType(0);
    auto zeroConst = ConstantOp::create(builder, loc, immType, 0);
    auto vregType = ctx.createVRegType();
    voff = V_MOV_B32::create(builder, loc, vregType, zeroConst);
  }

  // operand(2): LDS destination memref -> get LDS base offset and shape info
  int64_t ldsBaseOffsetConst = 0;
  int64_t ldsRowStride = 0;  // bytes per row in LDS
  bool hasLdsBaseOffset = false;

  if (op->getNumOperands() > 2) {
    mlir::Value ldsMemref = op->getOperand(2);

    // Get the LDS base offset from memref.view
    if (auto baseOffset = ctx.getLDSBaseOffset(ldsMemref)) {
      // Try to extract constant value from the mapped offset
      if (auto immType = dyn_cast<ImmType>(baseOffset->getType())) {
        ldsBaseOffsetConst = immType.getValue();
        hasLdsBaseOffset = true;
      }
    }

    // Get the LDS memref shape to compute row stride
    // The LDS layout is typically [rows, cols] with element type f16 (2 bytes)
    if (auto memrefType = dyn_cast<MemRefType>(ldsMemref.getType())) {
      auto shape = memrefType.getShape();
      int64_t elementBytes = 2;  // Default f16
      if (auto elemType = memrefType.getElementType()) {
        if (elemType.isF16() || elemType.isBF16())
          elementBytes = 2;
        else if (elemType.isF32())
          elementBytes = 4;
      }
      if (shape.size() >= 2) {
        // row stride = number of columns * element bytes
        ldsRowStride = shape[1] * elementBytes;
      }
    }
  }

  // operand(3): LDS row index -> used to compute m0 offset
  // M0 = lds_base + row * row_stride
  int64_t m0Const = ldsBaseOffsetConst;
  bool canUseImmediateM0 = hasLdsBaseOffset;

  if (op->getNumOperands() > 3) {
    mlir::Value ldsRowIndex = op->getOperand(3);
    auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);

    if (ldsRowMapped) {
      // Try to extract constant value from row index
      if (auto immType = dyn_cast<ImmType>(ldsRowMapped->getType())) {
        int64_t rowIdxConst = immType.getValue();
        // Compute m0 = base + row * stride
        if (ldsRowStride > 0) {
          m0Const = ldsBaseOffsetConst + rowIdxConst * ldsRowStride;
        } else {
          // Fallback: assume row index is already in bytes
          m0Const = ldsBaseOffsetConst + rowIdxConst;
        }
        canUseImmediateM0 = true;
      } else {
        // Row index is not a constant, need to compute dynamically
        canUseImmediateM0 = false;
      }
    }
  }

  // Set M0 for LDS destination
  if (canUseImmediateM0) {
    // Use immediate value directly - most efficient
    auto m0Imm = ctx.createImmType(m0Const);
    auto m0ConstVal = ConstantOp::create(builder, loc, m0Imm, m0Const);
    S_MOV_B32_M0::create(builder, loc, m0ConstVal);
  } else {
    // Need to compute m0 dynamically (fallback path)
    Value m0Val;
    if (op->getNumOperands() > 3) {
      mlir::Value ldsRowIndex = op->getOperand(3);
      auto ldsRowMapped = ctx.getMapper().getMapped(ldsRowIndex);
      if (ldsRowMapped) {
        m0Val = *ldsRowMapped;

        // Convert SGPR to VGPR if needed (v_mov_b32 can't take SGPR as source)
        // Use v_add_u32 with zero to broadcast SGPR to VGPR
        auto convertToVgpr = [&](Value val) -> Value {
          if (!isVGPRType(val.getType())) {
            // v_add_u32 can take SGPR as one operand and broadcasts it to all lanes
            auto zeroImm = ctx.createImmType(0);
            auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
            Value zeroVgpr = V_MOV_B32::create(builder, loc, ctx.createVRegType(), zeroConst);
            return V_ADD_U32::create(builder, loc, ctx.createVRegType(), zeroVgpr, val);
          }
          return val;
        };

        // Multiply row index by row stride if needed
        if (ldsRowStride > 1) {
          auto strideImm = ctx.createImmType(ldsRowStride);
          auto strideConst = ConstantOp::create(builder, loc, strideImm, ldsRowStride);
          m0Val = convertToVgpr(m0Val);
          m0Val = V_MUL_LO_U32::create(builder, loc, ctx.createVRegType(), m0Val, strideConst);
        }

        // Add base offset
        if (hasLdsBaseOffset && ldsBaseOffsetConst != 0) {
          auto baseImm = ctx.createImmType(ldsBaseOffsetConst);
          auto baseConst = ConstantOp::create(builder, loc, baseImm, ldsBaseOffsetConst);
          Value baseVgpr = V_MOV_B32::create(builder, loc, ctx.createVRegType(), baseConst);
          m0Val = convertToVgpr(m0Val);
          m0Val = V_ADD_U32::create(builder, loc, ctx.createVRegType(), m0Val, baseVgpr);
        }
      }
    }

    if (m0Val) {
      Value m0Src = m0Val;
      // If source is a VGPR, convert to SGPR using v_readfirstlane_b32
      if (isVGPRType(m0Src.getType())) {
        auto sregType = ctx.createSRegType();
        m0Src = V_READFIRSTLANE_B32::create(builder, loc, sregType, m0Src);
      }
      S_MOV_B32_M0::create(builder, loc, m0Src);
    } else {
      // Fallback: use zero LDS offset
      auto zeroImm = ctx.createImmType(0);
      auto zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);
      S_MOV_B32_M0::create(builder, loc, zeroConst);
    }
  }

  // Emit buffer_load_dword_lds with correct voffset
  auto soffImm = ctx.createImmType(0);
  auto soffConst = ConstantOp::create(builder, loc, soffImm, 0);
  BUFFER_LOAD_DWORD_LDS::create(builder, loc, voff, *srcMapped, soffConst);

  return success();
}

/// Handle amdgpu.raw_buffer_load - direct buffer load
LogicalResult handleRawBufferLoad(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get operands: memref, indices, bounds check, indexOffset
  if (op->getNumOperands() < 1) {
    return op->emitError("raw_buffer_load requires at least a memref operand");
  }

  Value memref = op->getOperand(0);
  auto srcMapped = ctx.getMapper().getMapped(memref);
  if (!srcMapped) {
    return op->emitError("buffer descriptor not mapped");
  }

  // Get offset operand if present
  Value voffset;
  if (op->getNumOperands() > 1) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(1));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  // Determine load width from result type
  Type resultType = op->getResult(0).getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(resultType)) {
    numElements = vecType.getNumElements();
  }

  Operation *loadInstr;
  if (numElements == 1) {
    auto vregType = ctx.createVRegType(1);
    loadInstr = BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregType}, *srcMapped, voffset);
  } else if (numElements == 2) {
    auto vregType = ctx.createVRegType(2, 2);
    loadInstr = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{vregType}, *srcMapped, voffset);
  } else if (numElements == 4) {
    auto vregType = ctx.createVRegType(4, 4);
    loadInstr = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{vregType}, *srcMapped, voffset);
  } else {
    return op->emitError("unsupported buffer load width");
  }

  ctx.getMapper().mapValue(op->getResult(0), loadInstr->getResult(0));
  return success();
}

/// Handle amdgpu.raw_buffer_store - direct buffer store
LogicalResult handleRawBufferStore(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Get operands: value, memref, indices
  if (op->getNumOperands() < 2) {
    return op->emitError("raw_buffer_store requires value and memref operands");
  }

  Value storeValue = op->getOperand(0);
  Value memref = op->getOperand(1);

  auto valueMapped = ctx.getMapper().getMapped(storeValue);
  auto srd = ctx.getMapper().getMapped(memref);

  if (!valueMapped) {
    return op->emitError("store value not mapped");
  }
  if (!srd) {
    return op->emitError("buffer descriptor not mapped");
  }

  // Get offset operand if present
  Value voffset;
  if (op->getNumOperands() > 2) {
    auto voffMapped = ctx.getMapper().getMapped(op->getOperand(2));
    if (voffMapped) {
      voffset = *voffMapped;
    }
  }
  if (!voffset) {
    auto immZero = ctx.createImmType(0);
    voffset = ConstantOp::create(builder, loc, immZero, 0);
  }

  // Determine store width from value type
  Type valueType = storeValue.getType();
  int64_t numElements = 1;
  if (auto vecType = dyn_cast<VectorType>(valueType)) {
    numElements = vecType.getNumElements();
  }

  // Use signature: (data, srd, voffset)
  if (numElements == 1) {
    BUFFER_STORE_DWORD::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 2) {
    BUFFER_STORE_DWORDX2::create(builder, loc, *valueMapped, *srd, voffset);
  } else if (numElements == 4) {
    BUFFER_STORE_DWORDX4::create(builder, loc, *valueMapped, *srd, voffset);
  } else {
    return op->emitError("unsupported buffer store width");
  }

  return success();
}

/// Handle rocdl.readfirstlane
LogicalResult handleReadFirstLane(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto sregType = ctx.createSRegType();

  if (op->getNumOperands() < 1) {
    return op->emitError("readfirstlane requires an operand");
  }

  auto src = ctx.getMapper().getMapped(op->getOperand(0));
  if (!src) {
    return op->emitError("source operand not mapped");
  }

  auto result = V_READFIRSTLANE_B32::create(builder, loc, sregType, *src);
  ctx.getMapper().mapValue(op->getResult(0), result);

  return success();
}

/// Handle rocdl.s.waitcnt
LogicalResult handleSWaitcnt(Operation *op, TranslationContext &ctx) {
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();

  // Extract waitcnt bitfield from operation
  // GFX9+ encoding:
  //   bits 3:0 = vmcnt[3:0]
  //   bits 6:4 = expcnt
  //   bits 11:8 = lgkmcnt[3:0]
  //   bits 15:14 = vmcnt[5:4] (high bits)
  //
  // A value of 15 (0xF) for lgkmcnt means "don't wait" (max outstanding)
  // A value of 63 (0x3F) for vmcnt means "don't wait" (6-bit counter on GFX9+)
  int64_t vmcnt = -1, lgkmcnt = -1;  // -1 means "not specified"

  if (auto bitfieldAttr = op->getAttrOfType<IntegerAttr>("bitfield")) {
    int64_t bitfield = bitfieldAttr.getInt();
    // Decode vmcnt (6 bits: low 4 bits + high 2 bits)
    int64_t vmcnt_lo = bitfield & 0xF;
    int64_t vmcnt_hi = (bitfield >> 14) & 0x3;
    int64_t vmcnt_full = vmcnt_lo | (vmcnt_hi << 4);
    // Decode lgkmcnt (4 bits)
    int64_t lgkmcnt_full = (bitfield >> 8) & 0xF;

    // Only emit wait if not max value (max = "don't wait")
    if (vmcnt_full < 63) {  // 63 = 0x3F = max vmcnt
      vmcnt = vmcnt_full;
    }
    if (lgkmcnt_full < 15) {  // 15 = 0xF = max lgkmcnt
      lgkmcnt = lgkmcnt_full;
    }
  }

  // If neither wait is needed, skip emitting
  if (vmcnt < 0 && lgkmcnt < 0) {
    return success();
  }

  // Create the appropriate waitcnt
  if (vmcnt >= 0 && lgkmcnt >= 0) {
    auto vmcntAttr = builder.getI32IntegerAttr(vmcnt);
    auto lgkmcntAttr = builder.getI32IntegerAttr(lgkmcnt);
    S_WAITCNT::create(builder, loc, vmcntAttr, lgkmcntAttr, IntegerAttr());
  } else if (vmcnt >= 0) {
    S_WAITCNT_VMCNT::create(builder, loc, vmcnt);
  } else {
    S_WAITCNT_LGKMCNT::create(builder, loc, lgkmcnt);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Handler Registration
//===----------------------------------------------------------------------===//

void OpHandlerRegistry::registerDefaultHandlers() {
  // GPU dialect
  registerHandler("gpu.thread_id", handleGPUThreadId);
  registerHandler("gpu.block_id", handleGPUBlockId);
  registerHandler("gpu.barrier", handleGPUBarrier);
  registerHandler("gpu.block_dim", handleGPUBlockDim);
  registerHandler("gpu.grid_dim", handleGPUGridDim);
  registerHandler("gpu.lane_id", handleGPULaneId);
  registerHandler("gpu.subgroup_broadcast", handleGPUSubgroupBroadcast);

  // Arith dialect - basic operations
  registerHandler("arith.constant", handleArithConstant);
  registerHandler("arith.addi", handleArithAddI);
  registerHandler("arith.subi", handleArithSubI);
  registerHandler("arith.muli", handleArithMulI);
  registerHandler("arith.divui", handleArithDivUI);
  registerHandler("arith.remui", handleArithRemUI);
  registerHandler("arith.index_cast", handleArithIndexCast);

  // Arith dialect - bitwise operations
  registerHandler("arith.andi", handleArithAndI);
  registerHandler("arith.ori", handleArithOrI);
  registerHandler("arith.xori", handleArithXorI);

  // Arith dialect - shift operations
  registerHandler("arith.shli", handleArithShLI);
  registerHandler("arith.shrui", handleArithShRUI);
  registerHandler("arith.shrsi", handleArithShRSI);

  // Arith dialect - type conversions
  registerHandler("arith.extui", handleArithExtUI);
  registerHandler("arith.extsi", handleArithExtSI);
  registerHandler("arith.trunci", handleArithTruncI);

  // Arith dialect - comparison and select
  registerHandler("arith.cmpi", handleArithCmpI);
  registerHandler("arith.select", handleArithSelect);

  // Arith dialect - floating-point operations
  registerHandler("arith.addf", handleArithAddF);
  registerHandler("arith.subf", handleArithSubF);
  registerHandler("arith.mulf", handleArithMulF);
  registerHandler("arith.divf", handleArithDivF);
  registerHandler("arith.negf", handleArithNegF);
  registerHandler("arith.cmpf", handleArithCmpF);

  // Math dialect
  registerHandler("math.fma", handleMathFma);

  // Affine dialect
  registerHandler("affine.apply", handleAffineApply);

  // Vector dialect
  registerHandler("vector.load", handleVectorLoad);
  registerHandler("vector.store", handleVectorStore);
  registerHandler("vector.extract_strided_slice", handleVectorExtractStridedSlice);
  registerHandler("vector.broadcast", handleVectorBroadcast);
  registerHandler("vector.extract", handleVectorExtract);
  registerHandler("vector.insert", handleVectorInsert);
  registerHandler("vector.shape_cast", handleVectorShapeCast);
  registerHandler("vector.transfer_read", handleVectorTransferRead);
  registerHandler("vector.transfer_write", handleVectorTransferWrite);
  registerHandler("vector.fma", handleVectorFma);
  registerHandler("vector.reduction", handleVectorReduction);

  // AMDGPU dialect
  registerHandler("amdgpu.lds_barrier", handleAMDGPULdsBarrier);
  registerHandler("amdgpu.mfma", handleAMDGPUMfma);
  registerHandler("amdgpu.fat_raw_buffer_cast", handleFatRawBufferCast);
  registerHandler("amdgpu.gather_to_lds", handleGatherToLds);
  registerHandler("amdgpu.raw_buffer_load", handleRawBufferLoad);
  registerHandler("amdgpu.raw_buffer_store", handleRawBufferStore);

  // ROCDL dialect
  registerHandler("rocdl.readfirstlane", handleReadFirstLane);
  registerHandler("rocdl.s.waitcnt", handleSWaitcnt);

  // IREE/Stream dialect (unregistered operations)
  registerHandler("stream.binding.subspan", handleBindingSubspan);
  registerHandler("iree_input.binding.subspan", handleBindingSubspan);

  // MemRef dialect
  registerHandler("memref.alloc", handleMemRefAlloc);
  registerHandler("memref.view", handleMemRefView);
  registerHandler("memref.reinterpret_cast", handleMemRefReinterpretCast);
  registerHandler("memref.subview", handleMemRefSubView);
  registerHandler("memref.load", handleMemRefLoad);
  registerHandler("memref.store", handleMemRefStore);
  registerHandler("memref.cast", handleMemRefCast);

  // SCF dialect
  registerHandler("scf.for", handleSCFFor);
  registerHandler("scf.if", handleSCFIf);
  registerHandler("scf.yield", handleSCFYield);
}

//===----------------------------------------------------------------------===//
// Translation Functions
//===----------------------------------------------------------------------===//

LogicalResult translateOperation(Operation *op, TranslationContext &ctx) {
  StringRef opName = op->getName().getStringRef();

  // Skip terminators (handled by parent)
  if (op->hasTrait<OpTrait::IsTerminator>())
    return success();

  // Look up handler
  auto &registry = OpHandlerRegistry::getDefault();
  if (auto handler = registry.getHandler(opName)) {
    return (*handler)(op, ctx);
  }

  // No handler - emit comment for debugging
  LLVM_DEBUG(llvm::dbgs() << "No handler for: " << opName << "\n");
  ctx.emitComment(("unhandled: " + opName).str());
  return success();
}

LogicalResult translateModule(ModuleOp module, StringRef targetId) {
  // Get target
  auto target = getTargetKindAttr(module.getContext(), targetId);
  if (!target) {
    return module.emitError() << "Unknown target: " << targetId;
  }

  // Collect GPU functions to translate (can't erase during walk)
  SmallVector<gpu::GPUFuncOp> gpuFuncsToTranslate;
  module.walk([&](gpu::GPUFuncOp gpuFunc) {
    if (!gpuFunc.isDeclaration())
      gpuFuncsToTranslate.push_back(gpuFunc);
  });

  // Translate each GPU function
  for (auto gpuFunc : gpuFuncsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create a program for this GPU function
    auto program = createProgramFromGPUFunc(gpuFunc, builder, targetId);
    if (!program)
      continue;

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);

    // Map function arguments
    for (auto arg : gpuFunc.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      // Check if argument is a memref type
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        // Queue SRD setup for this binding
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        ctx.queueSRDSetup(arg, argIdx, bufferSize);
      } else {
        // Non-memref args (i32, index, etc.) - map to VGPR
        auto vregType = ctx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, gpuFunc.getLoc(), vregType, argIdx, 1);
        ctx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations to queue SRD setups
    for (Operation &op : gpuFunc.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, ctx);
      }
    }

    // Emit SRD prologue (s_load, s_waitcnt, s_mov instructions)
    ctx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : gpuFunc.getBody().front()) {
      if (failed(translateOperation(&op, ctx)))
        return failure();
    }

    // Emit endpgm
    S_ENDPGM::create(builder, gpuFunc.getLoc());

    // Set the number of kernel arguments attribute on the program
    size_t numKernelArgs = ctx.getNumKernelArgs();
    program->setAttr("num_kernel_args",
                     builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));

    // Set the LDS size attribute if any LDS was allocated
    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase the original GPU function to avoid symbol collision
    gpuFunc.erase();
  }

  // Clean up empty gpu.module containers
  SmallVector<gpu::GPUModuleOp> gpuModulesToErase;
  module.walk([&](gpu::GPUModuleOp gpuModule) {
    // Check if the module is now empty (only contains gpu.module_end terminator)
    auto *body = gpuModule.getBody();
    if (body) {
      // If only one op (the terminator) remains, erase the module
      if (body->getOperations().size() <= 1)
        gpuModulesToErase.push_back(gpuModule);
    }
  });
  for (auto gpuModule : gpuModulesToErase)
    gpuModule.erase();

  // Collect func.func ops to translate (can't erase during walk)
  SmallVector<func::FuncOp> funcsToTranslate;
  module.walk([&](func::FuncOp funcOp) {
    // Skip if already handled as GPU func
    if (funcOp->getParentOfType<gpu::GPUModuleOp>())
      return;
    // Skip function declarations (no body)
    if (funcOp.isDeclaration())
      return;
    funcsToTranslate.push_back(funcOp);
  });

  // Translate each func.func
  for (auto funcOp : funcsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create program from func.func
    auto program = createProgramFromFunc(funcOp, builder, targetId);
    if (!program)
      continue;

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    // Create target instance and keep it alive for the duration of ctx
    auto targetInstance = getTargetKindAttr(module.getContext(), targetId);
    TranslationContext ctx(builder, program, targetInstance);

    // Map function arguments
    for (auto arg : funcOp.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      // Check if argument is a memref type
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        // Queue SRD setup for this binding
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        ctx.queueSRDSetup(arg, argIdx, bufferSize);
      } else {
        // Non-memref args (i32, index, etc.) - map to VGPR
        auto vregType = ctx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, funcOp.getLoc(), vregType, argIdx, 1);
        ctx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations to queue SRD setups
    for (Operation &op : funcOp.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, ctx);
      }
    }

    // Emit SRD prologue (s_load, s_waitcnt, s_mov instructions)
    ctx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : funcOp.getBody().front()) {
      (void)translateOperation(&op, ctx);
    }

    // Emit endpgm
    S_ENDPGM::create(builder, funcOp.getLoc());

    // Set the number of kernel arguments attribute on the program
    size_t numKernelArgs = ctx.getNumKernelArgs();
    program->setAttr("num_kernel_args",
                     builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));

    // Set the LDS size attribute if any LDS was allocated
    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase the original func.func to avoid symbol name collision
    funcOp.erase();
  }

  return success();
}

LogicalResult translateModule(ModuleOp module, const TranslationOptions &options) {
  // Get target
  auto target = getTargetKindAttr(module.getContext(), options.targetId);
  if (!target) {
    return module.emitError() << "Unknown target: " << options.targetId;
  }

  // Collect func.func ops to translate (can't erase during walk)
  SmallVector<func::FuncOp> funcsToTranslate;
  module.walk([&](func::FuncOp funcOp) {
    // Skip function declarations (no body)
    if (funcOp.isDeclaration())
      return;
    funcsToTranslate.push_back(funcOp);
  });

  // Translate each func.func
  for (auto funcOp : funcsToTranslate) {
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    // Create target attribute
    auto *ctx = builder.getContext();
    auto loc = funcOp.getLoc();
    auto targetAttr =
        TargetAttr::get(ctx, getTargetKindAttr(ctx, options.targetId), 5);

    // Create ABI attribute
    auto abiAttr = KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

    // Determine workgroup size - use explicit options if provided
    ArrayAttr workgroupSizeAttr;
    if (options.hasExplicitWorkgroupSize()) {
      auto [wgX, wgY, wgZ] = options.getWorkgroupSize();
      SmallVector<Attribute, 3> sizes = {
          builder.getI64IntegerAttr(wgX),
          builder.getI64IntegerAttr(wgY),
          builder.getI64IntegerAttr(wgZ)
      };
      workgroupSizeAttr = builder.getArrayAttr(sizes);
    } else {
      // Fall back to extraction from MLIR (translation_info or gpu.thread_id)
      if (auto translationInfo = funcOp->getAttr("translation_info")) {
        // Parse workgroup_size from translation_info
        std::string attrStr;
        llvm::raw_string_ostream os(attrStr);
        translationInfo.print(os);
        os.flush();
        auto pos = attrStr.find("workgroup_size");
        if (pos != std::string::npos) {
          auto bracketStart = attrStr.find('[', pos);
          auto bracketEnd = attrStr.find(']', bracketStart);
          if (bracketStart != std::string::npos && bracketEnd != std::string::npos) {
            std::string arrayStr = attrStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
            SmallVector<Attribute, 3> sizes;
            std::stringstream ss(arrayStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
              size_t start = token.find_first_not_of(" \t");
              size_t end = token.find_last_not_of(" \t");
              if (start != std::string::npos && end != std::string::npos) {
                std::string numStr = token.substr(start, end - start + 1);
                bool isValid = !numStr.empty();
                for (size_t i = 0; i < numStr.size(); ++i) {
                  char c = numStr[i];
                  if (i == 0 && c == '-') continue;
                  if (!std::isdigit(static_cast<unsigned char>(c))) {
                    isValid = false;
                    break;
                  }
                }
                if (isValid) {
                  int64_t val = std::stoll(numStr);
                  sizes.push_back(builder.getI64IntegerAttr(val));
                }
              }
            }
            if (sizes.size() >= 3) {
              workgroupSizeAttr = builder.getArrayAttr(sizes);
            }
          }
        }
      }
      // Try gpu.thread_id upper_bound if still not found
      if (!workgroupSizeAttr) {
        int64_t wgSizeX = 64, wgSizeY = 1, wgSizeZ = 1;
        funcOp.walk([&](gpu::ThreadIdOp threadIdOp) {
          if (auto upperBoundAttr = threadIdOp->getAttrOfType<IntegerAttr>("upper_bound")) {
            int64_t bound = upperBoundAttr.getInt();
            switch (threadIdOp.getDimension()) {
              case gpu::Dimension::x: wgSizeX = bound; break;
              case gpu::Dimension::y: wgSizeY = bound; break;
              case gpu::Dimension::z: wgSizeZ = bound; break;
            }
          }
        });
        SmallVector<Attribute, 3> sizes = {
            builder.getI64IntegerAttr(wgSizeX),
            builder.getI64IntegerAttr(wgSizeY),
            builder.getI64IntegerAttr(wgSizeZ)
        };
        workgroupSizeAttr = builder.getArrayAttr(sizes);
      }
    }

    // Create program with workgroup size
    auto program = ProgramOp::create(builder, loc, funcOp.getName(), targetAttr, abiAttr,
        /*vgprs=*/int64_t{256},
        /*sgprs=*/int64_t{104},
        /*workgroup_size=*/workgroupSizeAttr,
        /*lds_size=*/IntegerAttr{});

    if (program.getBody().empty())
      program.getBody().emplaceBlock();

    // Set up translation context
    builder.setInsertionPointToStart(&program.getBodyBlock());
    auto targetInstance = getTargetKindAttr(ctx, options.targetId);
    TranslationContext transCtx(builder, program, targetInstance);

    // Map function arguments
    for (auto arg : funcOp.getBody().getArguments()) {
      int64_t argIdx = arg.getArgNumber();
      if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
        int64_t bufferSize = computeBufferSizeFromMemRef(memrefType);
        transCtx.queueSRDSetup(arg, argIdx, bufferSize);
      } else {
        auto vregType = transCtx.createVRegType();
        auto vreg = PrecoloredVRegOp::create(builder, funcOp.getLoc(), vregType, argIdx, 1);
        transCtx.getMapper().mapValue(arg, vreg);
      }
    }

    // First pass: handle binding.subspan operations
    for (Operation &op : funcOp.getBody().front()) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "stream.binding.subspan" ||
          opName == "iree_input.binding.subspan" ||
          opName == "memref.reinterpret_cast") {
        (void)translateOperation(&op, transCtx);
      }
    }

    // Emit SRD prologue
    transCtx.emitSRDPrologue();

    // Second pass: translate all operations
    for (Operation &op : funcOp.getBody().front()) {
      (void)translateOperation(&op, transCtx);
    }

    // Emit endpgm
    S_ENDPGM::create(builder, funcOp.getLoc());

    // Set kernel arguments count
    size_t numKernelArgs = transCtx.getNumKernelArgs();
    program->setAttr("num_kernel_args",
                     builder.getI64IntegerAttr(static_cast<int64_t>(numKernelArgs)));

    // Set LDS size if used
    int64_t ldsSize = transCtx.getTotalLDSSize();
    if (ldsSize > 0) {
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
    }

    // Erase original function
    funcOp.erase();
  }

  return success();
}

ProgramOp createProgramFromGPUFunc(gpu::GPUFuncOp gpuFunc,
                                    OpBuilder &builder,
                                    StringRef targetId) {
  auto *ctx = builder.getContext();
  auto loc = gpuFunc.getLoc();

  // Create target attribute
  auto targetAttr = TargetAttr::get(ctx, getTargetKindAttr(ctx, targetId), 5);

  // Create ABI attribute with default bindings
  auto abiAttr = KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

  // Create program
  auto program = ProgramOp::create(builder, loc, gpuFunc.getName(), targetAttr, abiAttr,
      /*vgprs=*/int64_t{256},
      /*sgprs=*/int64_t{104},
      /*workgroup_size=*/ArrayAttr{},
      /*lds_size=*/IntegerAttr{});

  // Ensure the body region has a block
  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

ProgramOp createProgramFromFunc(func::FuncOp funcOp,
                                 OpBuilder &builder,
                                 StringRef targetId) {
  auto *ctx = builder.getContext();
  auto loc = funcOp.getLoc();

  // Create target attribute
  auto targetAttr = TargetAttr::get(ctx, getTargetKindAttr(ctx, targetId), 5);

  // Create ABI attribute
  auto abiAttr = KernelABIAttr::get(ctx, 0, 0, std::nullopt, std::nullopt, std::nullopt);

  // Try to extract workgroup size from translation_info attribute
  // The attribute looks like: #iree_codegen.translation_info<... workgroup_size = [64, 1, 1] ...>
  // When parsed with unregistered dialects, it becomes an opaque attribute that we need
  // to parse from its string representation.
  ArrayAttr workgroupSizeAttr;
  if (auto translationInfo = funcOp->getAttr("translation_info")) {
    // Try as dictionary first (in case it's a registered dialect)
    if (auto dictAttr = dyn_cast<DictionaryAttr>(translationInfo)) {
      if (auto wgSize = dictAttr.get("workgroup_size")) {
        workgroupSizeAttr = dyn_cast<ArrayAttr>(wgSize);
      }
    }
    // If not a dictionary, parse from string representation
    if (!workgroupSizeAttr) {
      std::string attrStr;
      llvm::raw_string_ostream os(attrStr);
      translationInfo.print(os);
      os.flush();

      // Parse "workgroup_size = [X, Y, Z]" from the string
      // Look for pattern: workgroup_size = [num, num, num]
      auto pos = attrStr.find("workgroup_size");
      if (pos != std::string::npos) {
        auto bracketStart = attrStr.find('[', pos);
        auto bracketEnd = attrStr.find(']', bracketStart);
        if (bracketStart != std::string::npos && bracketEnd != std::string::npos) {
          std::string arrayStr = attrStr.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
          SmallVector<Attribute, 3> sizes;
          std::stringstream ss(arrayStr);
          std::string token;
          while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos) {
              std::string numStr = token.substr(start, end - start + 1);
              // Check if it's a valid integer
              bool isValid = !numStr.empty();
              for (size_t i = 0; i < numStr.size(); ++i) {
                char c = numStr[i];
                if (i == 0 && c == '-') continue;  // Allow leading minus
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                  isValid = false;
                  break;
                }
              }
              if (isValid) {
                int64_t val = std::stoll(numStr);
                sizes.push_back(builder.getI64IntegerAttr(val));
              }
            }
          }
          if (sizes.size() >= 3) {
            workgroupSizeAttr = builder.getArrayAttr(sizes);
          }
        }
      }
    }
  }

  // If no translation_info, try to extract from gpu.thread_id upper_bound attrs
  if (!workgroupSizeAttr) {
    int64_t wgSizeX = 64, wgSizeY = 1, wgSizeZ = 1;  // defaults
    funcOp.walk([&](gpu::ThreadIdOp threadIdOp) {
      if (auto upperBoundAttr = threadIdOp->getAttrOfType<IntegerAttr>("upper_bound")) {
        int64_t bound = upperBoundAttr.getInt();
        switch (threadIdOp.getDimension()) {
          case gpu::Dimension::x: wgSizeX = bound; break;
          case gpu::Dimension::y: wgSizeY = bound; break;
          case gpu::Dimension::z: wgSizeZ = bound; break;
        }
      }
    });
    SmallVector<Attribute, 3> sizes = {
        builder.getI64IntegerAttr(wgSizeX),
        builder.getI64IntegerAttr(wgSizeY),
        builder.getI64IntegerAttr(wgSizeZ)
    };
    workgroupSizeAttr = builder.getArrayAttr(sizes);
  }

  // Create program
  auto program = ProgramOp::create(builder, loc, funcOp.getName(), targetAttr, abiAttr,
      /*vgprs=*/int64_t{256},
      /*sgprs=*/int64_t{104},
      /*workgroup_size=*/workgroupSizeAttr,
      /*lds_size=*/IntegerAttr{});

  // Ensure the body region has a block
  if (program.getBody().empty())
    program.getBody().emplaceBlock();

  return program;
}

} // namespace waveasm
