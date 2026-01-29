// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H
#define WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <optional>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Value Mapping (Pure SSA)
//===----------------------------------------------------------------------===//

/// Maps MLIR values from input dialect to waveasm SSA values
class ValueMapper {
public:
  ValueMapper() = default;

  /// Map an MLIR value to an waveasm value
  void mapValue(mlir::Value inputValue, mlir::Value asmValue) {
    valueMap[inputValue] = asmValue;
  }

  /// Get the mapped value for an MLIR value
  std::optional<mlir::Value> getMapped(mlir::Value mlirValue) const {
    auto it = valueMap.find(mlirValue);
    if (it != valueMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if a value is mapped
  bool isMapped(mlir::Value mlirValue) const {
    return valueMap.contains(mlirValue);
  }

private:
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
};

//===----------------------------------------------------------------------===//
// Expression Key for CSE in Expression Emitter
//===----------------------------------------------------------------------===//

/// Key for identifying equivalent expressions during translation
struct ExprKey {
  llvm::StringRef opName;
  llvm::SmallVector<mlir::Value, 4> operands;
  llvm::SmallVector<int64_t, 2> constants;

  bool operator==(const ExprKey &rhs) const {
    return opName == rhs.opName && operands == rhs.operands &&
           constants == rhs.constants;
  }
};

//===----------------------------------------------------------------------===//
// SRD (Shader Resource Descriptor) Info
//===----------------------------------------------------------------------===//

/// Information about a buffer's SRD setup
struct SRDInfo {
  mlir::Value baseSGPR;      // SGPR pair holding base address
  mlir::Value srdSGPR;       // SGPR quad holding full SRD
  int64_t bindingIndex;      // Which kernel argument this refers to
  int64_t bufferSize;        // Buffer size in bytes (for bounds checking)
  int64_t elementSize;       // Element size in bytes

  SRDInfo() : bindingIndex(-1), bufferSize(0), elementSize(4) {}
  SRDInfo(mlir::Value base, mlir::Value srd, int64_t binding, int64_t size, int64_t elemSize)
      : baseSGPR(base), srdSGPR(srd), bindingIndex(binding),
        bufferSize(size), elementSize(elemSize) {}
};

//===----------------------------------------------------------------------===//
// Bit Range for OR vs ADD optimization
//===----------------------------------------------------------------------===//

/// Represents a range of bits [lowBit, highBit] that an expression can occupy.
/// Used to determine if two additions can be fused into OR when bits don't overlap.
struct BitRange {
  int64_t lowBit;   // Lowest bit index used (0-based)
  int64_t highBit;  // Highest bit index used (0-based)

  BitRange() : lowBit(0), highBit(31) {}  // Default: full 32-bit range
  BitRange(int64_t low, int64_t high) : lowBit(low), highBit(high) {}

  /// Check if this range overlaps with another
  bool overlaps(const BitRange &other) const {
    return highBit >= other.lowBit && other.highBit >= lowBit;
  }

  /// Merge two non-overlapping ranges (for OR result)
  BitRange merge(const BitRange &other) const {
    return BitRange(std::min(lowBit, other.lowBit),
                    std::max(highBit, other.highBit));
  }

  /// Extend range for ADD result (accounts for carry)
  BitRange extendForAdd(const BitRange &other) const {
    return BitRange(std::min(lowBit, other.lowBit),
                    std::max(highBit, other.highBit) + 1);
  }

  /// Shift range left by n bits
  BitRange shiftLeft(int64_t n) const {
    return BitRange(lowBit + n, highBit + n);
  }

  /// Shift range right by n bits
  BitRange shiftRight(int64_t n) const {
    return BitRange(std::max(int64_t(0), lowBit - n),
                    std::max(int64_t(0), highBit - n));
  }

  /// Create range from a constant value
  static BitRange fromConstant(int64_t value) {
    if (value == 0) return BitRange(0, 0);
    int64_t bits = 0;
    int64_t v = value;
    while (v > 0) { bits++; v >>= 1; }
    return BitRange(0, bits - 1);
  }

  /// Create range for a value with known max
  static BitRange fromMaxValue(int64_t maxValue) {
    if (maxValue <= 0) return BitRange(0, 0);
    int64_t bits = 0;
    int64_t v = maxValue;
    while (v > 0) { bits++; v >>= 1; }
    return BitRange(0, bits - 1);
  }
};

//===----------------------------------------------------------------------===//
// Loop Context for scf.for with iter_args
//===----------------------------------------------------------------------===//

/// Context for tracking loop state during translation
struct LoopContext {
  mlir::Value inductionVar;           // Current loop counter
  llvm::SmallVector<mlir::Value, 4> iterArgs;  // Loop-carried values
  llvm::StringRef labelName;          // Loop label for branching
  int64_t depth;                      // Nesting depth

  LoopContext() : depth(0) {}
};

//===----------------------------------------------------------------------===//
// Translation Context (Pure SSA)
//===----------------------------------------------------------------------===//

/// Context for translating MLIR to waveasm pure SSA IR
class TranslationContext {
public:
  TranslationContext(mlir::OpBuilder &builder, ProgramOp program,
                     TargetAttrInterface target);

  /// Get the MLIR builder
  mlir::OpBuilder &getBuilder() { return builder; }

  /// Get the target program
  ProgramOp getProgram() { return program; }

  /// Get the target architecture
  TargetAttrInterface getTarget() { return target; }

  /// Get the value mapper
  ValueMapper &getMapper() { return mapper; }

  /// Create a virtual VGPR type
  VRegType createVRegType(int64_t size = 1, int64_t alignment = 1);

  /// Create a virtual SGPR type
  SRegType createSRegType(int64_t size = 1, int64_t alignment = 1);

  /// Create an immediate type with a value
  ImmType createImmType(int64_t value);

  /// Emit a label
  LabelOp emitLabel(llvm::StringRef name);

  /// Emit a comment
  CommentOp emitComment(llvm::StringRef text);

  /// Emit a raw assembly line
  RawOp emitRaw(llvm::StringRef code);

  /// Get the current insertion point
  mlir::Block::iterator getInsertionPoint();

  /// Track binding information for IREE stream.binding.subspan
  void trackBinding(mlir::Value value, int64_t bindingIndex) {
    bindingMap[value] = bindingIndex;
  }

  /// Get binding index for a value
  std::optional<int64_t> getBindingIndex(mlir::Value value) const {
    auto it = bindingMap.find(value);
    if (it != bindingMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Set cache swizzle stride for a buffer
  void setCacheSwizzleStride(mlir::Value value, int64_t stride) {
    cacheSwizzleMap[value] = stride;
  }

  /// Get cache swizzle stride for a buffer
  std::optional<int64_t> getCacheSwizzleStride(mlir::Value value) const {
    auto it = cacheSwizzleMap.find(value);
    if (it != cacheSwizzleMap.end())
      return it->second;
    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // SRD Management
  //===--------------------------------------------------------------------===//

  /// Register an SRD for a memref binding
  void registerSRD(mlir::Value memref, const SRDInfo &info) {
    srdMap[memref] = info;
  }

  /// Get SRD info for a memref
  std::optional<SRDInfo> getSRD(mlir::Value memref) const {
    auto it = srdMap.find(memref);
    if (it != srdMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if SRD is already set up for a memref
  bool hasSRD(mlir::Value memref) const {
    return srdMap.contains(memref);
  }

  //===--------------------------------------------------------------------===//
  // Loop Context Management
  //===--------------------------------------------------------------------===//

  /// Push a new loop context
  void pushLoopContext(const LoopContext &ctx) {
    loopStack.push_back(ctx);
  }

  /// Pop the current loop context
  void popLoopContext() {
    if (!loopStack.empty())
      loopStack.pop_back();
  }

  /// Get the current loop context (innermost loop)
  std::optional<LoopContext> getCurrentLoop() const {
    if (loopStack.empty())
      return std::nullopt;
    return loopStack.back();
  }

  /// Get loop nesting depth
  int64_t getLoopDepth() const { return loopStack.size(); }

  //===--------------------------------------------------------------------===//
  // Expression CSE (Common Subexpression Elimination)
  //===--------------------------------------------------------------------===//

  /// Check if an expression result is already cached
  std::optional<mlir::Value> getCachedExpr(llvm::StringRef opName,
                                            mlir::ValueRange operands,
                                            llvm::ArrayRef<int64_t> constants = {}) const;

  /// Cache an expression result
  void cacheExpr(llvm::StringRef opName, mlir::ValueRange operands,
                  llvm::ArrayRef<int64_t> constants, mlir::Value result);

  /// Clear expression cache (e.g., at loop boundaries)
  void clearExprCache() { exprCache.clear(); }

  /// Generate a unique label name
  std::string generateLabel(llvm::StringRef prefix);

  //===--------------------------------------------------------------------===//
  // Pending SRD Setup (Deferred to Prologue)
  //===--------------------------------------------------------------------===//

  /// Information about a pending SRD setup
  struct PendingSRD {
    mlir::Value memref;      // The memref value this SRD is for
    int64_t argIndex;        // Which kernel argument (0, 1, 2, ...)
    int64_t bufferSize;      // Size in bytes for SRD[2]
    int64_t srdBaseIndex;    // SGPR index for SRD (e.g., 8 for s[8:11])
  };

  /// Queue an SRD setup for a binding
  void queueSRDSetup(mlir::Value memref, int64_t argIndex, int64_t bufferSize);

  /// Emit all pending SRD setup instructions (called at start of kernel body)
  void emitSRDPrologue();

  /// Get the SRD SGPR index for a memref
  std::optional<int64_t> getSRDIndex(mlir::Value memref) const;

  /// Set SRD base index for a memref
  void setSRDIndex(mlir::Value memref, int64_t srdBaseIndex);

  /// Get next available SRD index
  int64_t getNextSRDIndex();

  /// Get next available swizzle SRD index (for cache swizzle SRDs)
  /// These are allocated after the regular SRDs, starting at s24
  int64_t getNextSwizzleSRDIndex() {
    int64_t idx = nextSwizzleSRDIndex;
    nextSwizzleSRDIndex += 4;  // Each SRD uses 4 consecutive SGPRs
    return idx;
  }

  /// Update buffer size for a pending SRD (called when we see reinterpret_cast)
  void updateSRDBufferSize(mlir::Value memref, int64_t bufferSize);

  /// Get the number of kernel arguments (based on pending SRD count)
  size_t getNumKernelArgs() const { return pendingSRDs.size(); }

  //===--------------------------------------------------------------------===//
  // Split Vector Result Tracking
  //===--------------------------------------------------------------------===//

  /// Register split results from a vector load (for later use in stores)
  void registerSplitResults(mlir::Value originalValue,
                            llvm::ArrayRef<mlir::Value> splitResults) {
    for (auto result : splitResults) {
      splitResultsMap[originalValue].push_back(result);
    }
  }

  /// Get split results for a value (returns empty if not a split vector)
  llvm::ArrayRef<mlir::Value> getSplitResults(mlir::Value value) const {
    auto it = splitResultsMap.find(value);
    if (it != splitResultsMap.end())
      return it->second;
    return {};
  }

  /// Check if a value has split results
  bool hasSplitResults(mlir::Value value) const {
    return splitResultsMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Thread ID Upper Bound Tracking (for affine simplification)
  //===--------------------------------------------------------------------===//

  /// Register thread ID upper bound for a value
  void setThreadIdUpperBound(mlir::Value value, int64_t upperBound) {
    threadIdUpperBounds[value] = upperBound;
  }

  /// Get thread ID upper bound for a value (returns 0 if not tracked)
  int64_t getThreadIdUpperBound(mlir::Value value) const {
    auto it = threadIdUpperBounds.find(value);
    if (it != threadIdUpperBounds.end())
      return it->second;
    return 0;
  }

  /// Check if a value is a thread ID with known upper bound
  bool hasThreadIdUpperBound(mlir::Value value) const {
    return threadIdUpperBounds.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Bit Range Tracking (for OR vs ADD optimization)
  //===--------------------------------------------------------------------===//

  /// Set bit range for a value
  void setBitRange(mlir::Value value, const BitRange &range) {
    bitRangeMap[value] = range;
  }

  /// Get bit range for a value (returns full 32-bit range if not tracked)
  BitRange getBitRange(mlir::Value value) const {
    auto it = bitRangeMap.find(value);
    if (it != bitRangeMap.end())
      return it->second;
    return BitRange();  // Default: full 32-bit range (conservative)
  }

  /// Check if a value has a tracked bit range
  bool hasBitRange(mlir::Value value) const {
    return bitRangeMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Constant Offset Tracking (for buffer store offset:N optimization)
  //===--------------------------------------------------------------------===//

  /// Set constant offset for a value (for use in buffer store offset:N)
  void setConstOffset(mlir::Value value, int64_t offset) {
    constOffsetMap[value] = offset;
  }

  /// Get constant offset for a value (returns 0 if not tracked)
  int64_t getConstOffset(mlir::Value value) const {
    auto it = constOffsetMap.find(value);
    if (it != constOffsetMap.end())
      return it->second;
    return 0;
  }

  /// Check if a value has a tracked constant offset
  bool hasConstOffset(mlir::Value value) const {
    return constOffsetMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // LDS Base Offset Tracking (for memref.view byte offsets)
  //===--------------------------------------------------------------------===//

  /// Set LDS base offset for a memref (byte offset from memref.view)
  void setLDSBaseOffset(mlir::Value memref, mlir::Value offset) {
    ldsBaseOffsetMap[memref] = offset;
  }

  /// Get LDS base offset for a memref (returns nullopt if not tracked)
  std::optional<mlir::Value> getLDSBaseOffset(mlir::Value memref) const {
    auto it = ldsBaseOffsetMap.find(memref);
    if (it != ldsBaseOffsetMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if a memref has a tracked LDS base offset
  bool hasLDSBaseOffset(mlir::Value memref) const {
    return ldsBaseOffsetMap.contains(memref);
  }

  //===--------------------------------------------------------------------===//
  // LDS Size Tracking (for kernel descriptor)
  //===--------------------------------------------------------------------===//

  /// Add to the total LDS size requirement
  void addLDSSize(int64_t size) { totalLDSSize += size; }

  /// Get the total LDS size requirement
  int64_t getTotalLDSSize() const { return totalLDSSize; }

  //===--------------------------------------------------------------------===//
  // System Register Usage Tracking (for kernel descriptor metadata)
  //===--------------------------------------------------------------------===//

  /// Mark that the kernel uses workgroup ID in dimension x/y/z
  void setUsesWorkgroupIdX(bool uses = true) { usesWorkgroupIdX = uses; }
  void setUsesWorkgroupIdY(bool uses = true) { usesWorkgroupIdY = uses; }
  void setUsesWorkgroupIdZ(bool uses = true) { usesWorkgroupIdZ = uses; }

  /// Mark that the kernel uses workitem/thread ID
  void setUsesWorkitemId(bool uses = true) { usesWorkitemId = uses; }

  /// Query system register usage
  bool getUsesWorkgroupIdX() const { return usesWorkgroupIdX; }
  bool getUsesWorkgroupIdY() const { return usesWorkgroupIdY; }
  bool getUsesWorkgroupIdZ() const { return usesWorkgroupIdZ; }
  bool getUsesWorkitemId() const { return usesWorkitemId; }

  /// Get the number of user SGPRs (for computing system SGPR offsets)
  /// User SGPRs include: kernarg ptr (2) + preloaded args (on gfx950+)
  int64_t getUserSgprCount() const {
    // Base: 2 SGPRs for kernarg_segment_ptr
    int64_t count = 2;
    // On gfx950+ with kernarg preloading, add preloaded args
    if (llvm::isa<GFX950TargetAttr>(target)) {
      // Each kernel arg pointer uses 2 SGPRs
      count += getNumKernelArgs() * 2;
    }
    return count;
  }

  /// Get SGPR index for workgroup ID in the given dimension (0=x, 1=y, 2=z)
  /// System SGPRs (workgroup IDs) come after user SGPRs
  int64_t getWorkgroupIdSgprIndex(int dimension) const {
    int64_t baseIndex = getUserSgprCount();
    return baseIndex + dimension;
  }

  /// Check if this is a multi-wave kernel (more than 64 threads per workgroup)
  /// Multi-wave means workgroup_size_y > 1 or workgroup_size_z > 1
  /// (We assume wave size of 64 and workgroup_size_x is always a multiple of 64)
  bool isMultiWaveKernel() const {
    // Get workgroup_size attribute (using const-safe accessor)
    mlir::ArrayAttr workgroupSizeAttr = program->getAttrOfType<mlir::ArrayAttr>("workgroup_size");

    if (!workgroupSizeAttr || workgroupSizeAttr.size() < 2)
      return false;

    int64_t wgY = 1, wgZ = 1;
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[1])) {
      wgY = intAttr.getInt();
    }
    if (workgroupSizeAttr.size() >= 3) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[2])) {
        wgZ = intAttr.getInt();
      }
    }
    // Multi-wave if y > 1 or z > 1 (matching Python abi.py convention)
    return wgY > 1 || wgZ > 1;
  }

  /// Get workgroup size as (x, y, z) tuple
  std::tuple<int64_t, int64_t, int64_t> getWorkgroupSize() const {
    // Get workgroup_size attribute (using const-safe accessor)
    mlir::ArrayAttr workgroupSizeAttr = program->getAttrOfType<mlir::ArrayAttr>("workgroup_size");
    int64_t wgX = 64, wgY = 1, wgZ = 1;  // defaults
    if (workgroupSizeAttr && workgroupSizeAttr.size() >= 1) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[0])) {
        wgX = intAttr.getInt();
      }
      if (workgroupSizeAttr.size() >= 2) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[1])) {
          wgY = intAttr.getInt();
        }
      }
      if (workgroupSizeAttr.size() >= 3) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[2])) {
          wgZ = intAttr.getInt();
        }
      }
    }
    return {wgX, wgY, wgZ};
  }

private:
  mlir::OpBuilder &builder;
  ProgramOp program;
  TargetAttrInterface target;
  ValueMapper mapper;
  llvm::DenseMap<mlir::Value, int64_t> bindingMap;
  llvm::DenseMap<mlir::Value, int64_t> cacheSwizzleMap;
  llvm::DenseMap<mlir::Value, SRDInfo> srdMap;
  llvm::DenseMap<mlir::Value, int64_t> srdIndexMap;  // memref -> SRD SGPR base index
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>> splitResultsMap;  // originalValue -> split results
  llvm::DenseMap<mlir::Value, int64_t> threadIdUpperBounds;  // thread ID -> upper bound
  llvm::DenseMap<mlir::Value, BitRange> bitRangeMap;  // value -> bit range (for OR vs ADD opt)
  llvm::DenseMap<mlir::Value, int64_t> constOffsetMap;  // value -> constant offset (for buffer store offset:N)
  llvm::DenseMap<mlir::Value, mlir::Value> ldsBaseOffsetMap;  // memref -> LDS byte offset from memref.view
  llvm::SmallVector<LoopContext, 4> loopStack;
  llvm::SmallVector<PendingSRD, 4> pendingSRDs;
  llvm::StringMap<mlir::Value> exprCache;
  int64_t labelCounter = 0;
  int64_t nextSRDIndex = -1;  // Will be computed lazily, starts after user+system SGPRs
  int64_t nextSwizzleSRDIndex = 24;  // Swizzle SRDs start at s24 (after regular arg SRDs)
  int64_t totalLDSSize = 0;  // Total LDS allocation size in bytes
  bool srdPrologueEmitted = false;
  // System register usage tracking
  bool usesWorkgroupIdX = false;
  bool usesWorkgroupIdY = false;
  bool usesWorkgroupIdZ = false;
  bool usesWorkitemId = false;
};

//===----------------------------------------------------------------------===//
// Operation Handlers
//===----------------------------------------------------------------------===//

/// Handler function type for translating operations
using OpHandler = std::function<mlir::LogicalResult(mlir::Operation *,
                                                     TranslationContext &)>;

/// Registry of operation handlers
class OpHandlerRegistry {
public:
  OpHandlerRegistry();

  /// Register a handler for an operation
  void registerHandler(mlir::StringRef opName, OpHandler handler);

  /// Get the handler for an operation
  std::optional<OpHandler> getHandler(mlir::StringRef opName) const;

  /// Check if an operation has a handler
  bool hasHandler(mlir::StringRef opName) const;

  /// Get the default registry with built-in handlers
  static OpHandlerRegistry &getDefault();

private:
  void registerDefaultHandlers();

  llvm::StringMap<OpHandler> handlers;
};

//===----------------------------------------------------------------------===//
// Translation Options
//===----------------------------------------------------------------------===//

/// Options for MLIR to waveasm translation
struct TranslationOptions {
  /// Target architecture (gfx942, gfx950, gfx1250)
  std::string targetId = "gfx942";

  /// Workgroup size (x, y, z). If any dimension is 0, use defaults.
  int64_t workgroupSizeX = 0;
  int64_t workgroupSizeY = 0;
  int64_t workgroupSizeZ = 0;

  /// Subgroup (wavefront) size
  int64_t subgroupSize = 64;

  /// Check if workgroup size is explicitly specified
  bool hasExplicitWorkgroupSize() const {
    return workgroupSizeX > 0 || workgroupSizeY > 0 || workgroupSizeZ > 0;
  }

  /// Get normalized workgroup size (with defaults for 0 values)
  std::tuple<int64_t, int64_t, int64_t> getWorkgroupSize() const {
    return {workgroupSizeX > 0 ? workgroupSizeX : 64,
            workgroupSizeY > 0 ? workgroupSizeY : 1,
            workgroupSizeZ > 0 ? workgroupSizeZ : 1};
  }
};

//===----------------------------------------------------------------------===//
// Translation Functions
//===----------------------------------------------------------------------===//

/// Translate a single MLIR operation to waveasm
mlir::LogicalResult translateOperation(mlir::Operation *op,
                                         TranslationContext &ctx);

/// Translate an MLIR module containing GPU kernels to waveasm programs
mlir::LogicalResult translateModule(mlir::ModuleOp module,
                                     llvm::StringRef targetId);

/// Translate an MLIR module with explicit options
mlir::LogicalResult translateModule(mlir::ModuleOp module,
                                     const TranslationOptions &options);

/// Create an waveasm.program from a GPU function
ProgramOp createProgramFromGPUFunc(mlir::gpu::GPUFuncOp gpuFunc,
                                    mlir::OpBuilder &builder,
                                    llvm::StringRef targetId);

/// Create an waveasm.program from a func.func
ProgramOp createProgramFromFunc(mlir::func::FuncOp funcOp,
                                 mlir::OpBuilder &builder,
                                 llvm::StringRef targetId);

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H
