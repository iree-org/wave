// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H
#define WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Target Feature Flags
//===----------------------------------------------------------------------===//

enum class TargetFeature : uint32_t {
  None = 0,
  HasMFMA = 1 << 0,          // Matrix fused multiply-add
  HasFP8 = 1 << 1,           // FP8 support
  HasPackedFP32 = 1 << 2,    // Packed FP32 operations
  HasWave32 = 1 << 3,        // Wave32 mode support
  HasWave64 = 1 << 4,        // Wave64 mode support
  HasXF32 = 1 << 5,          // Extended FP32 (TF32)
  HasScaledMFMA = 1 << 6,    // Scaled MFMA instructions
  HasAtomicFAdd = 1 << 7,    // Atomic float add
  HasGlobalLoadLDS = 1 << 8, // Global load to LDS
  HasFlatScratch = 1 << 9,   // Flat scratch support
  HasAGPRs = 1 << 10,        // Accumulator GPRs
};

inline TargetFeature operator|(TargetFeature a, TargetFeature b) {
  return static_cast<TargetFeature>(static_cast<uint32_t>(a) |
                                     static_cast<uint32_t>(b));
}

inline TargetFeature operator&(TargetFeature a, TargetFeature b) {
  return static_cast<TargetFeature>(static_cast<uint32_t>(a) &
                                     static_cast<uint32_t>(b));
}

inline bool hasFeature(TargetFeature features, TargetFeature query) {
  return (features & query) == query;
}

//===----------------------------------------------------------------------===//
// Wave Size
//===----------------------------------------------------------------------===//

enum class WaveSize : int64_t {
  Wave32 = 32,
  Wave64 = 64,
};

//===----------------------------------------------------------------------===//
// AMDGCN Target Interface
//===----------------------------------------------------------------------===//

/// Abstract interface for AMDGCN targets
class AMDGCNTarget {
public:
  virtual ~AMDGCNTarget() = default;

  //===--------------------------------------------------------------------===//
  // Target Identification
  //===--------------------------------------------------------------------===//

  /// Get target ID string (e.g., "gfx942")
  virtual llvm::StringRef getTargetId() const = 0;

  /// Get target architecture generation (GFX9, GFX10, GFX12)
  virtual llvm::StringRef getArchGeneration() const = 0;

  /// Get compute unit architecture (CDNA2, CDNA3, RDNA4)
  virtual llvm::StringRef getComputeArch() const = 0;

  //===--------------------------------------------------------------------===//
  // Register Limits
  //===--------------------------------------------------------------------===//

  /// Get maximum number of VGPRs
  virtual int64_t getMaxVGPRs() const = 0;

  /// Get maximum number of SGPRs
  virtual int64_t getMaxSGPRs() const = 0;

  /// Get maximum number of AGPRs (if supported)
  virtual int64_t getMaxAGPRs() const = 0;

  /// Get default wave size
  virtual WaveSize getDefaultWaveSize() const = 0;

  /// Get supported wave sizes
  virtual llvm::SmallVector<WaveSize> getSupportedWaveSizes() const = 0;

  //===--------------------------------------------------------------------===//
  // Memory Configuration
  //===--------------------------------------------------------------------===//

  /// Get maximum LDS size per workgroup in bytes
  virtual int64_t getMaxLDSSize() const = 0;

  /// Get LDS bank count
  virtual int64_t getLDSBankCount() const = 0;

  /// Get LDS bank width in bytes
  virtual int64_t getLDSBankWidth() const = 0;

  //===--------------------------------------------------------------------===//
  // Instruction Latencies
  //===--------------------------------------------------------------------===//

  /// Get MFMA instruction latency (if supported)
  virtual int64_t getMFMALatency(llvm::StringRef instrName) const = 0;

  /// Get global memory load latency
  virtual int64_t getGlobalLoadLatency() const = 0;

  /// Get LDS load latency
  virtual int64_t getLDSLoadLatency() const = 0;

  //===--------------------------------------------------------------------===//
  // Feature Queries
  //===--------------------------------------------------------------------===//

  /// Get feature flags
  virtual TargetFeature getFeatures() const = 0;

  /// Check if a feature is supported
  bool hasFeature(TargetFeature feature) const {
    return waveasm::hasFeature(getFeatures(), feature);
  }

  //===--------------------------------------------------------------------===//
  // Wait Count Limits
  //===--------------------------------------------------------------------===//

  /// Get maximum vmcnt value
  virtual int64_t getMaxVmcnt() const = 0;

  /// Get maximum lgkmcnt value
  virtual int64_t getMaxLgkmcnt() const = 0;

  /// Get maximum expcnt value
  virtual int64_t getMaxExpcnt() const = 0;

  //===--------------------------------------------------------------------===//
  // Code Object
  //===--------------------------------------------------------------------===//

  /// Get default code object version
  virtual int64_t getDefaultCodeObjectVersion() const = 0;

  /// Get supported code object versions
  virtual llvm::SmallVector<int64_t> getSupportedCodeObjectVersions() const = 0;

  //===--------------------------------------------------------------------===//
  // Assembly Emission
  //===--------------------------------------------------------------------===//

  /// Get target directive for assembly
  virtual std::string getTargetDirective() const = 0;

  /// Get ABI version string
  virtual std::string getABIVersion() const = 0;

  //===--------------------------------------------------------------------===//
  // Instruction Support
  //===--------------------------------------------------------------------===//

  /// Check if an instruction is supported on this target
  virtual bool supportsInstruction(llvm::StringRef instrName) const = 0;

  /// Get target-specific instruction name (for aliased instructions)
  virtual std::optional<std::string>
  getTargetInstructionName(llvm::StringRef genericName) const = 0;
};

//===----------------------------------------------------------------------===//
// Target Registry
//===----------------------------------------------------------------------===//

/// Get target implementation for a given target ID
std::unique_ptr<AMDGCNTarget> getAMDGCNTarget(llvm::StringRef targetId);

/// Check if a target ID is supported
bool isTargetSupported(llvm::StringRef targetId);

/// Get list of all supported target IDs
llvm::SmallVector<llvm::StringRef> getSupportedTargets();

} // namespace waveasm

#endif // WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H
