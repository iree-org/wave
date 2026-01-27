// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// GFX950 Target (MI325X, CDNA3.5)
//===----------------------------------------------------------------------===//

class GFX950Target : public AMDGCNTarget {
public:
  GFX950Target() = default;

  //===--------------------------------------------------------------------===//
  // Target Identification
  //===--------------------------------------------------------------------===//

  llvm::StringRef getTargetId() const override { return "gfx950"; }

  llvm::StringRef getArchGeneration() const override { return "GFX9"; }

  llvm::StringRef getComputeArch() const override { return "CDNA3.5"; }

  //===--------------------------------------------------------------------===//
  // Register Limits
  //===--------------------------------------------------------------------===//

  int64_t getMaxVGPRs() const override { return 512; }

  int64_t getMaxSGPRs() const override { return 106; }

  int64_t getMaxAGPRs() const override { return 512; }

  WaveSize getDefaultWaveSize() const override { return WaveSize::Wave64; }

  llvm::SmallVector<WaveSize> getSupportedWaveSizes() const override {
    return {WaveSize::Wave64};
  }

  //===--------------------------------------------------------------------===//
  // Memory Configuration
  //===--------------------------------------------------------------------===//

  int64_t getMaxLDSSize() const override { return 65536; } // 64 KB

  int64_t getLDSBankCount() const override { return 32; }

  int64_t getLDSBankWidth() const override { return 4; } // 4 bytes

  //===--------------------------------------------------------------------===//
  // Instruction Latencies
  //===--------------------------------------------------------------------===//

  int64_t getMFMALatency(llvm::StringRef instrName) const override {
    // GFX950 has improved MFMA throughput
    if (instrName.contains("f32_32x32"))
      return 64;
    if (instrName.contains("f32_16x16"))
      return 32;
    if (instrName.contains("f16_32x32"))
      return 32; // Improved over GFX942
    if (instrName.contains("f16_16x16"))
      return 16;
    if (instrName.contains("bf16"))
      return 16;
    if (instrName.contains("fp8") || instrName.contains("f8"))
      return 16;
    return 16;
  }

  int64_t getGlobalLoadLatency() const override { return 100; }

  int64_t getLDSLoadLatency() const override { return 20; }

  //===--------------------------------------------------------------------===//
  // Feature Queries
  //===--------------------------------------------------------------------===//

  TargetFeature getFeatures() const override {
    return TargetFeature::HasMFMA | TargetFeature::HasFP8 |
           TargetFeature::HasWave64 | TargetFeature::HasAtomicFAdd |
           TargetFeature::HasFlatScratch | TargetFeature::HasAGPRs |
           TargetFeature::HasScaledMFMA | TargetFeature::HasXF32;
  }

  //===--------------------------------------------------------------------===//
  // Wait Count Limits
  //===--------------------------------------------------------------------===//

  int64_t getMaxVmcnt() const override { return 63; }

  int64_t getMaxLgkmcnt() const override { return 15; }

  int64_t getMaxExpcnt() const override { return 7; }

  //===--------------------------------------------------------------------===//
  // Code Object
  //===--------------------------------------------------------------------===//

  int64_t getDefaultCodeObjectVersion() const override { return 5; }

  llvm::SmallVector<int64_t> getSupportedCodeObjectVersions() const override {
    return {4, 5};
  }

  //===--------------------------------------------------------------------===//
  // Assembly Emission
  //===--------------------------------------------------------------------===//

  std::string getTargetDirective() const override {
    return ".amdgcn_target \"amdgcn-amd-amdhsa--gfx950\"";
  }

  std::string getABIVersion() const override { return "amdhsa"; }

  //===--------------------------------------------------------------------===//
  // Instruction Support
  //===--------------------------------------------------------------------===//

  bool supportsInstruction(llvm::StringRef instrName) const override {
    // GFX950 supports most GFX9 instructions plus new ones
    return true;
  }

  std::optional<std::string>
  getTargetInstructionName(llvm::StringRef genericName) const override {
    // No aliasing needed
    return std::nullopt;
  }
};

// Factory function for GFX950
std::unique_ptr<AMDGCNTarget> createGFX950Target() {
  return std::make_unique<GFX950Target>();
}

} // namespace waveasm
