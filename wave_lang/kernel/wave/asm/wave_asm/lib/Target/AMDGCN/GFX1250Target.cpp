// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// GFX1250 Target (RDNA4)
//===----------------------------------------------------------------------===//

class GFX1250Target : public AMDGCNTarget {
public:
  GFX1250Target() = default;

  //===--------------------------------------------------------------------===//
  // Target Identification
  //===--------------------------------------------------------------------===//

  llvm::StringRef getTargetId() const override { return "gfx1250"; }

  llvm::StringRef getArchGeneration() const override { return "GFX12"; }

  llvm::StringRef getComputeArch() const override { return "RDNA4"; }

  //===--------------------------------------------------------------------===//
  // Register Limits
  //===--------------------------------------------------------------------===//

  int64_t getMaxVGPRs() const override { return 256; }

  int64_t getMaxSGPRs() const override { return 106; }

  int64_t getMaxAGPRs() const override { return 0; } // No AGPRs on RDNA

  WaveSize getDefaultWaveSize() const override { return WaveSize::Wave32; }

  llvm::SmallVector<WaveSize> getSupportedWaveSizes() const override {
    return {WaveSize::Wave32, WaveSize::Wave64};
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
    // GFX1250 uses WMMA instead of MFMA
    if (instrName.contains("wmma"))
      return 16;
    return 0; // No traditional MFMA support
  }

  int64_t getGlobalLoadLatency() const override { return 80; }

  int64_t getLDSLoadLatency() const override { return 16; }

  //===--------------------------------------------------------------------===//
  // Feature Queries
  //===--------------------------------------------------------------------===//

  TargetFeature getFeatures() const override {
    return TargetFeature::HasWave32 | TargetFeature::HasWave64 |
           TargetFeature::HasAtomicFAdd | TargetFeature::HasFlatScratch |
           TargetFeature::HasPackedFP32 | TargetFeature::HasGlobalLoadLDS;
  }

  //===--------------------------------------------------------------------===//
  // Wait Count Limits
  //===--------------------------------------------------------------------===//

  int64_t getMaxVmcnt() const override { return 63; }

  int64_t getMaxLgkmcnt() const override { return 63; }

  int64_t getMaxExpcnt() const override { return 7; }

  //===--------------------------------------------------------------------===//
  // Code Object
  //===--------------------------------------------------------------------===//

  int64_t getDefaultCodeObjectVersion() const override { return 5; }

  llvm::SmallVector<int64_t> getSupportedCodeObjectVersions() const override {
    return {5};
  }

  //===--------------------------------------------------------------------===//
  // Assembly Emission
  //===--------------------------------------------------------------------===//

  std::string getTargetDirective() const override {
    return ".amdgcn_target \"amdgcn-amd-amdhsa--gfx1250\"";
  }

  std::string getABIVersion() const override { return "amdhsa"; }

  //===--------------------------------------------------------------------===//
  // Instruction Support
  //===--------------------------------------------------------------------===//

  bool supportsInstruction(llvm::StringRef instrName) const override {
    // GFX1250 (RDNA4) doesn't support traditional MFMA
    if (instrName.contains("v_mfma_"))
      return false;

    // Check for CDNA-specific instructions
    if (instrName.contains("_accvgpr_"))
      return false;

    return true;
  }

  std::optional<std::string>
  getTargetInstructionName(llvm::StringRef genericName) const override {
    // Some instructions may have different names on RDNA
    // Add aliasing here if needed
    return std::nullopt;
  }
};

// Factory function for GFX1250
std::unique_ptr<AMDGCNTarget> createGFX1250Target() {
  return std::make_unique<GFX1250Target>();
}

} // namespace waveasm
