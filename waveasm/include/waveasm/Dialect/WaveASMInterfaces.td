// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMINTERFACES
#define WaveASM_DIALECT_WAVEASMINTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def WaveASM_TargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
      Interface for target attributes.
  }];

  let methods = [
      //===--------------------------------------------------------------------===//
      // Target Identification
      //===--------------------------------------------------------------------===//
      InterfaceMethod<
          "Get target architecture generation (GFX9, GFX10, GFX12).",
          "::llvm::StringRef", "getArchGeneration", (ins)>,
      InterfaceMethod<"Get compute unit architecture (CDNA2, CDNA3, RDNA4).",
                      "::llvm::StringRef", "getComputeArch", (ins)>,

      //===--------------------------------------------------------------------===//
      // Feature Queries
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get feature flags.", "TargetFeature", "getFeatures",
                      (ins)>,

      //===--------------------------------------------------------------------===//
      // Register Limits
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get maximum number of VGPRs.", "int64_t", "getMaxVGPRs",
                      (ins)>,
      InterfaceMethod<"Get maximum number of SGPRs.", "int64_t", "getMaxSGPRs",
                      (ins)>,
      InterfaceMethod<"Get default wave size.", "int64_t", "getDefaultWaveSize",
                      (ins)>,
      InterfaceMethod<"Get supported wave sizes.",
                      "::llvm::SmallVector<int64_t>", "getSupportedWaveSizes",
                      (ins)>,
      InterfaceMethod<"Get maximum number of AGPRs (if supported).", "int64_t",
                      "getMaxAGPRs", (ins)>,

      //===--------------------------------------------------------------------===//
      // Memory Configuration
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get maximum LDS size per workgroup in bytes.", "int64_t",
                      "getMaxLDSSize", (ins)>,
      InterfaceMethod<"Get LDS bank count.", "int64_t", "getLDSBankCount",
                      (ins)>,
      InterfaceMethod<"Get LDS bank width in bytes.", "int64_t",
                      "getLDSBankWidth", (ins)>,

      //===--------------------------------------------------------------------===//
      // Instruction Latencies
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get MFMA instruction latency (if supported).", "int64_t",
                      "getMFMALatency", (ins "::llvm::StringRef":$instrName)>,
      InterfaceMethod<"Get global memory load latency.", "int64_t",
                      "getGlobalLoadLatency", (ins)>,
      InterfaceMethod<"Get LDS load latency.", "int64_t", "getLDSLoadLatency",
                      (ins)>,

      //===--------------------------------------------------------------------===//
      // Wait Count Limits
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get maximum vmcnt value.", "int64_t", "getMaxVmcnt",
                      (ins)>,
      InterfaceMethod<"Get maximum lgkmcnt value.", "int64_t", "getMaxLgkmcnt",
                      (ins)>,
      InterfaceMethod<"Get maximum expcnt value.", "int64_t", "getMaxExpcnt",
                      (ins)>,

      //===--------------------------------------------------------------------===//
      // Code Object
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get default code object version.", "int64_t",
                      "getDefaultCodeObjectVersion", (ins)>,
      InterfaceMethod<"Get supported code object versions.",
                      "::llvm::SmallVector<int64_t>",
                      "getSupportedCodeObjectVersions", (ins)>,

      //===--------------------------------------------------------------------===//
      // Assembly Emission
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Get target directive for assembly.", "::llvm::StringRef",
                      "getTargetDirective", (ins)>,
      InterfaceMethod<"Get ABI version string.", "::llvm::StringRef",
                      "getABIVersion", (ins)>,

      //===--------------------------------------------------------------------===//
      // Instruction Support
      //===--------------------------------------------------------------------===//
      InterfaceMethod<"Check if an instruction is supported on this target.",
                      "bool", "supportsInstruction",
                      (ins "::llvm::StringRef":$instrName)>,
      InterfaceMethod<
          "Get target-specific instruction name (for aliased instructions).",
          "std::optional<std::string>", "getTargetInstructionName",
          (ins "::llvm::StringRef":$genericName)>,
  ];

  let extraClassDeclaration = [{
    bool hasFeature(TargetFeature feature) const {
      return ::waveasm::hasFeature(getFeatures(), feature);
    }

    // Check if target supports MFMA.
    bool supportsMFMA() const {
      return hasFeature(TargetFeature::HasMFMA);
    }

    // Check if target supports FP8.
    bool supportsFP8() const {
      return hasFeature(TargetFeature::HasFP8);
    }
  }];
}

//===----------------------------------------------------------------------===//
// WaveASM Operation Traits
//
// These traits categorize operations for passes like CSE, liveness, etc.
// Using traits instead of string matching provides type safety and
// allows the compiler to verify correctness at compile time.
//===----------------------------------------------------------------------===//

// Trait for arithmetic/ALU operations that are pure and CSE-eligible
def WaveASM_ArithmeticOp : NativeOpTrait<"ArithmeticOp">;

// Trait for MFMA operations
def WaveASM_MFMAOp : NativeOpTrait<"MFMAOp">;

// Trait for memory operations (loads/stores)
def WaveASM_MemoryOp : NativeOpTrait<"MemoryOp">;

// Trait for control flow operations (branches, barriers)
def WaveASM_ControlFlowOp : NativeOpTrait<"ControlFlowOp">;

// Trait for special register operations (M0, VCC, EXEC)
def WaveASM_SpecialRegOp : NativeOpTrait<"SpecialRegOp">;

#endif // WaveASM_DIALECT_WAVEASMINTERFACES
