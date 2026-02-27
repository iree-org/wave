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

// Interface for MFMA operations - provides typed accessors for operands
def WaveASM_MFMAOp : OpInterface<"MFMAOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for matrix fused multiply-add (MFMA) operations.
    All MFMA ops have the same operand structure: a, b, acc -> dst.
  }];

  let methods = [
    InterfaceMethod<
      "Get the first matrix operand (A).",
      "::mlir::Value", "getA", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,
    InterfaceMethod<
      "Get the second matrix operand (B).",
      "::mlir::Value", "getB", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(1);
      }]
    >,
    InterfaceMethod<
      "Get the accumulator operand.",
      "::mlir::Value", "getAcc", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(2);
      }]
    >,
    InterfaceMethod<
      "Get the destination/result value.",
      "::mlir::Value", "getDst", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
    >,
    InterfaceMethod<
      "Get the tied operand index (accumulator index for tying).",
      "int", "getTiedOperandIndex", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return 2; // acc is always operand index 2
      }]
    >
  ];
}

// Interface for VMEM load operations (buffer_load_*, global_load_*)
// Provides typed accessors for saddr, voffset, soffset
def WaveASM_VMEMLoadOp : OpInterface<"VMEMLoadOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for VMEM load operations (buffer and global loads).
    Operands: saddr (SRD/pointer), voffset (per-lane offset), soffset (scalar offset).
  }];

  let methods = [
    InterfaceMethod<
      "Get the buffer descriptor / address operand.",
      "::mlir::Value", "getSaddr", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,
    InterfaceMethod<
      "Get the vector offset operand.",
      "::mlir::Value", "getVoffset", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(1);
      }]
    >,
    InterfaceMethod<
      "Get the scalar offset operand.",
      "::mlir::Value", "getSoffset", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(2);
      }]
    >
  ];
}

// Interface for VMEM store operations (buffer_store_*, global_store_*)
// Provides typed accessors for data, saddr, voffset
def WaveASM_VMEMStoreOp : OpInterface<"VMEMStoreOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for VMEM store operations (buffer and global stores).
    Operands: data (value to store), saddr (SRD/pointer), voffset (per-lane offset).
  }];

  let methods = [
    InterfaceMethod<
      "Get the data operand (value to store).",
      "::mlir::Value", "getData", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,
    InterfaceMethod<
      "Get the buffer descriptor / address operand.",
      "::mlir::Value", "getSaddr", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(1);
      }]
    >,
    InterfaceMethod<
      "Get the vector offset operand.",
      "::mlir::Value", "getVoffset", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(2);
      }]
    >
  ];
}

// Interface for LDS load operations (ds_read_*)
// Provides typed accessor for vaddr
def WaveASM_LDSLoadOp : OpInterface<"LDSLoadOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for LDS load operations.
    Operands: vaddr (LDS address).
  }];

  let methods = [
    InterfaceMethod<
      "Get the LDS address operand.",
      "::mlir::Value", "getVaddr", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >
  ];
}

// Interface for LDS store operations (ds_write_*)
// Provides typed accessors for data, vaddr
def WaveASM_LDSStoreOp : OpInterface<"LDSStoreOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for LDS store operations.
    Operands: data (value to store), vaddr (LDS address).
  }];

  let methods = [
    InterfaceMethod<
      "Get the data operand (value to store).",
      "::mlir::Value", "getData", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,
    InterfaceMethod<
      "Get the LDS address operand.",
      "::mlir::Value", "getVaddr", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(1);
      }]
    >
  ];
}

// Interface for VMEM-to-LDS load operations (buffer_load_*_lds)
// Provides typed accessors for voffset, srd, soffset
// Note: Different operand order from VMEMLoadOp - voffset comes first
def WaveASM_VMEMToLDSLoadOp : OpInterface<"VMEMToLDSLoadOpInterface"> {
  let cppNamespace = "::waveasm";
  let description = [{
    Interface for VMEM-to-LDS (gather-to-LDS) load operations.
    Operands: voffset (per-lane offset), srd (buffer descriptor), soffset (scalar offset).
  }];

  let methods = [
    InterfaceMethod<
      "Get the vector offset operand.",
      "::mlir::Value", "getVoffset", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(0);
      }]
    >,
    InterfaceMethod<
      "Get the buffer descriptor operand.",
      "::mlir::Value", "getSrd", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(1);
      }]
    >,
    InterfaceMethod<
      "Get the scalar offset operand.",
      "::mlir::Value", "getSoffset", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getOperand(2);
      }]
    >
  ];
}

// Trait for memory operations (loads/stores)
def WaveASM_MemoryOp : NativeOpTrait<"MemoryOp">;

// Trait for control flow operations (branches, barriers)
def WaveASM_ControlFlowOp : NativeOpTrait<"ControlFlowOp">;

// Trait for special register operations (M0, VCC, EXEC)
def WaveASM_SpecialRegOp : NativeOpTrait<"SpecialRegOp">;

#endif // WaveASM_DIALECT_WAVEASMINTERFACES
