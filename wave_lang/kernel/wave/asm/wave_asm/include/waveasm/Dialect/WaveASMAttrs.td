// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMATTRS
#define WaveASM_DIALECT_WAVEASMATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "waveasm/Dialect/WaveASMDialect.td"

// Base class for WAVEASM attributes
class WAVEASMAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WaveASMDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Kernel ABI Attribute
//===----------------------------------------------------------------------===//

def WaveASM_KernelABIAttr : WAVEASMAttr<"KernelABI", "abi"> {
  let summary = "ABI bindings for precolored registers";
  let description = [{
    Defines the GPU ABI for kernel entry. These registers are precolored
    and reserved from allocation.

    Parameters:
    - tid: VGPR index for thread ID (typically 0)
    - kernarg: SGPR index for kernarg pointer low bits (typically 0, pair uses 0:1)
    - wg_id_x, wg_id_y, wg_id_z: Optional SGPR indices for workgroup IDs

    Example:
    ```mlir
    waveasm.program @kernel
      abi = #waveasm.abi<tid = 0, kernarg = 0, wg_id_x = 2>
    ```
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "0">:$tid,
    DefaultValuedParameter<"int64_t", "0">:$kernarg,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_x,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_y,
    OptionalParameter<"std::optional<int64_t>">:$wg_id_z
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Get reserved VGPR indices
    ::llvm::SmallVector<int64_t> getReservedVGPRs() const {
      return {getTid()};
    }

    // Get reserved SGPR indices
    ::llvm::SmallVector<int64_t> getReservedSGPRs() const {
      ::llvm::SmallVector<int64_t> reserved;
      // Kernarg pointer is a pair (low, high)
      reserved.push_back(getKernarg());
      reserved.push_back(getKernarg() + 1);
      if (auto x = getWgIdX()) reserved.push_back(*x);
      if (auto y = getWgIdY()) reserved.push_back(*y);
      if (auto z = getWgIdZ()) reserved.push_back(*z);
      return reserved;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Target Attribute
//===----------------------------------------------------------------------===//

def WaveASM_TargetAttr : WAVEASMAttr<"Target", "target"> {
  let summary = "Target GPU architecture";
  let description = [{
    Specifies the target GPU architecture and code object version.

    Supported targets:
    - gfx942: AMD CDNA3 (MI300 series)
    - gfx950: AMD CDNA3+ (future MI series)
    - gfx1250: AMD RDNA4

    Code object versions:
    - 4: Standard code object
    - 5: With kernel argument preloading support

    Example:
    ```mlir
    waveasm.program @kernel target = #waveasm.target<"gfx942", 5>
    ```
  }];

  let parameters = (ins
    StringRefParameter<"id">:$id,
    DefaultValuedParameter<"int64_t", "5">:$code_object_version
  );

  let assemblyFormat = "`<` $id (`,` $code_object_version^)? `>`";

  let extraClassDeclaration = [{
    // Check if target supports MFMA
    bool supportsMFMA() const {
      return getId().starts_with("gfx9");
    }

    // Check if target supports FP8
    bool supportsFP8() const {
      return getId() == "gfx942" || getId() == "gfx950";
    }

    // Get max VGPRs for target
    int64_t getMaxVGPRs() const {
      if (getId() == "gfx1250") return 256;   // RDNA4
      if (getId() == "gfx942") return 512;    // MI300
      if (getId() == "gfx950") return 512;    // Future MI
      return 256;
    }

    // Get max SGPRs for target
    int64_t getMaxSGPRs() const {
      if (getId() == "gfx1250") return 106;
      return 104;
    }

    // Get wave size
    int64_t getWaveSize() const {
      if (getId() == "gfx1250") return 32;    // RDNA4 can use wave32
      return 64;                               // CDNA uses wave64
    }
  }];
}

#endif // WaveASM_DIALECT_WAVEASMATTRS
