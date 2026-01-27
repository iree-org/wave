// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMTYPES
#define WaveASM_DIALECT_WAVEASMTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "waveasm/Dialect/WaveASMDialect.td"

// Base class for WAVEASM types
class WAVEASMType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WaveASMDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Register Class Enum
//===----------------------------------------------------------------------===//

def WaveASM_RegClass : I32EnumAttr<"RegClass", "Register class", [
  I32EnumAttrCase<"VGPR", 0, "vgpr">,
  I32EnumAttrCase<"SGPR", 1, "sgpr">,
  I32EnumAttrCase<"AGPR", 2, "agpr">,
]> {
  let cppNamespace = "::waveasm";
}

//===----------------------------------------------------------------------===//
// Virtual Register Types (Pre-Allocation)
//===----------------------------------------------------------------------===//

def WaveASM_VRegType : WAVEASMType<"VReg", "vreg"> {
  let summary = "Virtual VGPR";
  let description = [{
    Represents a virtual VGPR that will be assigned to a physical VGPR
    during register allocation. This is pure SSA - each definition creates
    a new value.

    Parameters:
    - size: Number of consecutive registers (1, 2, 4, 8, 16)
    - alignment: Required physical alignment (default = size)

    Examples:
    ```mlir
    !waveasm.vreg           // single VGPR
    !waveasm.vreg<4>        // 4 consecutive VGPRs
    !waveasm.vreg<4, 4>     // 4 VGPRs, 4-aligned (for MFMA)
    ```
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "1">:$size,
    DefaultValuedParameter<"int64_t", "1">:$alignment
  );

  let assemblyFormat = "(`<` $size^ (`,` $alignment^)? `>`)?";

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, 1, 1);
    }]>,
    TypeBuilder<(ins "int64_t":$size), [{
      return $_get($_ctxt, size, size);  // alignment defaults to size
    }]>
  ];

  let extraClassDeclaration = [{
    bool isScalar() const { return getSize() == 1; }
    bool isVector() const { return getSize() > 1; }
  }];
}

def WaveASM_SRegType : WAVEASMType<"SReg", "sreg"> {
  let summary = "Virtual SGPR";
  let description = [{
    Represents a virtual SGPR that will be assigned to a physical SGPR
    during register allocation.

    Parameters:
    - size: Number of consecutive registers (1, 2, 4, 8, 16)
    - alignment: Required physical alignment (default = size)

    Examples:
    ```mlir
    !waveasm.sreg           // single SGPR
    !waveasm.sreg<2>        // SGPR pair (64-bit pointer)
    !waveasm.sreg<4>        // 4 SGPRs (128-bit descriptor)
    ```
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "1">:$size,
    DefaultValuedParameter<"int64_t", "1">:$alignment
  );

  let assemblyFormat = "(`<` $size^ (`,` $alignment^)? `>`)?";

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, 1, 1);
    }]>,
    TypeBuilder<(ins "int64_t":$size), [{
      return $_get($_ctxt, size, size);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isScalar() const { return getSize() == 1; }
    bool isPair() const { return getSize() == 2; }
  }];
}

//===----------------------------------------------------------------------===//
// Physical Register Types (Post-Allocation / Precolored)
//===----------------------------------------------------------------------===//

def WaveASM_PVRegType : WAVEASMType<"PVReg", "pvreg"> {
  let summary = "Physical VGPR";
  let description = [{
    Represents a physical VGPR at a specific hardware register index.
    Used for:
    - ABI-mandated registers (v0 for thread ID)
    - Precolored registers
    - After register allocation

    Parameters:
    - index: Physical register number (0-255)
    - size: Number of consecutive registers (1, 2, 4, etc.)

    Examples:
    ```mlir
    !waveasm.pvreg<0>       // v0
    !waveasm.pvreg<4, 4>    // v[4:7]
    ```
  }];

  let parameters = (ins
    "int64_t":$index,
    DefaultValuedParameter<"int64_t", "1">:$size
  );

  let assemblyFormat = "`<` $index (`,` $size^)? `>`";

  let extraClassDeclaration = [{
    int64_t getEndIndex() const { return getIndex() + getSize(); }
  }];
}

def WaveASM_PSRegType : WAVEASMType<"PSReg", "psreg"> {
  let summary = "Physical SGPR";
  let description = [{
    Represents a physical SGPR at a specific hardware register index.
    Used for:
    - ABI-mandated registers (s[0:1] for kernarg pointer)
    - Precolored registers
    - After register allocation

    Parameters:
    - index: Physical register number (0-105)
    - size: Number of consecutive registers

    Examples:
    ```mlir
    !waveasm.psreg<0, 2>    // s[0:1] (kernarg ptr)
    !waveasm.psreg<2>       // s2 (workgroup id)
    ```
  }];

  let parameters = (ins
    "int64_t":$index,
    DefaultValuedParameter<"int64_t", "1">:$size
  );

  let assemblyFormat = "`<` $index (`,` $size^)? `>`";

  let extraClassDeclaration = [{
    int64_t getEndIndex() const { return getIndex() + getSize(); }
  }];
}

//===----------------------------------------------------------------------===//
// Immediate Value Type
//===----------------------------------------------------------------------===//

def WaveASM_ImmType : WAVEASMType<"Imm", "imm"> {
  let summary = "Immediate constant value";
  let description = [{
    Represents an immediate constant that can be used as an instruction operand.
    Supports both inline constants and literal constants.

    Examples:
    ```mlir
    !waveasm.imm<0>
    !waveasm.imm<42>
    !waveasm.imm<0x1000>
    ```
  }];

  let parameters = (ins "int64_t":$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    // Check if this immediate can be encoded as an inline constant
    bool isInlineConstant() const {
      int64_t v = getValue();
      // Integer inline constants: -16 to 64
      if (v >= -16 && v <= 64) return true;
      // Common float constants (as int bit patterns)
      if (v == 0x3F800000 || v == 0xBF800000 ||  // 1.0f, -1.0f
          v == 0x40000000 || v == 0xC0000000 ||  // 2.0f, -2.0f
          v == 0x40800000 || v == 0xC0800000)    // 4.0f, -4.0f
        return true;
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Type Predicates for Op Definitions
//===----------------------------------------------------------------------===//

// Any virtual register
def WaveASM_AnyVReg : AnyTypeOf<[WaveASM_VRegType], "virtual VGPR">;
def WaveASM_AnySReg : AnyTypeOf<[WaveASM_SRegType], "virtual SGPR">;
def WaveASM_AnyVirtualReg : AnyTypeOf<[WaveASM_VRegType, WaveASM_SRegType], "virtual register">;

// Any physical register
def WaveASM_AnyPVReg : AnyTypeOf<[WaveASM_PVRegType], "physical VGPR">;
def WaveASM_AnyPSReg : AnyTypeOf<[WaveASM_PSRegType], "physical SGPR">;
def WaveASM_AnyPhysicalReg : AnyTypeOf<[WaveASM_PVRegType, WaveASM_PSRegType], "physical register">;

// Any VGPR (virtual or physical)
def WaveASM_AnyVGPR : AnyTypeOf<[WaveASM_VRegType, WaveASM_PVRegType], "any VGPR">;

// Any SGPR (virtual or physical)
def WaveASM_AnySGPR : AnyTypeOf<[WaveASM_SRegType, WaveASM_PSRegType], "any SGPR">;

// Any register
def WaveASM_AnyReg : AnyTypeOf<[
  WaveASM_VRegType, WaveASM_SRegType,
  WaveASM_PVRegType, WaveASM_PSRegType
], "any register">;

// Immediate
def WaveASM_Imm : AnyTypeOf<[WaveASM_ImmType], "immediate">;

// VGPR or immediate (common for src1)
def WaveASM_VRegOrImm : AnyTypeOf<[WaveASM_VRegType, WaveASM_PVRegType, WaveASM_ImmType],
                                  "VGPR or immediate">;

// SGPR or immediate
def WaveASM_SRegOrImm : AnyTypeOf<[WaveASM_SRegType, WaveASM_PSRegType, WaveASM_ImmType],
                                  "SGPR or immediate">;

// VALU source: can be VGPR, SGPR, or Immediate (SGPR gets broadcast to all lanes)
def WaveASM_VALUSrc : AnyTypeOf<[
  WaveASM_VRegType, WaveASM_SRegType,
  WaveASM_PVRegType, WaveASM_PSRegType,
  WaveASM_ImmType
], "VALU source operand">;

// Any operand (register or immediate)
def WaveASM_AnyOperand : AnyTypeOf<[
  WaveASM_VRegType, WaveASM_SRegType,
  WaveASM_PVRegType, WaveASM_PSRegType,
  WaveASM_ImmType
], "any operand">;

#endif // WaveASM_DIALECT_WAVEASMTYPES
