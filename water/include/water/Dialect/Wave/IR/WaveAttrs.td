// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVEATTRS
#define WATER_DIALECT_WAVE_WAVEATTRS

include "water/Dialect/Wave/IR/WaveDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//-----------------------------------------------------------------------------
// Address space enum and attribute
//-----------------------------------------------------------------------------

def WaveAddressSpaceEnum
    : I32EnumAttr<"WaveAddressSpace", "Where the tensor is located", [
  I32EnumAttrCase<"Unspecified", 0, "unspecified">,
  I32EnumAttrCase<"Global", 1, "global">,
  I32EnumAttrCase<"Shared", 2, "shared">,
  I32EnumAttrCase<"Register", 3, "register">,
]> {
  let cppNamespace = "::wave";
  let genSpecializedAttr = 0;
}

def WaveAddressSpaceAttr
    : EnumAttr<WaveDialect, WaveAddressSpaceEnum, "address_space"> {
  let assemblyFormat = "`<` $value `>`";
}
//-----------------------------------------------------------------------------
// MMA kind enum and attribute
//-----------------------------------------------------------------------------

def WaveMmaKindEnum
    : I32EnumAttr<"WaveMmaKind",
                  "Matrix multiply/accumulate module configuration", [
  // Configurations introduced in CDNA1.
  I32EnumAttrCase<"F32_16x16x16_F16", 0x1020, "f32_16x16x16_f16">,
  I32EnumAttrCase<"F32_32x32x8_F16", 0x1021, "f32_32x32x8_f16">,
  I32EnumAttrCase<"F32_16x16x32_K8_F16", 0x1022, "f32_16x16x32_k8_f16">,
  I32EnumAttrCase<"F32_32x32x16_K8_F16", 0x1023, "f32_32x32x16_k8_f16">,
  I32EnumAttrCase<"I32_16x16x16_I8", 0x10C0, "i32_16x16x16_i8">,
  I32EnumAttrCase<"I32_32x32x8_I8", 0x10C1, "i32_32x32x8_i8">,

  // Configurations introduced in CDNA3.
  I32EnumAttrCase<"F32_16x16x32_F8", 0x1230, "f32_16x16x32_f8">,
  I32EnumAttrCase<"F32_32x32x16_F8", 0x1231, "f32_32x32x16_f8">,
  I32EnumAttrCase<"F32_16x16x32_K4_F8", 0x1232, "f32_16x16x32_k4_f8">,
  I32EnumAttrCase<"F32_32x32x16_K4_F8", 0x1233, "f32_32x32x16_k4_f8">,
  I32EnumAttrCase<"I32_16x16x32_I8", 0x12C0, "i32_16x16x32_i8">,
  I32EnumAttrCase<"I32_32x32x16_I8", 0x12C1, "i32_32x32x16_i8">,

  // Configurations introduced in CDNA4.
  I32EnumAttrCase<"F32_16x16x128_F8F6F4", 0x1340, "f32_16x16x128_f8f6f4">,
  I32EnumAttrCase<"F32_32x32x64_F8F6F4", 0x1341, "f32_32x32x64_f8f6f4">
]> {
  let cppNamespace = "::wave";
  let genSpecializedAttr = 0;
}

def WaveMmaKindAttr : EnumAttr<WaveDialect, WaveMmaKindEnum, "mma_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

//-----------------------------------------------------------------------------
// Other attributes
//-----------------------------------------------------------------------------

def WaveSymbolAttr : AttrDef<WaveDialect, "WaveSymbol"> {
  let mnemonic = "symbol";
  let description = [{
    Base symbol used in symbolic expressions in the Wave dialect. This is
    different from the SymbolRef hierarchy of attributes since it refers to
    symbols, aka variables, in mathematical expressions as opposed to symbols
    identifying global objects such as functions and global variables in
    programs. Wave expression symbols are not part of the symbol table.
  }];

  let parameters = (ins StringRefParameter<"name of the symbol">:$name);
  let assemblyFormat = "`<` $name `>`";

  let extraClassDeclaration = [{
    void printAsSymbolRef(::llvm::raw_ostream &os) const {
      os << "@" << getName();
    }
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEATTRS
