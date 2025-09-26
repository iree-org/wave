// Copyright 2025 The Wave Authors
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
  I32EnumAttrCase<"F32_32x32x64_F8F6F4", 0x1341, "f32_32x32x64_f8f6f4">,
  I32EnumAttrCase<"F32_32x32x16_BF16", 0x1320, "f32_32x32x16_bf16">,
  I32EnumAttrCase<"F32_16x16x32_BF16", 0x1321, "f32_16x16x32_bf16">,
  I32EnumAttrCase<"F32_32x32x16_F16", 0x1322, "f32_32x32x16_f16">,
  I32EnumAttrCase<"F32_16x16x32_F16", 0x1323, "f32_16x16x32_f16">,

]> {
  let cppNamespace = "::wave";
  let genSpecializedAttr = 0;
}

def WaveMmaKindAttr : EnumAttr<WaveDialect, WaveMmaKindEnum, "mma_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

//-----------------------------------------------------------------------------
// Normal form enum and attribute
//-----------------------------------------------------------------------------

def NORMAL_FORM_FUNC_BOUNDARY : I32BitEnumAttrCaseBit<
    "FunctionBoundarySpecified", 0, "full_func_boundary">;
def NORMAL_FORM_OP_TYPES : I32BitEnumAttrCaseBit<
    "OpTypesSpecified",          1, "full_op_types">;
def NORMAL_FORM_INDEX_EXPRS : I32BitEnumAttrCaseBit<
    "IndexExprsSpecified",       2, "index_exprs">;

def NORMAL_FORM_FULL_TYPES : I32BitEnumAttrCaseGroup<
    "AllTypesSpecified", [
        NORMAL_FORM_FUNC_BOUNDARY, NORMAL_FORM_OP_TYPES
    ], "full_types">;

def WaveNormalFormEnum : I32BitEnumAttr<"WaveNormalForm", "", [
  I32BitEnumAttrCaseNone<"None", "none">,
  // Bits.
  NORMAL_FORM_FUNC_BOUNDARY,
  NORMAL_FORM_OP_TYPES,
  NORMAL_FORM_INDEX_EXPRS,

  // Group aliases.
  NORMAL_FORM_FULL_TYPES
]> {
  let cppNamespace = "::wave";
  let separator = ",";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def WaveNormalFormAttr
    : EnumAttr<WaveDialect, WaveNormalFormEnum, "normal_form"> {
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

def WaveIndexMappingAttr : AttrDef<WaveDialect, "WaveIndexMapping"> {
  let mnemonic = "index_mapping";
  let description = [{
    An affine map with named symbols for Wave indexing expressions.

    This attribute preserves meaningful symbol names (e.g., WG0, BLOCK_M, T0)
    while storing affine maps internally for start, step, and stride. The
    symbol_names array corresponds 1:1 to the symbols in the affine
    expressions, where s0 maps to symbol_names[0], s1 to symbol_names[1], etc.

    Custom syntax: [symbol_names] -> (start_expr, step_expr, stride_expr)
    Example: [WG0, BLOCK_M] -> (WG0 * BLOCK_M + 42, 1, BLOCK_M)
  }];

  let parameters = (ins
    ArrayRefParameter<"WaveSymbolAttr">:$symbol_names,
    "::mlir::AffineMap":$start,
    "::mlir::AffineMap":$step,
    "::mlir::AffineMap":$stride
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the symbol name at the given index.
    ::llvm::StringRef getSymbolName(unsigned index) {
      auto names = getSymbolNames();
      if (index >= names.size()) return ::llvm::StringRef();
      return names[index].getName();
    }

    /// Get all symbol names as a vector of string references.
    ::llvm::SmallVector<::llvm::StringRef> getAllSymbolNames() const {
      ::llvm::SmallVector<::llvm::StringRef> result;
      auto names = getSymbolNames();
      result.reserve(names.size());
      for (auto symbolAttr : names) {
        result.push_back(symbolAttr.getName());
      }
      return result;
    }

  }];
}

def WaveHyperparameterAttr : AttrDef<WaveDialect, "WaveHyperparameter"> {
  let mnemonic = "hyperparameters";
  let description = [{
    This attribute contains a dictionary mapping from symbol parameters (strings)
    to their concrete integer values. It is used to resolve symbolic shapes
    in wave.tensor types when converting to memref or vector types.

    Example:
    ```
    #wave.hyperparameters<{B = 64, H = 128, W = 256}>
    ```
  }];

  let parameters = (ins "::mlir::DictionaryAttr":$mapping);
  let assemblyFormat = "`<` $mapping `>`";

  let extraClassDeclaration = [{
    /// Get the concrete value for a symbol name, returns std::nullopt if not found
    std::optional<int64_t> getSymbolValue(::llvm::StringRef symbolName) const;

    /// Check if a symbol exists in the mapping
    bool hasSymbol(::llvm::StringRef symbolName) const;

    /// Get the number of symbols in the mapping
    size_t getNumSymbols() const { return getMapping().size(); }
  }];
}


def WaveDistributedShapeAttr : AttrDef<WaveDialect, "DistributedShape"> {
  let mnemonic = "distributed_shape";
  let description = [{
    Distributed (physical) tensor shape as affine expressions over named
    Wave symbols.
    Stores:
      - symbol_names: [WaveSymbolAttr] giving names (e.g., M, BLOCK_M, BLOCK_K)
      The i-th name maps to the s<i> in the affine expressions
      - shape: affine_map<()[s0..sN] -> (expr_0, ..., expr_{rank-1})>
      Each result expr_k defines the k-th physical dimension.

      Parsing/printing:
      #wave.distributed_shape<[M, K] -> (M, K+4)>
  }];

  let parameters = (ins
    ArrayRefParameter<"WaveSymbolAttr">:$symbol_names,
    "::mlir::AffineMap":$shape
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().getNumResults(); }
    unsigned getNumSymbols() const { return getShape().getNumSymbols(); }

    std::optional<llvm::SmallVector<int64_t>>
    getResolvedShape(::wave::WaveHyperparameterAttr hyper) const;
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEATTRS
