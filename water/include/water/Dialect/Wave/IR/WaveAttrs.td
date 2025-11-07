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
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

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

// WaveMmaKind encoding is adopted from IREE's 0xABCD scheme:
// A:vendor, B:architecture, C:type-class group, D=variant.
// See IREE docs/tests for full details: https://github.com/iree-org/iree/blob/c65dc6dc28491b4768a72ff9d563edeb377627a9/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td#L136

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
  I32EnumAttrCase<"F32_16x16x32_F16", 0x1320, "f32_16x16x32_f16">,
  I32EnumAttrCase<"F32_32x32x16_F16", 0x1321, "f32_32x32x16_f16">,
  I32EnumAttrCase<"F32_16x16x32_BF16", 0x1322, "f32_16x16x32_bf16">,
  I32EnumAttrCase<"F32_32x32x16_BF16", 0x1323, "f32_32x32x16_bf16">,

]> {
  let cppNamespace = "::wave";
  let genSpecializedAttr = 0;
}

def WaveMmaKindAttr : EnumAttr<WaveDialect, WaveMmaKindEnum, "mma_kind"> {
  let assemblyFormat = "`<` $value `>`";
  let extraClassDeclaration = [{
    /// Returns the MMA specification (MNK tile sizes and element types)
    /// for the given kind.
    static ::wave::WaveMmaSpec getSpec(::mlir::MLIRContext *ctx,
                                       ::wave::WaveMmaKind kind);
    /// Returns the (M,N,K) shape for the given kind.
    static std::tuple<uint32_t, uint32_t, uint32_t>
    getShape(::mlir::MLIRContext *ctx, ::wave::WaveMmaKind kind);
  }];
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
def NORMAL_FORM_MEMORY_ONLY_TYPES : I32BitEnumAttrCaseBit<
    "MemoryOnlyTypes",           3, "memory_only_types">;

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
  NORMAL_FORM_MEMORY_ONLY_TYPES,

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
// Workgroup dim enum and attribute
//-----------------------------------------------------------------------------

def WaveWorkgroupDimEnum
    : I32EnumAttr<"WaveWorkgroupDim", "Workgroup dimensions", [
  I32EnumAttrCase<"X", 0, "x">,
  I32EnumAttrCase<"Y", 1, "y">,
  I32EnumAttrCase<"Z", 2, "z">,
]> {
  let cppNamespace = "::wave";
  let genSpecializedAttr = 0;
}

def WaveWorkgroupDimAttr
    : EnumAttr<WaveDialect, WaveWorkgroupDimEnum, "workgroup_dim"> {
  let assemblyFormat = "`<` $value `>`";
}

//-----------------------------------------------------------------------------
// Constraint attributes
//-----------------------------------------------------------------------------

def HardwareConstraintAttr : AttrDef<WaveDialect, "HardwareConstraint"> {
  let mnemonic = "hardware_constraint";
  let description = [{
    A constraint of the form:
    ```mlir
    #wave.hardware_constraint<threads_per_wave = 64,
                              waves_per_block = [1, 1, 1],
                              mma_type = #wave.mma_kind<f32_16x16x16_f16>,
                              vector_shapes = {M = 1, N = 1, K = 64},
                              max_bits_per_load = 128>
    ```

    specifies that the hardware supports N `threads_per_wave`
    and that we want all mma operations in the microkernel to be
    mapped to a hardware mma instruction of shape (16x16x16).
    This translates to a hardware specific index constraint.
    `waves_per_block` is used to determine how many waves should make up
    one workgroup.

    Not all computation graphs have mma operators in them. In
    these situations, the user can specify the vector shape they
    want to tile to by specifying the `vector_shapes` dictionary
    which maps a tensor dimension to its corresponding tile size.

    Both mma constraints and vector shapes can be specified, but
    the mapping from symbols to shapes should be injective.

    `max_bits_per_load` specifies the maximum number of bits that can
    be loaded with a single load instruction.
  }];
  let parameters = (ins "unsigned":$threads_per_wave,
                        OptionalArrayRefParameter<"unsigned">:$waves_per_block,
                        OptionalParameter<"::wave::WaveMmaKindAttr">:$mma_type,
                        OptionalParameter<"::mlir::DictionaryAttr">:$vector_shapes,
                        DefaultValuedParameter<"unsigned", "128">:$max_bits_per_load);

  let assemblyFormat = [{
    `<` `threads_per_wave` `=` $threads_per_wave
    (`,` `waves_per_block` `=` `[` $waves_per_block^ `]`)?
    (`,` `mma_type` `=` $mma_type^)?
    (`,` `vector_shapes` `=` $vector_shapes^)?
    (`,` `max_bits_per_load` `=` $max_bits_per_load^)? `>`
  }];

  let genVerifyDecl = 1;
}

def WorkgroupConstraintAttr : AttrDef<WaveDialect, "WorkgroupConstraint"> {
  let mnemonic = "workgroup_constraint";
  let description = [{
    A constraint of the form:
    ```mlir
    #wave.workgroup_constraint<dim = <"M">,
                               tile_size = <[BLOCK_M] -> (BLOCK_M)>,
                               workgroup_dim = <x>,
                               primary = true>
    ```

    specifies that we want to distribute dimension M along workgroup dim `x`
    with a tile size of BLOCK_M resulting in M // BLOCK_M workgroups along that
    dimension.
  }];
  let parameters = (ins "::wave::WaveSymbolAttr":$dim,
                        "::wave::WaveExprListAttr":$tile_size,
                        "::wave::WaveWorkgroupDimAttr":$workgroup_dim,
                        DefaultValuedParameter<"bool", "true">:$primary);

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def WaveConstraintAttr : AttrDef<WaveDialect, "WaveConstraint"> {
  let mnemonic = "wave_constraint";
  let description = [{
    A constraint of the form:
    ```mlir
    #wave.wave_constraint<dim = <"K">,
                          tile_size = <[WAVE_K] -> (WAVE_K)>,
                          wg_constraint = #wg_constraint>
    ```

    that we want distribute the K dimension among multiple waves with
    each wave operating on a tile size of WAVE_K. The assumption is
    that the K dimension has already been distributed among workgroups.
    If the K dimension has been distributed among workgroups with a
    tile size of BLOCK_K, then the number of waves along the K dimension
    is given by BLOCK_K // WAVE_K.
  }];
  let parameters = (ins "::wave::WaveSymbolAttr":$dim,
                        "::wave::WaveExprListAttr":$tile_size,
                        OptionalParameter<"::wave::WorkgroupConstraintAttr">:$wg_constraint);

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def TilingConstraintAttr : AttrDef<WaveDialect, "TilingConstraint"> {
  let mnemonic = "tiling_constraint";
  let description = [{
    A constraint of the form:
    ```mlir
    #wave.tiling_constraint<dim = <"K">,
                            tile_size = <[TILE_K] -> (TILE_K)>>
    ```

    specifies that we want to distribute the K dimension among multiple threads. With
    each thread operating on a tile size of TILE_K.
  }];
  let parameters = (ins "::wave::WaveSymbolAttr":$dim,
                        "::wave::WaveExprListAttr":$tile_size);

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def DeviceConstraintAttr : AttrDef<WaveDialect, "DeviceConstraint"> {
  let mnemonic = "device_constraint";
  let description = [{
    A constraint of the form:
    ```mlir
    #wave.device_constraint<dim = <"M">, tile_size = <[DEVICE_M] -> (DEVICE_M)>, device_dim = 0>
    ```

    specifies that we want to distribute dimension `M` along the device with a tile size of `DEVICE_M`.
    This translates to an index constraint for all tensors of the shape [M, ?] ->
    index += (device_id * DEVICE_M, 0), where device_id is the id
    of the device on which the tensor is located.

    Device id is a tuple of (dev_id_x, devid_y, dev_id_z) where
    dev_id_x is the id of the device in the x dimension, dev_id_y is
    the id of the device in the y dimension and dev_id_z is the id of the
    device in the z dimension. The device id is used to compute the index
    offset for the tensor along the specified dimension.

    Device constraints can be applied for multiple dimensions. For example:

    ```mlir
    wave.constraints = [
      #wave.device_constraint<dim = <"M">, <tile_size = [DEVICE_M] -> (DEVICE_M)>, device_dim = 0>
      #wave.device_constraint<dim = <"N">, <tile_size = [DEVICE_N] -> (DEVICE_N)>, device_dim = 1>
    ]
    ```

    specifies that we want to distribute `DEVICE_M` x `DEVICE_N` tiles of the tensor
    across the devices, where `DEVICE_M` and `DEVICE_N` are the tile sizes for
    dimensions `M` and `N` respectively.
  }];
  let parameters = (ins "::wave::WaveSymbolAttr":$dim,
                        "::wave::WaveExprListAttr":$tile_size,
                        "unsigned":$device_dim);

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
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

def WaveExprListAttr : AttrDef<WaveDialect, "WaveExprList"> {
  let mnemonic = "expr_list";
  let description = [{
  A tuple of affine expressions over named Wave symbols.
  This attribute is used to encode:
    • Bounds expressions for read/write ops
    • Distributed (physical) shapes for allocations

    Stores:
      - symbol_names: [WaveSymbolAttr] giving names (e.g., M, BLOCK_M, BLOCK_K)
      The i-th name maps to the s<i> in the affine expressions
      - shape: affine_map<()[s0..sN] -> (expr_0, ..., expr_{rank-1})>
      Each result expr_k defines the k-th physical dimension.

      Parsing/printing:
      #wave.expr_list<[M, K] -> (M, K+4)>
  }];

  let parameters = (ins
    ArrayRefParameter<"WaveSymbolAttr">:$symbols,
    "::mlir::AffineMap":$map
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    unsigned getRank() const { return getMap().getNumResults(); }
    unsigned getNumSymbols() const { return getMap().getNumSymbols(); }
    uint64_t getSize() const { return getMap().getNumResults(); }

    std::optional<llvm::SmallVector<int64_t>>
    getResolvedShape(::wave::WaveHyperparameterAttr hyper) const;
  }];
}

def WaveReadWriteBoundsAttr : AttrDef<WaveDialect, "WaveReadWriteBounds"> {
  let mnemonic = "read_write_bounds";
  let description = [{
    This attribute contains a dictionary mapping from symbolic dimension (strings)
    to a WaveExprListAttr specifying the bounds of the read/write operations
    this is attached to.

    Example:
    ```
    #wave.read_write_bounds<{M = #wave.expr_list<[BLOCK_M] -> BLOCK_M * 2>}>
    ```
  }];

  let parameters = (ins "::mlir::DictionaryAttr":$mapping);
  let assemblyFormat = "`<` $mapping `>`";

  let extraClassDeclaration = [{
    /// Check if a symbolic dimension exists in the mapping
    bool hasSymbol(::llvm::StringRef symbolicDim) const {
      return getMapping().get(symbolicDim) != nullptr;
    }

    /// Get the number of symbols in the mapping
    size_t getNumSymbols() const { return getMapping().size(); }
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEATTRS
