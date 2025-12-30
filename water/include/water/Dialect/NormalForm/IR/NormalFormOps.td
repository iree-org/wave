// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/NormalForm/IR/NormalFormDialect.td"
include "water/Dialect/NormalForm/IR/NormalFormInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"



#ifndef WATER_DIALECT_NORMALFORM_NORMALFORMOPS
#define WATER_DIALECT_NORMALFORM_NORMALFORMOPS

//-----------------------------------------------------------------------------
// Base class for all NormalForm operations.
//-----------------------------------------------------------------------------

class NormalFormOp<string mnemonic, list<Trait> traits = []> :
    Op<NormalFormDialect, mnemonic, traits>;

//-----------------------------------------------------------------------------
// Attribute constraints.
//-----------------------------------------------------------------------------

/// An array of attributes implementing NormalFormAttrInterface.
def NormalFormAttrArray : TypedArrayAttrBase<NormalFormAttrInterface,
    "array of NormalFormAttrInterface">;

//-----------------------------------------------------------------------------
// Structure Ops
//-----------------------------------------------------------------------------

def ModuleOp : NormalFormOp<"module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "A container that enforces IR invariants specified by normal form attributes.";
  let description = [{
    The `normalform.module` operation defines a scoped region whose contents must
    satisfy the constraints specified by the attached normal form attributes. Each
    attribute must implement `NormalFormAttrInterface`, which provides hooks for
    verifying types, attributes, and operations within the module.

    During region verification, the module walks all nested operations and
    invokes the interface methods (`verifyType`, `verifyAttribute`,
    `verifyOperation`) on each element for all normal form attributes.
    Normal forms are verified in the order given by the normal form array attribute,
    walking over each operation in pre-order while also visiting its types, followed by
    its attributes.

    Example:

    ```mlir
    // Enforce that all tensor types are fully specified.
    normalform.module @my_kernel [#wave.normal_form<full_types>] {
      func.func @compute(%arg: !wave.tensor<[64, 128] of f32>) {
        return
      }
    }

    // Multiple normal form attributes from different dialects.
    normalform.module @validated [#wave.normal_form<full_types>, #other.normal_form] {
      func.func @compute(%arg: !wave.tensor<[64, 128] of f32>) {
        return
      }
    }
    ```
  }];
  let arguments = (ins NormalFormAttrArray:$normal_forms, OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    ($sym_name^)? $normal_forms attr-dict-with-keyword $bodyRegion
  }];

  let builders = [
    OpBuilder<(ins
      CArg<"::llvm::ArrayRef<NormalFormAttrInterface>">:$normal_forms,
      CArg<"std::optional<::llvm::StringRef>", "{}">:$name)>
  ];

  let extraClassDeclaration = [{
    /// Construct a module from the given location with an optional name.
    static ModuleOp create(::mlir::Location loc,
                           ::llvm::ArrayRef<NormalFormAttrInterface> normalForms,
                           std::optional<::llvm::StringRef> name = {});

    /// Return the name of this module if present.
    std::optional<::llvm::StringRef> getName() { return getSymName(); }

    /// Checks whether the normal forms passed in `normalForms` apply to this
    /// module and attaches them to the module if true. Returns true if the module was changed.
    bool inferNormalForms(::llvm::ArrayRef<NormalFormAttrInterface> normalForms);

    /// Checks wheter a given normal form applies to this module.
    ::llvm::LogicalResult
    verifyNormalForm(NormalFormAttrInterface normalForm, bool emitDiagnostics);

    /// Adds normal form attributes to the module. Returns true if the module was changed.
    bool addNormalForms(::llvm::ArrayRef<NormalFormAttrInterface> normalForms);

    /// Removes normal form attributes from the module. Returns true if the module was changed.
    bool removeNormalForms(::llvm::ArrayRef<NormalFormAttrInterface> normalForms);

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() const { return true; }
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}


#endif // WATER_DIALECT_NORMALFORM_NORMALFORMOPS
