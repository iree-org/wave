// Copyright 2025 The Wave Authors
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
// Structure Ops
//-----------------------------------------------------------------------------

def ModuleOp : NormalFormOp<"module", [
    AffineScope, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "A top level compilation unit that define a Wave module";
  let description = [{
    This op defines a Wave module using a MLIR region.

    ```
    normalform.module @symbol_name <normalform> {
        ...
    }
    ```
  }];
  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name, NormalFormAttrInterface:$normal_form);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    ($sym_name^)? $normal_form attr-dict-with-keyword $bodyRegion
  }];

  // let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $bodyRegion";
  let builders = [
    OpBuilder<(ins
      CArg<"::normalform::NormalFormAttrInterface">:$normal_form,
      CArg<"std::optional<::llvm::StringRef>", "{}">:$name)>
  ];

  let extraClassDeclaration = [{
    /// Construct a module from the given location with an optional name.
    static ModuleOp create(::mlir::Location loc,
                               normalform::NormalFormAttrInterface normalForm,
                               std::optional<::llvm::StringRef> name = {});

    /// Return the name of this module if present.
    std::optional<::llvm::StringRef> getName() { return getSymName(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let hasVerifier = 0;
  let hasRegionVerifier = 1;

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}


#endif // WATER_DIALECT_NORMALFORM_NORMALFORMOPS
