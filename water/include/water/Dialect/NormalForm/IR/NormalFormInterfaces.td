// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES
#define WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Normal form attribute interface.
//===----------------------------------------------------------------------===//
def NormalFormAttrInterface : AttrInterface<"NormalFormAttrInterface"> {
  let description = [{
    Interface for attributes that define normal form constraints on IR.

    A normal form attribute specifies invariants that types, attributes, and
    operations must satisfy. When attached to a `normalform.module`, all
    contained IR is verified against the constraints defined by the attribute.

    Implementers should override one or more of `verifyType`, `verifyAttribute`,
    or `verifyOperation` to define custom verification logic. The default
    implementations return success.
  }];
  let cppNamespace = "::normalform";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
       Verify the type using this normal form attribute.
      }],
      /*returnType=*/"::llvm::LogicalResult",
      /*name=*/"verifyType",
      /*arguments=*/(ins "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError, "::mlir::Type":$type),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::success();
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Verify the attribute using this normal form attribute.
      }],
      /*returnType=*/"::llvm::LogicalResult",
      /*name=*/"verifyAttribute",
      /*arguments=*/(ins "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError, "::mlir::Attribute":$attr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::success();
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
       Verify the operation using this normal form attribute.
      }],
      /*returnType=*/"::llvm::LogicalResult",
      /*name=*/"verifyOperation",
      /*arguments=*/(ins "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError, "::mlir::Operation*":$op),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::success();
      }]
    >,
  ];
}


#endif // WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES
