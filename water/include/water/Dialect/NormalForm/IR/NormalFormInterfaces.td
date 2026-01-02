// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES
#define WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU target attribute interface.
//===----------------------------------------------------------------------===//
def NormalFormAttrInterface : AttrInterface<"NormalFormAttrInterface"> {
  let description = [{
    TODO
  }];
  let cppNamespace = "::normalform";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
       Verify the type using this normal form attribute.
      }],
      /*returnType=*/"::llvm::LogicalResult",
      /*name=*/"verifyType",
      /*arguments=*/(ins "::mlir::Type":$type),
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
      /*arguments=*/(ins "::mlir::Attribute":$attr),
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
      /*arguments=*/(ins "::mlir::Operation*":$op),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::success();
      }]
    >,
  ];
}


#endif // WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES
