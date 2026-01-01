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
    InterfaceMethod<[{
       TODO: type verification hook
      }],
      "::llvm::LogicalResult", "verifyType",
      (ins "::mlir::Type":$type)>,
    InterfaceMethod<[{
        TODO: attribute verification hook
      }], "::llvm::LogicalResult", "verifyAttribute",
        (ins "::mlir::Attribute":$attr)>,
    InterfaceMethod<[{
       TODO: op verification hook
      }],
      "::llvm::LogicalResult", "verifyOperation",
      (ins "::mlir::Operation*":$op)>,
  ];
}


#endif // WATER_DIALECT_NORMALFORM_NORMALFORMINTERFACES
