// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT
#define WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT

include "mlir/IR/DialectBase.td"

def NormalFormDialect : Dialect {
  let name = "normalform";
  let summary = "Dialect for TODO...";

  let description = [{
    TODO...
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasOperationAttrVerify = 0;

  let extraClassDeclaration = [{
    /// Name of the normal form attribute.
    const static constexpr ::llvm::StringLiteral kNormalFormAttrName =
        "normalform.normal_form";
  }];
}

#endif // WATER_DIALECT_NORMALFORM_NORMALFORMDIALECT
