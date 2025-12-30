// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

#include "water/Dialect/NormalForm/IR/NormalFormDialect.cpp.inc"

void normalform::NormalFormDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "water/Dialect/NormalForm/IR/NormalFormOps.cpp.inc"
      >();
}
