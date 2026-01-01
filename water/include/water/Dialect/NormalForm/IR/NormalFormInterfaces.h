// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
#define WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace normalform {} // namespace normalform

#include "water/Dialect/NormalForm/IR/NormalFormAttrInterfaces.h.inc"

#endif // WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
