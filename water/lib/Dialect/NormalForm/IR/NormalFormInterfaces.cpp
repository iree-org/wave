// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

#include "water/Dialect/NormalForm/IR/NormalFormAttrInterfaces.cpp.inc"
