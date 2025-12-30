// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
#define WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include "water/Dialect/NormalForm/IR/NormalFormAttrInterfaces.h.inc"

namespace llvm {
template <>
struct PointerLikeTypeTraits<normalform::NormalFormAttrInterface>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline normalform::NormalFormAttrInterface
  getFromVoidPointer(void *p) {
    return normalform::NormalFormAttrInterface(
        mlir::Attribute::getFromOpaquePointer(p));
  }
};
} // namespace llvm

#endif // WATER_DIALECT_NORMALFORM_IR_NORMALFORMINTERFACES_H
