// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVETYPES_H
#define WATER_DIALECT_WAVE_IR_WAVETYPES_H

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Types.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveTypes.h.inc"

#endif // WATER_DIALECT_WAVE_IR_WAVETYPES_H
