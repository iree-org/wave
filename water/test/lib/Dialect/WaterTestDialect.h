// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_TEST_LIB_DIALECT_WATERTESTDIALECT_H
#define WATER_TEST_LIB_DIALECT_WATERTESTDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "WaterTestDialect.h.inc"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "WaterTestDialectOps.h.inc"

#endif // WATER_TEST_LIB_DIALECT_WATERTESTDIALECT_H
