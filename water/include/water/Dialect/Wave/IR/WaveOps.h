// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEOPS_H
#define WATER_DIALECT_WAVE_IR_WAVEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

namespace mlir {
class RewriterBase;
}

#define GET_OP_CLASSES
#include "water/Dialect/Wave/IR/WaveOps.h.inc"

#endif // WATER_DIALECT_WAVE_IR_WAVEOPS_H
