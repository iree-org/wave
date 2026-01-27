// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Dialect/WaveASMAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace waveasm;

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

// Note: ProgramOp now uses the new attribute format from WAVEASMAttrs.td
// Verification is handled by TableGen-generated code for basic structure.
// Custom verification can be added here if needed.

//===----------------------------------------------------------------------===//
// TableGen'd Operation Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "waveasm/Dialect/WaveASMOps.cpp.inc"
