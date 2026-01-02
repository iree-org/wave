// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_CONVERSION_PASSES_H
#define WATER_CONVERSION_PASSES_H

#include "water/Conversion/NormalFormToBuiltin/NormalFormToBuiltin.h"

namespace mlir {
namespace water {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "water/Conversion/Passes.h.inc"

} // namespace water
} // namespace mlir
#endif // WATER_CONVERSION_PASSES_H
