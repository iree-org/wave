// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace wave {

#define GEN_PASS_DECL
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES_H
