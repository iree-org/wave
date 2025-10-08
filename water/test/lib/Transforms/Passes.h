// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_TEST_LIB_TRANSFORMS_PASSES_H
#define WATER_TEST_LIB_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Compiler.h"
#include <memory>

#include "water/Dialect/Wave/IR/WaveDialect.h"

namespace mlir {
namespace water::test {
#define GEN_PASS_DECL
#include "Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

/// Non-inlined registration function that can be turned into a global symbol
/// visible in the library.
void registerAllPasses();
} // namespace water::test
} // namespace mlir

#endif // WATER_TEST_LIB_TRANSFORMS_PASSES_H
