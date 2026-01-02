// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_CONVERSION_NORMALFORMTOBUILTIN_NORMALFORMTOBUILTIN_H
#define WATER_CONVERSION_NORMALFORMTOBUILTIN_NORMALFORMTOBUILTIN_H

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_LOWERAFFINEPASS
#include "water/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the NormalForm dialect to the
/// Builtin dialect.
void populateNormalFormToBuiltinConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // WATER_CONVERSION_NORMALFORMTOBUILTIN_NORMALFORMTOBUILTIN_H
