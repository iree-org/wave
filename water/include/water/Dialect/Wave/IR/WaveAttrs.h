// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEATTRS_H
#define WATER_DIALECT_WAVE_IR_WAVEATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace wave {

struct WaveMmaSpec {
  int64_t m;
  int64_t n;
  int64_t k;
  mlir::Type aType;
  mlir::Type bType;
  mlir::Type accType;
};

} // namespace wave

#include "water/Dialect/Wave/IR/WaveEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.h.inc"

namespace wave {

/// Verify that all provided ExprAttr attributes have the same rank. Returns
/// success if all ranks match, failure otherwise.
llvm::LogicalResult
verifyExprAttrsSameRank(llvm::ArrayRef<WaveExprListAttr> exprs);

/// Verify that all provided ExprAttr attributes have no symbols (i.e., they are
/// constant expressions). Returns success if all have zero symbols, failure
/// otherwise.
llvm::LogicalResult
verifyExprAttrsNoSymbols(llvm::ArrayRef<WaveExprListAttr> exprs);

} // namespace wave

namespace wave::detail {
/// Verifies that the provided operation and its descendants satisfies the
/// required normal forms. Emits diagnostics if requested, otherwise just
/// returns failure.
llvm::LogicalResult verifyNormalFormAttr(mlir::Operation *root,
                                         wave::WaveNormalForm form,
                                         bool emitDiagnostics);
} // namespace wave::detail

#endif // WATER_DIALECT_WAVE_IR_WAVEATTRS_H
