// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMATTRS_H
#define WaveASM_DIALECT_WAVEASMATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#define GET_ATTRDEF_CLASSES
#include "waveasm/Dialect/WaveASMAttrs.h.inc"

#endif // WaveASM_DIALECT_WAVEASMATTRS_H
