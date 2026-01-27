// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace waveasm;

// Note: Attribute definitions (GET_ATTRDEF_CLASSES) are included in WaveASMDialect.cpp
// to ensure storage types are complete when registering attributes.
