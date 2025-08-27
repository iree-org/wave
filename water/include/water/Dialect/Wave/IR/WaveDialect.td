// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVEDIALECT
#define WATER_DIALECT_WAVE_WAVEDIALECT

include "mlir/IR/DialectBase.td"

def WaveDialect : Dialect {
  let name = "wave";
  let summary = "Dialect for Wave DSL operations and types";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    // Registration functions. The bodies of these must be placed in the file
    // containing private implementations of attributes and types, respectively.
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEDIALECT
