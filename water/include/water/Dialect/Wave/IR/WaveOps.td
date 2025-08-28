// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "mlir/IR/OpBase.td"

#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

def MmaOp : Op<WaveDialect, "mma"> {
  let summary = "Matrix multiply and accumulate";

  let arguments = (ins
    Arg<WaveTensorType, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorType, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorType, "Accumulator for addition">:$accumulator
  );

  let results = (outs
    Res<WaveTensorType, "Accumulated result">:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $accumulator attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
