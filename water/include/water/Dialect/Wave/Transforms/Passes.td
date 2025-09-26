// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerWaveToMLIRPass : Pass<"lower-wave-to-mlir"> {
  let summary = "Lower Wave dialect to upstream MLIR dialects";
  let description = [{
    This pass lowers operations from the Wave dialect to upstream MLIR
    dialects, notably arith and vector.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def WaterWaveInferTypesPass : Pass<"water-wave-infer-types"> {
  let summary = "Infer fully-specified Wave dialect types";
  let description = [{
    Performs forward and backward sparse dataflow analysis propagating shape
    information for Wave tensor types. All operations using these types are
    expected to implement WaveInferTypeOpInterface to indicate rules for shape
    propagation across them. Traits provide implementations for common cases.

    Reports errors if a type conflict is detected, for example, if a value with
    `any` shape would need two different concrete shapes based on propagation
    from different control flow paths or from different directions.
  }];

  let options = [
    Option</*varName=*/"force",
           /*arg=*/"force",
           /*valueType=*/"bool",
           /*default=*/"false",
           "Keep processing after error emission without updating the type. "
           "May construct invalid IR useful for debugging">
  ];
}

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES
