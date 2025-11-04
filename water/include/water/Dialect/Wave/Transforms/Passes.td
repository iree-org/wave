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
    "::mlir::math::MathDialect",
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

def WaterWavePropagateElementsPerThreadPass
    : Pass<"water-wave-propagate-elements-per-thread"> {
  let summary = "Replace register-resident tensor types with vector types";
  let description = [{
    Performs forward and backward sparse dataflow analysis propagating
    information necessary to deduce the number of elements that each operation
    processes per thread. This information comes from either:

      * TODO: specialized operations that have constraints on elements owned by
        each thread, such as MMA;
      * memory access operations with dedicated attributes;
      * TODO: compilation options.

    All operations using register-resident Wave dialect tensor types are
    expected to implement WaveElementsPerThreadOpInterface or have the
    NoOpElementsPerThreadOpTrait to propagate this information. Other traits
    provide implementations for common cases.

    Reports errors if a conflict is detected, for example, if a value is read
    with 4 elements per thread and then written with 8 elements per thread or
    used in a context where a different number is expected.

    Rewrites the IR to replace register-resident tensor types with vector types
    indicating the number of elements processed by each thread.
  }];
}

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES
