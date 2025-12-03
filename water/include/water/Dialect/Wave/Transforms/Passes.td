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

def WaterWaveInferIndexExprsPass : Pass<"water-wave-infer-index-exprs"> {
  let summary = "Infer index expressions for Wave operations";
  let description = [{
    Performs forward and backward sparse dataflow analysis to infer index
    expressions for wave-tensor typed operations when possible. Index
    expressions indicate how the index space of the tensor is distributed
    across devices, workgroups, workitems and registers within a single
    workitem. They take the form of a dictionary with symbols used in the
    shape of the tensor-typed value as keys and triples of affine functions
    of device, workgroup, workitem and register identifiers as results.
    The elements of the triple indicate:

      - the offset of the first accessed element along the dimension by
        the workitem;
      - the total number of elements accessed;
      - the step between elements accessed by the same workitem.

    Index expressions are a property of a value. As the result of this pass,
    they will be attached as the `index` attribute of the operation producing
    the value. Some operations, namely MmaOp and WriteOp, may additionally
    contain indexes of their operands for legacy reasons.

    The inference starts by initializing the indexes based on the
    `wave.constraints` attribute for special operations, currently:

      - MMA operations

    to ensure the data is distributed to registers in a way that is expected
    by the matrix multiplication engine. It then propagates index expressions
    forward and backward by JOINing them. The JOIN function is as follows:

      - lattice top and bottom have the conventional neutral and saturating
        semantics;
      - two non-identical index expressions may be joined if their
        coefficients for device and workgroup indices are identical and
        one of them does not depend on workitem or register indices;
        the result of such JOIN is the expression that does depend on
        workitem or register indices;
      - all other index expressions are deemed incompatible and JOINing
        them immediately reaches lattice top.

    The pass reports when the index expression lattice reaches top for the
    first time for some value, which is indicative of a conflict. This means
    the program is semantically invalid and must be updated to ensure
    compatible indexing, e.g., by storing intermediate values into memory
    with one index expression, synchronizing, and reading back with another.
  }];
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
