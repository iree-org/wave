// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVETYPES
#define WATER_DIALECT_WAVE_WAVETYPES

include "water/Dialect/Wave/IR/WaveDialect.td"
include "mlir/IR/AttrTypeBase.td"

def WaveTensorType : TypeDef<WaveDialect, "WaveTensor"> {
  let mnemonic = "tensor";
  let description = [{
    Tensor with symbolic shapes used in Wave dialects. The exact shape may be
    unknown in the earlier stages and will be inferred later.
  }];

  let parameters = (ins
    // TODO: extend this to a larger scope of symbol expressions as needed.
    // TODO: consider supporting yet-uninferred dimensions or groups of
    // dimensions, e.g., we may know that the trailing dimensions must be
    // certain symbols but not know the expressions for leading dimensions or
    // maybe even their number.

    ArrayRefParameter<
      "::wave::WaveSymbolAttr",
      "Shape of the tensor when fully inferred, missing otherwise">:$shape,
    // "::wave::WaveSymbolAttr":$shape,
    TypeParameter<"bool", "">:$fully_specified,
    TypeParameter<"::mlir::Type", "Type of the tensor elements">:$element_type
  );

  let assemblyFormat = "`<` custom<TensorShape>($shape, $fully_specified) `of` $element_type `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Return the rank of the fully-specified tensor.
    size_t getRank() {
      assert(getFullySpecified() && "requested rank of an under-specified tensor");
      return getShape().size();
    }
  }];
}

#endif // WATER_DIALECT_WAVE_WAVETYPES
