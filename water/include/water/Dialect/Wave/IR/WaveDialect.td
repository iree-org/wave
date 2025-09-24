// Copyright 2025 The Wave Authors
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

  let description = [{
    Passes on the Wave dialect follow the concept of normal forms. A module
    containing Wave dialect operations may satisfy conditions of several normal
    forms defined by the `WaveNormalFormAttr`. When this is the case, the module
    (or one of its ancestors), is expected to have the corresponding attribute
    attached. The attribute enforces normal form invariants in its verifiers.
    Passes can then check whether the attribute is present as a precondition for
    them to run, i.e., only apply to IR in certain normal forms. Passes may also
    update the attribute to reflect new normal forms after they complete, which
    the following passes may rely on. For example, the type inference pass may
    set the attribute indicating that all types are fully inferred, which is
    a precondition for the lowering pass to run.

    Passes on the Wave dialect follow the concept of normal forms. A module
    containing Wave dialect operations may satisfy conditions of several normal
    forms defined by the `WaveNormalFormAttr`. When this is the case, the module
    (or some its ancestor), is expected to have the corresponding attribute
    attached. The attribute enforces normal form invariants in its verifiers.
    Passes can then check whether the attribute is present as a precondition for
    them to run, i.e., only apply to IR in certain normal forms. Passes may also
    update the attribute to reflect new normal forms after they complete, which
    the following passes may rely on. For example, the type inference pass may
    set the attribute indicating that all types are fully inferred, which is a
    precondition for the lowering pass to run.

    This allows us to represent invariants at a finer granularity in the pass
    pipeline than op- or dialect-level verifiers, which otherwise would require
    introducing new operations or dialects to express the invariants in their
    verifiers. This also enables passes to specify pre- and post-conditions in a
    way that is visible in the IR.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    // Name of a hyperparameter attribute.
    const static constexpr ::llvm::StringLiteral kHyperparameterAttrName =
        "wave.hyperparameters";

    // Name of the normal form attribute.
    const static constexpr ::llvm::StringLiteral kNormalFormAttrName =
        "wave.normal_form";

    // Name of the inherent attribute for op index expressions;
    const static constexpr ::llvm::StringLiteral kIndexExprAttrName = "index";

    // Name of the attribute specifying the number of elements per thread.
    const static constexpr ::llvm::StringLiteral kElementsPerThreadAttrName = "wave.elements_per_thread";

    // Registration functions. The bodies of these must be placed in the file
    // containing private implementations of attributes and types, respectively.
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEDIALECT
