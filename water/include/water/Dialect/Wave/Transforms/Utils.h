// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_UTILS_H
#define WATER_DIALECT_WAVE_TRANSFORMS_UTILS_H

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "llvm/ADT/ArrayRef.h"

namespace wave {

// Callback to generate a delayed diagnostic. The diagnostic message should be
// attached to the argument. It may or may not be emitted.
using EmitDelayedErrorFn = std::function<void(mlir::InFlightDiagnostic &)>;

// Information to emit delayed errors.
struct DelayedErrorEmitterInfo {
  // Returns the delayed error for the given operation.
  std::function<EmitDelayedErrorFn(mlir::Operation *)> getDelayedError;

  // Returns true if there are any delayed errors.
  std::function<bool()> hasDelayedErrors;
};

// Sets the attribute indicating that the operation satisfies provided normal
// forms. The presence of the attribute, in turn, performs verification of the
// normal form every time a verifier runs on the operation, including by default
// after every pass.
//
// By default, preserves existing normal forms and adds the new ones. Set
// preserve=false to replace all existing forms with the provided forms.
llvm::LogicalResult
setNormalFormPassPostcondition(llvm::ArrayRef<wave::WaveNormalForm> forms,
                               mlir::Operation *root, bool preserve = true);

// Clears all normal form attributes from the operation, effectively setting
// the normal form to None.
llvm::LogicalResult clearNormalFormPassPostcondition(mlir::Operation *root);

// Verifies if the operation, typically the root operation about to be processed
// by a pass, satisfies the required normal form by checking the presence of the
// attribute that enforces verification. Emits diagnostics and returns failures
// when it is not the case. Does *NOT* actually run verification, this is
// automated by the presence of the attribute.
llvm::LogicalResult
verifyNormalFormPassPrecondition(llvm::ArrayRef<wave::WaveNormalForm> forms,
                                 mlir::Operation *root,
                                 llvm::StringRef passName);

} // namespace wave

#endif
