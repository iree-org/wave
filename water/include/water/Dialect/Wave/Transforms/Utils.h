// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"

namespace wave {

// Sets the attribute indicating that the operation satisfies provided normal
// forms. The presence of the attribute, in turn, performs verification of the
// normal form every time a verifier runs on the operation, including by default
// after every pass.
//
// By default, preserves existing normal forms and adds the new form. Set
// preserve=false to replace all existing forms with the provided form.
llvm::LogicalResult setNormalFormPassPostcondition(wave::WaveNormalForm form,
                                                   mlir::Operation *root,
                                                   bool preserve = true);

// Clears all normal form attributes from the operation, effectively setting
// the normal form to None.
llvm::LogicalResult clearNormalFormPassPostcondition(mlir::Operation *root);

// Verifies if the operation, typically the root operation about to be processed
// by a pass, satisfies the required normal form by checking the presence of the
// attribute that enforces verification. Emits diagnostics and returns failures
// when it is not the case. Does *NOT* actually run verification, this is
// automated by the presence of the attribute.
llvm::LogicalResult verifyNormalFormPassPrecondition(wave::WaveNormalForm form,
                                                     mlir::Operation *root,
                                                     llvm::StringRef passName);

} // namespace wave
