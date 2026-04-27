// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_TRANSFORMS_EDITIR_H
#define WATER_TRANSFORMS_EDITIR_H

#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <memory>

namespace mlir::water {

/// Write the IR of `op` to a temporary file, open it in an editor, block
/// until the user presses Enter, then re-parse the file and replace the
/// operation's regions and attributes with the edited content.
///
/// `editor` overrides the editor command; when empty, $EDITOR is used.
/// `passLabel` is used in the prompt to identify which pass triggered the
/// edit.
///
/// Returns failure if any step (temp file, parse, region replacement) fails.
LogicalResult editIRInteractively(Operation *op, llvm::StringRef editor = "",
                                  llvm::StringRef passLabel = "");

using PassFilter = std::function<bool(Pass *, Operation *)>;

/// Create a PassInstrumentation that calls `editIRInteractively` before/after
/// passes matched by the respective filter. Either filter may be null to
/// skip that hook.
std::unique_ptr<PassInstrumentation>
createEditIRInstrumentation(PassFilter shouldEditBefore,
                            PassFilter shouldEditAfter,
                            llvm::StringRef editor = "");

} // namespace mlir::water

#endif // WATER_TRANSFORMS_EDITIR_H
