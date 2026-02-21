// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_TRANSFORMS_APPLYMOVES_H
#define WAVEASM_TRANSFORMS_APPLYMOVES_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace waveasm {

/// Result of applying a sequence of move commands to a module.
struct MoveResult {
  bool success;
  std::string error;
  unsigned failedCommand;
};

/// Apply a sequence of Conductor move commands to a tagged module.
///
/// Commands are strings of the form:
///   move <tag> after <tag>
///   move <tag> before <tag>
///   swap <tag> <tag>
///
/// The module must already have NameLoc tags attached (via TagInstructions).
/// Returns a MoveResult indicating success or the first error encountered.
MoveResult applyMoves(mlir::ModuleOp module,
                      llvm::ArrayRef<std::string> commands);

/// Parse CONDUCTOR command lines from raw file text.
/// Scans for lines matching `// CONDUCTOR: <command>` and collects them
/// until a `done` command is found or input is exhausted.
llvm::SmallVector<std::string> parseConductorCommands(llvm::StringRef text);

} // namespace waveasm

#endif // WAVEASM_TRANSFORMS_APPLYMOVES_H
