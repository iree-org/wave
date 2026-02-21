// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_TRANSFORMS_APPLYMOVES_H
#define WAVEASM_TRANSFORMS_APPLYMOVES_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <variant>

namespace waveasm {

/// Move an op before a reference op.
struct MoveBefore {
  std::string tag;
  std::string refTag;
};

/// Move an op after a reference op.
struct MoveAfter {
  std::string tag;
  std::string refTag;
};

/// Swap two ops.
struct Swap {
  std::string tag1;
  std::string tag2;
};

using Command = std::variant<MoveBefore, MoveAfter, Swap>;

/// Result of applying a sequence of move commands to a module.
struct MoveResult {
  bool success;
  std::string error;
  unsigned failedCommand;
};

/// Result of parsing CONDUCTOR commands from raw text.
struct ParseResult {
  bool success;
  std::string error;
  unsigned failedLine; // 0-based index of the failing CONDUCTOR line.
  llvm::SmallVector<Command> commands;
};

/// Parse CONDUCTOR command lines from raw file text.
/// Scans for lines matching `// CONDUCTOR: <command>` and parses them
/// into typed Command structs until `done` or end of input.
ParseResult parseConductorCommands(llvm::StringRef text);

/// Apply a sequence of parsed Conductor commands to a tagged module.
///
/// The module must already have NameLoc tags attached (via TagInstructions).
/// Returns a MoveResult indicating success or the first error encountered.
MoveResult applyMoves(mlir::ModuleOp module, llvm::ArrayRef<Command> commands);

} // namespace waveasm

#endif // WAVEASM_TRANSFORMS_APPLYMOVES_H
