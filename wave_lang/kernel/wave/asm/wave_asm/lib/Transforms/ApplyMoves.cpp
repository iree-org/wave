// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// ApplyMoves — parse and apply Conductor move/swap commands on tagged IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/ApplyMoves.h"

#include "mlir/IR/Location.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "waveasm-apply-moves"

using namespace mlir;

namespace {

/// Op mnemonics that must never be moved.
bool isPinned(Operation *op) {
  StringRef name = op->getName().stripDialect();
  return name == "condition" || name == "s_endpgm" || name == "s_barrier";
}

/// Build a map from NameLoc tag string to Operation*.
llvm::StringMap<Operation *> buildTagMap(ModuleOp module) {
  llvm::StringMap<Operation *> map;
  module.walk([&](Operation *op) {
    if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc()))
      map[nameLoc.getName().strref()] = op;
  });
  return map;
}

/// Validate that an op can be moved (not pinned, resolves to a tag).
std::string validateMovable(Operation *op, StringRef tag) {
  if (isPinned(op))
    return ("cannot move pinned op '" + tag + "'").str();
  return "";
}

/// Check that two ops are in the same block.
std::string validateSameBlock(Operation *a, StringRef tagA, Operation *b,
                              StringRef tagB) {
  if (a->getBlock() != b->getBlock())
    return ("'" + tagA + "' and '" + tagB + "' are in different blocks").str();
  return "";
}

} // namespace

namespace waveasm {

llvm::SmallVector<std::string> parseConductorCommands(llvm::StringRef text) {
  llvm::SmallVector<std::string> commands;
  llvm::SmallVector<StringRef> lines;
  text.split(lines, '\n');

  for (StringRef line : lines) {
    StringRef trimmed = line.ltrim();
    if (!trimmed.starts_with("// CONDUCTOR:"))
      continue;
    StringRef cmd = trimmed.drop_front(strlen("// CONDUCTOR:")).trim();
    if (cmd == "done")
      break;
    if (!cmd.empty())
      commands.push_back(cmd.str());
  }
  return commands;
}

MoveResult applyMoves(ModuleOp module, llvm::ArrayRef<std::string> commands) {
  auto tagMap = buildTagMap(module);

  for (auto [idx, cmd] : llvm::enumerate(commands)) {
    StringRef line(cmd);
    StringRef rest;

    auto fail = [&](const std::string &msg) -> MoveResult {
      return {false, msg, static_cast<unsigned>(idx)};
    };

    if (line.starts_with("move ")) {
      rest = line.drop_front(strlen("move "));

      // Parse: <tag> (after|before) <tag>.
      StringRef tag1, direction, tag2;
      std::tie(tag1, rest) = rest.split(' ');
      std::tie(direction, tag2) = rest.split(' ');

      if (tag1.empty() || tag2.empty() || direction.empty())
        return fail("malformed move command: '" + cmd + "'");

      if (direction != "after" && direction != "before")
        return fail("expected 'after' or 'before', got '" + direction.str() +
                    "'");

      auto it1 = tagMap.find(tag1);
      if (it1 == tagMap.end())
        return fail("unknown tag '" + tag1.str() + "'");
      auto it2 = tagMap.find(tag2);
      if (it2 == tagMap.end())
        return fail("unknown tag '" + tag2.str() + "'");

      Operation *op1 = it1->second;
      Operation *op2 = it2->second;

      std::string err = validateMovable(op1, tag1);
      if (!err.empty())
        return fail(err);

      err = validateSameBlock(op1, tag1, op2, tag2);
      if (!err.empty())
        return fail(err);

      if (direction == "after")
        op1->moveAfter(op2);
      else
        op1->moveBefore(op2);

      LDBG() << "move " << tag1 << " " << direction << " " << tag2;

    } else if (line.starts_with("swap ")) {
      rest = line.drop_front(strlen("swap "));

      StringRef tag1, tag2;
      std::tie(tag1, tag2) = rest.split(' ');

      if (tag1.empty() || tag2.empty())
        return fail("malformed swap command: '" + cmd + "'");

      auto it1 = tagMap.find(tag1);
      if (it1 == tagMap.end())
        return fail("unknown tag '" + tag1.str() + "'");
      auto it2 = tagMap.find(tag2);
      if (it2 == tagMap.end())
        return fail("unknown tag '" + tag2.str() + "'");

      Operation *op1 = it1->second;
      Operation *op2 = it2->second;

      std::string err = validateMovable(op1, tag1);
      if (!err.empty())
        return fail(err);
      err = validateMovable(op2, tag2);
      if (!err.empty())
        return fail(err);

      err = validateSameBlock(op1, tag1, op2, tag2);
      if (!err.empty())
        return fail(err);

      // Swap: move op1 after op2, then move op2 to op1's original position.
      Operation *op1Next = op1->getNextNode();
      if (op1Next == op2) {
        // Adjacent: just swap order.
        op1->moveAfter(op2);
      } else {
        Operation *op2Next = op2->getNextNode();
        if (op2Next == op1) {
          op2->moveAfter(op1);
        } else {
          // Non-adjacent: use a stable reference point.
          op1->moveAfter(op2);
          if (op1Next)
            op2->moveBefore(op1Next);
          else
            op2->moveBefore(op2->getBlock(), op2->getBlock()->end());
        }
      }

      LDBG() << "swap " << tag1 << " " << tag2;

    } else {
      return fail("unknown command: '" + cmd + "'");
    }
  }

  return {true, "", 0};
}

} // namespace waveasm
