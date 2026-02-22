// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// ApplyMoves — parse and apply Conductor move/swap commands on tagged IR.
//===----------------------------------------------------------------------===//

#include "waveasm/Transforms/ApplyMoves.h"

#include "mlir/IR/Dominance.h"
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

/// Validate that an op can be moved (not pinned).
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

/// Get a human-readable name for an operation (tag or mnemonic).
std::string opName(Operation *op) {
  if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc()))
    return nameLoc.getName().str();
  return op->getName().getStringRef().str();
}

/// Check SSA dominance for a single op after it has been moved.
/// Verifies: (1) all operands are defined above the op,
///           (2) all results are defined before their users.
std::string checkDominance(Operation *op) {
  DominanceInfo domInfo(op->getParentOfType<ModuleOp>());

  // Check that every operand still dominates this op.
  for (Value operand : op->getOperands()) {
    if (!domInfo.properlyDominates(operand, op)) {
      std::string defName = "block-arg";
      if (auto *defOp = operand.getDefiningOp())
        defName = opName(defOp);
      return "moving '" + opName(op) + "' breaks dominance: operand from '" +
             defName + "' no longer defined before use";
    }
  }

  // Check that every user of this op's results is still dominated.
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!domInfo.properlyDominates(op, user)) {
        return "moving '" + opName(op) +
               "' breaks dominance: result used by '" + opName(user) +
               "' which now appears before the definition";
      }
    }
  }

  return "";
}

} // namespace

namespace waveasm {

ParseResult parseConductorCommands(llvm::StringRef text) {
  ParseResult result;
  result.success = true;
  result.failedLine = 0;

  llvm::SmallVector<StringRef> lines;
  text.split(lines, '\n');

  unsigned cmdIdx = 0;
  for (StringRef line : lines) {
    StringRef trimmed = line.ltrim();
    if (!trimmed.starts_with("// CONDUCTOR:"))
      continue;
    StringRef raw = trimmed.drop_front(strlen("// CONDUCTOR:")).trim();
    if (raw.empty())
      continue;

    if (raw.starts_with("move ")) {
      StringRef rest = raw.drop_front(strlen("move "));
      auto [tag, rest2] = rest.split(' ');
      auto [direction, refTag] = rest2.split(' ');

      if (tag.empty() || refTag.empty() || direction.empty()) {
        result.success = false;
        result.error = ("malformed move command: '" + raw + "'").str();
        result.failedLine = cmdIdx;
        return result;
      }

      if (direction == "before") {
        result.commands.push_back(MoveBefore{tag.str(), refTag.str()});
      } else if (direction == "after") {
        result.commands.push_back(MoveAfter{tag.str(), refTag.str()});
      } else {
        result.success = false;
        result.error =
            ("expected 'after' or 'before', got '" + direction + "'").str();
        result.failedLine = cmdIdx;
        return result;
      }

    } else if (raw.starts_with("swap ")) {
      StringRef rest = raw.drop_front(strlen("swap "));
      auto [tag1, tag2] = rest.split(' ');

      if (tag1.empty() || tag2.empty()) {
        result.success = false;
        result.error = ("malformed swap command: '" + raw + "'").str();
        result.failedLine = cmdIdx;
        return result;
      }

      result.commands.push_back(Swap{tag1.str(), tag2.str()});

    } else {
      result.success = false;
      result.error = ("unknown command: '" + raw + "'").str();
      result.failedLine = cmdIdx;
      return result;
    }

    ++cmdIdx;
  }

  return result;
}

/// Resolve two tags, validate movability and same-block constraint.
/// On success sets op1/op2 and returns empty string.
static std::string
resolveAndValidate(const llvm::StringMap<Operation *> &tagMap, StringRef tag,
                   StringRef ref, bool checkRefMovable, Operation *&op1,
                   Operation *&op2) {
  auto it1 = tagMap.find(tag);
  if (it1 == tagMap.end())
    return ("unknown tag '" + tag + "'").str();
  auto it2 = tagMap.find(ref);
  if (it2 == tagMap.end())
    return ("unknown tag '" + ref + "'").str();

  op1 = it1->second;
  op2 = it2->second;

  std::string err = validateMovable(op1, tag);
  if (!err.empty())
    return err;
  if (checkRefMovable) {
    err = validateMovable(op2, ref);
    if (!err.empty())
      return err;
  }
  return validateSameBlock(op1, tag, op2, ref);
}

MoveResult applyMoves(ModuleOp module, llvm::ArrayRef<Command> commands) {
  auto tagMap = buildTagMap(module);

  for (auto [idx, cmd] : llvm::enumerate(commands)) {
    auto fail = [&](const std::string &msg) -> MoveResult {
      return {false, msg, static_cast<unsigned>(idx)};
    };

    Operation *op1 = nullptr, *op2 = nullptr;

    if (auto *move = std::get_if<MoveBefore>(&cmd)) {
      std::string err =
          resolveAndValidate(tagMap, move->tag, move->refTag, false, op1, op2);
      if (!err.empty())
        return fail(err);
      op1->moveBefore(op2);
      LDBG() << "move " << move->tag << " before " << move->refTag;
      err = checkDominance(op1);
      if (!err.empty())
        return fail(err);

    } else if (auto *move = std::get_if<MoveAfter>(&cmd)) {
      std::string err =
          resolveAndValidate(tagMap, move->tag, move->refTag, false, op1, op2);
      if (!err.empty())
        return fail(err);
      op1->moveAfter(op2);
      LDBG() << "move " << move->tag << " after " << move->refTag;
      err = checkDominance(op1);
      if (!err.empty())
        return fail(err);

    } else if (auto *swap = std::get_if<Swap>(&cmd)) {
      std::string err =
          resolveAndValidate(tagMap, swap->tag1, swap->tag2, true, op1, op2);
      if (!err.empty())
        return fail(err);

      // Swap by considering adjacency cases.
      Operation *op1Next = op1->getNextNode();
      if (op1Next == op2) {
        op1->moveAfter(op2);
      } else if (op2->getNextNode() == op1) {
        op2->moveAfter(op1);
      } else {
        op1->moveAfter(op2);
        if (op1Next)
          op2->moveBefore(op1Next);
        else
          op2->moveBefore(op2->getBlock(), op2->getBlock()->end());
      }
      LDBG() << "swap " << swap->tag1 << " " << swap->tag2;
      err = checkDominance(op1);
      if (!err.empty())
        return fail(err);
      err = checkDominance(op2);
      if (!err.empty())
        return fail(err);
    }
  }

  return {true, "", 0};
}

} // namespace waveasm
