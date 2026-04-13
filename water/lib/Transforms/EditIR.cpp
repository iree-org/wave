// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/EditIR.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Resolve which editor command to use: the explicit override if non-empty,
/// otherwise $EDITOR. Returns an empty string when neither is available.
static std::string resolveEditor(StringRef editorOverride) {
  if (!editorOverride.empty())
    return editorOverride.str();
  if (auto env = llvm::sys::Process::GetEnv("EDITOR"))
    return *env;
  return "";
}

/// Try to open `path` in `editorCmd`. Failures are non-fatal (the user can
/// always open the file manually via the printed path).
static void tryOpenInEditor(StringRef editorCmd, StringRef path,
                            Operation *op) {
  auto program = llvm::sys::findProgramByName(editorCmd);
  if (std::error_code ec = program.getError()) {
    op->emitWarning() << "could not find editor '" << editorCmd
                      << "': " << ec.message();
    return;
  }

  SmallVector<StringRef> args = {editorCmd, path};
  std::string errMsg;
  llvm::sys::ExecuteNoWait(*program, args, /*Env=*/std::nullopt,
                           /*Redirects=*/{}, /*MemoryLimit=*/0, &errMsg);
  if (!errMsg.empty())
    op->emitWarning() << "failed to launch editor: " << errMsg;
}

LogicalResult mlir::water::editIRInteractively(Operation *op, StringRef editor,
                                               StringRef passLabel) {
  MLIRContext *context = op->getContext();

  // Write the current IR to a temporary file.
  SmallString<128> tmpPath;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("water-edit", "mlir", tmpPath);
  if (ec)
    return op->emitError() << "failed to create temporary file: "
                           << ec.message();

  llvm::FileRemover tmpFileRemover(tmpPath);

  { // Scope ensures the file is flushed and closed before the editor opens it.
    llvm::raw_fd_ostream tmpFile(tmpPath, ec);
    if (ec)
      return op->emitError()
             << "failed to open temporary file for writing: " << ec.message();
    if (!passLabel.empty())
      tmpFile << "// -----// IR Edit " << passLabel << " //----- //\n";
    op->print(tmpFile);
    tmpFile << '\n';
  }

  std::string editorCmd = resolveEditor(editor);
  if (!editorCmd.empty())
    tryOpenInEditor(editorCmd, tmpPath, op);

  llvm::outs() << "=== water-edit-ir";
  if (!passLabel.empty())
    llvm::outs() << " " << passLabel;
  llvm::outs() << " ===\n"
               << "IR written to: " << tmpPath << "\n"
               << "Press Enter to continue, or 'q' to abort compilation...\n";
  llvm::outs().flush();

  while (true) {
    int first = std::getchar();
    if (first == EOF || first == 'q' || first == 'Q')
      return failure();
    // Drain the rest of the line so nothing leaks into the next iteration.
    for (int c = first; c != '\n' && c != EOF;)
      c = std::getchar();

    auto fileOrErr = llvm::MemoryBuffer::getFile(tmpPath);
    if (!fileOrErr)
      return op->emitError() << "failed to read edited file: "
                             << fileOrErr.getError().message();

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());

    std::string diagStr;
    llvm::raw_string_ostream diagOS(diagStr);
    SourceMgrDiagnosticHandler diagHandler(sourceMgr, context, diagOS);

    OwningOpRef<Operation *> parsedOp =
        parseSourceFile<Operation *>(sourceMgr, context);

    if (parsedOp) {
      if (op->getName() != (*parsedOp)->getName())
        return op->emitError()
               << "edited IR has a different top-level operation ('"
               << (*parsedOp)->getName() << "' vs '" << op->getName() << "')";

      unsigned numRegions = op->getNumRegions();
      for (unsigned i = 0; i < numRegions; ++i) {
        Region &existing = op->getRegion(i);
        Region &parsed = (*parsedOp)->getRegion(i);
        existing.getBlocks().clear();
        existing.getBlocks().splice(existing.getBlocks().end(),
                                    parsed.getBlocks());
      }
      op->setAttrs((*parsedOp)->getAttrDictionary());
      op->copyProperties((*parsedOp)->getPropertiesStorage());
      return success();
    }

    // Parse failed -- print diagnostics and re-prompt.
    // The file is left untouched so that line numbers remain correct.
    llvm::outs() << "Failed to parse edited IR:\n"
                 << diagStr
                 << "Press Enter to retry, or 'q' to abort compilation...\n";
    llvm::outs().flush();

    if (!editorCmd.empty())
      tryOpenInEditor(editorCmd, tmpPath, op);
  }
}

//===----------------------------------------------------------------------===//
// PassInstrumentation for --water-edit-ir-{before,after}
//===----------------------------------------------------------------------===//

namespace {

class EditIRInstrumentation : public PassInstrumentation {
public:
  EditIRInstrumentation(water::PassFilter shouldEditBefore,
                        water::PassFilter shouldEditAfter, StringRef editor)
      : shouldEditBefore(std::move(shouldEditBefore)),
        shouldEditAfter(std::move(shouldEditAfter)), editor(editor.str()) {}

  void runBeforePass(Pass *pass, Operation *op) override {
    if (!shouldEditBefore || !shouldEditBefore(pass, op))
      return;
    std::string label = ("before " + pass->getArgument()).str();
    if (failed(water::editIRInteractively(op, editor, label)))
      signalPassFailure(pass);
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    if (!shouldEditAfter || !shouldEditAfter(pass, op))
      return;
    std::string label = ("after " + pass->getArgument()).str();
    if (failed(water::editIRInteractively(op, editor, label)))
      signalPassFailure(pass);
  }

private:
  water::PassFilter shouldEditBefore;
  water::PassFilter shouldEditAfter;
  std::string editor;
};

} // namespace

std::unique_ptr<PassInstrumentation> mlir::water::createEditIRInstrumentation(
    PassFilter shouldEditBefore, PassFilter shouldEditAfter, StringRef editor) {
  return std::make_unique<EditIRInstrumentation>(
      std::move(shouldEditBefore), std::move(shouldEditAfter), editor);
}
