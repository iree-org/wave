// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// waveasm-conductor: CLI tool for applying Conductor move commands.
//
// Reads tagged WaveASM IR with embedded // CONDUCTOR: comments,
// parses and applies the move/swap commands, then prints the result.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Transforms/ApplyMoves.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> printDebugLocsInline(
    "print-debug-locs-inline",
    llvm::cl::desc("Print location information inline (pretty form)"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Main Function
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "WAVEASM Conductor move executor\n");

  // Read the raw input file to extract CONDUCTOR commands before parsing.
  auto inputFileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = inputFileOrErr.getError()) {
    llvm::errs() << "Error reading input file: " << ec.message() << "\n";
    return 1;
  }

  llvm::StringRef rawText = (*inputFileOrErr)->getBuffer();
  auto parseResult = waveasm::parseConductorCommands(rawText);

  if (!parseResult.success) {
    llvm::errs() << "conductor: parse error at command "
                 << parseResult.failedLine << ": " << parseResult.error << "\n";
    return 1;
  }
  if (parseResult.commands.empty()) {
    llvm::errs() << "No CONDUCTOR commands found in input\n";
    return 1;
  }

  // Set up MLIR context and parse the module.
  DialectRegistry registry;
  registry.insert<waveasm::WaveASMDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*inputFileOrErr), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Run tag-instructions pass to attach NameLoc tags.
  PassManager pm(&context);
  pm.addPass(waveasm::createWAVEASMTagInstructionsPass());
  if (failed(pm.run(*module))) {
    llvm::errs() << "Tag-instructions pass failed\n";
    return 1;
  }

  // Apply the move commands.
  waveasm::MoveResult result =
      waveasm::applyMoves(*module, parseResult.commands);
  if (!result.success) {
    llvm::errs() << "conductor: command " << result.failedCommand << ": "
                 << result.error << "\n";
    return 1;
  }

  // Verify the module after moves (catches broken dominance, etc.).
  if (failed(mlir::verify(*module))) {
    llvm::errs() << "conductor: verification failed after applying moves\n";
    return 1;
  }

  // Print the result.
  std::error_code ec;
  llvm::raw_fd_ostream outputStream(outputFilename, ec);
  if (ec) {
    llvm::errs() << "Error opening output file: " << ec.message() << "\n";
    return 1;
  }

  OpPrintingFlags flags;
  if (printDebugLocsInline) {
    flags.enableDebugInfo(/*prettyForm=*/true);
    flags.useLocalScope();
  }
  module->print(outputStream, flags);

  return 0;
}
