//===- WaterOptMain.h - MLIR Optimizer Driver main -------------*- C++ --*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef WATER_TOOLS_WATEROPT_WATEROPTMAIN_H
#define WATER_TOOLS_WATEROPT_WATEROPTMAIN_H

#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace mlir {
class DialectRegistry;

/// Register and parse command line options.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
/// - return std::pair<std::string, std::string> for
///   inputFilename and outputFilename command line option values.
std::tuple<std::string, std::string, std::string>
registerAndParseWaterCLIOptions(int argc, char **argv, llvm::StringRef toolName,
                                DialectRegistry &registry);

/// Perform the core processing behind `mlir-opt`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult WaterOptMain(llvm::raw_ostream &outputStream,
                           llvm::raw_ostream *diagnosticsStream,
                           std::unique_ptr<llvm::MemoryBuffer> buffer,
                           DialectRegistry &registry,
                           const MlirOptMainConfig &config);

/// Implementation for tools like `mlir-opt`.
/// This function can be used with registerAndParseCLIOptions so that
/// CLI options can be accessed before running WaterOptMain.
/// - inputFilename is the name of the input mlir file.
/// - outputFilename is the name of the output file.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult WaterOptMain(int argc, char **argv, llvm::StringRef inputFilename,
                           llvm::StringRef outputFilename,
                           llvm::StringRef diagnosticsFilename,
                           DialectRegistry &registry);

/// Implementation for tools like `mlir-opt`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult WaterOptMain(int argc, char **argv, llvm::StringRef toolName,
                           DialectRegistry &registry);

} // namespace mlir

#endif // WATER_TOOLS_WATEROPT_WATEROPTMAIN_H
