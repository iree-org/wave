//===- WaterOptMain.cpp - MLIR Optimizer Driver ---------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that runs an optimization pass and prints the result back
// out. It is designed to support unit testing.
//
//===----------------------------------------------------------------------===//

#include "water/Tools/water-opt/WaterOptMain.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;
using namespace llvm;

/// Unrolls a call site containing other call site locations as callers (call
/// stack) into a flat list of callee locations.
static void unpackCallLoc(CallSiteLoc loc, SmallVectorImpl<Location> &result) {
  SmallVector<CallSiteLoc> stack;
  stack.push_back(loc);
  while (!stack.empty()) {
    CallSiteLoc call = stack.pop_back_val();
    result.push_back(call.getCallee());
    // We recurse on callers, not callsites, but this is not strictly forbidden.
    assert(!isa<CallSiteLoc>(result.back()) &&
           "unexpected callsite location recursion");
    if (auto caller = dyn_cast<CallSiteLoc>(call.getCaller())) {
      stack.push_back(caller);
      continue;
    }
    result.push_back(call.getCaller());
  }
}

/// Converts the given location into the JSON format recursively.
static llvm::json::Value locationToJSON(Location loc) {
  return llvm::TypeSwitch<Location, llvm::json::Value>(loc)
      .Case([](FileLineColRange fileLineCol) {
        fileLineCol.getStartLine();
        return llvm::json::Object{
            {"file", fileLineCol.getFilename().strref()},
            {"start_line", fileLineCol.getStartLine()},
            {"end_line", fileLineCol.getEndLine()},
            {"start_column", fileLineCol.getStartColumn()},
            {"end_column", fileLineCol.getEndColumn()}};
      })
      .Case([](UnknownLoc unknown) {
        return llvm::json::Object{{"unknown", true}};
      })
      .Case([](NameLoc name) {
        return llvm::json::Object{{"name", name.getName().strref()},
                                  {"loc", locationToJSON(name.getChildLoc())}};
      })
      .Case([](FusedLoc fused) {
        return llvm::json::Object{
            {"fused", llvm::json::Array(llvm::map_range(fused.getLocations(),
                                                        locationToJSON))}};
      })
      .Case([](CallSiteLoc call) {
        SmallVector<Location> stack;
        unpackCallLoc(call, stack);
        return llvm::json::Object{
            {"callstack",
             llvm::json::Array(llvm::map_range(stack, locationToJSON))}};
      })
      .Default([](Location loc) {
        std::string str;
        llvm::raw_string_ostream os(str);
        loc.print(os);
        return llvm::json::Object{{"unsupported", true},
                                  {"loc", llvm::json::Value(os.str())}};
      });
}

namespace {
class JSONDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  JSONDiagnosticHandler(MLIRContext *ctx, llvm::raw_ostream &os)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([&](Diagnostic &diag) {
      llvm::json::Value json = locationToJSON(diag.getLocation());
      llvm::json::Object *jsonObject = json.getAsObject();

      assert(jsonObject);

      StringRef severity = [&]() -> StringRef {
        switch (diag.getSeverity()) {
        case DiagnosticSeverity::Error:
          return "error";
        case DiagnosticSeverity::Warning:
          return "warning";
        case DiagnosticSeverity::Note:
          return "note";
        case DiagnosticSeverity::Remark:
          return "remark";
        }
        llvm_unreachable("unhandled diagnostic severity switch case");
      }();

      jsonObject->try_emplace("message", diag.str());
      jsonObject->try_emplace("severity", severity);
      os << json << "\n";
      return success();
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Taken from upstream LLVM because these are not exposed.
//===----------------------------------------------------------------------===//

namespace {
/// A scoped diagnostic handler that suppresses certain diagnostics based on
/// the verbosity level and whether the diagnostic is a note.
class DiagnosticFilter : public ScopedDiagnosticHandler {
public:
  DiagnosticFilter(MLIRContext *ctx, VerbosityLevel verbosityLevel,
                   bool showNotes = true)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([verbosityLevel, showNotes](Diagnostic &diag) {
      auto severity = diag.getSeverity();
      switch (severity) {
      case DiagnosticSeverity::Error:
        // failure indicates that the error is not handled by the filter and
        // goes through to the default handler. Therefore, the error can be
        // successfully printed.
        return failure();
      case DiagnosticSeverity::Warning:
        if (verbosityLevel == VerbosityLevel::ErrorsOnly)
          return success();
        else
          return failure();
      case DiagnosticSeverity::Remark:
        if (verbosityLevel == VerbosityLevel::ErrorsOnly ||
            verbosityLevel == VerbosityLevel::ErrorsAndWarnings)
          return success();
        else
          return failure();
      case DiagnosticSeverity::Note:
        if (showNotes)
          return failure();
        else
          return success();
      }
      llvm_unreachable("Unknown diagnostic severity");
    });
  }
};
} // namespace

static LogicalResult loadIRDLDialects(StringRef irdlFile, MLIRContext &ctx) {
  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  ctx.appendDialectRegistry(registry);

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<MemoryBuffer> file = openInputFile(irdlFile, &errorMessage);
  if (!file) {
    emitError(UnknownLoc::get(&ctx)) << errorMessage;
    return failure();
  }

  // Give the buffer to the source manager.
  // This will be picked up by the parser.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &ctx);

  // Parse the input file.
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &ctx));
  if (!module)
    return failure();

  // Load IRDL dialects.
  return irdl::loadDialects(module.get());
}

// Return success if the module can correctly round-trip. This intended to test
// that the custom printers/parsers are complete.
static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirOptMainConfig &config,
                                       bool useBytecode) {
  // We use a new context to avoid resource handle renaming issue in the diff.
  MLIRContext roundtripContext;
  OwningOpRef<Operation *> roundtripModule;
  roundtripContext.appendDialectRegistry(
      op->getContext()->getDialectRegistry());
  if (op->getContext()->allowsUnregisteredDialects())
    roundtripContext.allowUnregisteredDialects();
  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, roundtripContext)))
    return failure();

  std::string testType = (useBytecode) ? "bytecode" : "textual";
  // Print a first time with custom format (or bytecode) and parse it back to
  // the roundtripModule.
  {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (useBytecode) {
      if (failed(writeBytecodeToFile(op, ostream))) {
        op->emitOpError()
            << "failed to write bytecode, cannot verify round-trip.\n";
        return failure();
      }
    } else {
      op->print(ostream,
                OpPrintingFlags().printGenericOpForm().enableDebugInfo());
    }
    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parseConfig(&roundtripContext, config.shouldVerifyOnParsing(),
                             &fallbackResourceMap);
    roundtripModule = parseSourceString<Operation *>(buffer, parseConfig);
    if (!roundtripModule) {
      op->emitOpError() << "failed to parse " << testType
                        << " content back, cannot verify round-trip.\n";
      return failure();
    }
  }

  // Print in the generic form for the reference module and the round-tripped
  // one and compare the outputs.
  std::string reference, roundtrip;
  {
    llvm::raw_string_ostream ostreamref(reference);
    op->print(ostreamref,
              OpPrintingFlags().printGenericOpForm().enableDebugInfo());
    llvm::raw_string_ostream ostreamrndtrip(roundtrip);
    roundtripModule.get()->print(
        ostreamrndtrip,
        OpPrintingFlags().printGenericOpForm().enableDebugInfo());
  }
  if (reference != roundtrip) {
    // TODO implement a diff.
    return op->emitOpError()
           << testType
           << " roundTrip testing roundtripped module differs "
              "from reference:\n<<<<<<Reference\n"
           << reference << "\n=====\n"
           << roundtrip << "\n>>>>>roundtripped\n";
  }

  return success();
}

static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirOptMainConfig &config) {
  auto txtStatus = doVerifyRoundTrip(op, config, /*useBytecode=*/false);
  auto bcStatus = doVerifyRoundTrip(op, config, /*useBytecode=*/true);
  return success(succeeded(txtStatus) && succeeded(bcStatus));
}

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult
performActions(raw_ostream &os,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
               MLIRContext *context, const MlirOptMainConfig &config) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Prepare the parser config, and attach any useful/necessary resource
  // handlers. Unhandled external resources are treated as passthrough, i.e.
  // they are not processed and will be emitted directly to the output
  // untouched.
  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(context, config.shouldVerifyOnParsing(),
                           &fallbackResourceMap);
  if (config.shouldRunReproducer())
    reproOptions.attachResourceParser(parseConfig);

  // Parse the input file and reset the context threading state.
  TimingScope parserTiming = timing.nest("Parser");
  OwningOpRef<Operation *> op = parseSourceFileForTool(
      sourceMgr, parseConfig, !config.shouldUseExplicitModule());
  parserTiming.stop();
  if (!op)
    return failure();

  // Perform round-trip verification if requested
  if (config.shouldVerifyRoundtrip() &&
      failed(doVerifyRoundTrip(op.get(), config)))
    return failure();

  context->enableMultithreading(wasThreadingEnabled);

  // Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (config.shouldRunReproducer() && failed(reproOptions.apply(pm)))
    return failure();
  if (failed(config.setupPassPipeline(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();

  // Generate reproducers if requested
  if (!config.getReproducerFilename().empty()) {
    StringRef anchorName = pm.getAnyOpAnchorName();
    const auto &passes = pm.getPasses();
    makeReproducer(anchorName, passes, op.get(),
                   config.getReproducerFilename());
  }

  // Print the output.
  TimingScope outputTiming = timing.nest("Output");
  if (config.shouldEmitBytecode()) {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);
    if (auto v = config.bytecodeVersionToEmit())
      writerConfig.setDesiredBytecodeVersion(*v);
    if (config.shouldElideResourceDataFromBytecode())
      writerConfig.setElideResourceDataFlag();
    return writeBytecodeToFile(op.get(), os, writerConfig);
  }

  if (config.bytecodeVersionToEmit().has_value())
    return emitError(UnknownLoc::get(pm.getContext()))
           << "bytecode version while not emitting bytecode";
  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                    &fallbackResourceMap);
  op.get()->print(os, asmState);
  os << '\n';
  return success();
}

//===----------------------------------------------------------------------===//
// END taken from upstream.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ADAPTED from upstream LLVM.
//===----------------------------------------------------------------------===//

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult processBuffer(raw_ostream &os, raw_ostream *diagnosticsOs,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   const MlirOptMainConfig &config,
                                   DialectRegistry &registry,
                                   llvm::ThreadPoolInterface *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool)
    context.setThreadPool(*threadPool);

  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, context)))
    return failure();

  // Parse the input file.
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  if (config.shouldVerifyDiagnostics())
    context.printOpOnDiagnostic(false);

  tracing::InstallDebugHandler installDebugHandler(context,
                                                   config.getDebugConfig());

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!config.shouldVerifyDiagnostics()) {
    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
    DiagnosticFilter diagnosticFilter(&context,
                                      config.getDiagnosticVerbosityLevel(),
                                      config.shouldShowNotes());

    if (!diagnosticsOs)
      return performActions(os, sourceMgr, &context, config);

    JSONDiagnosticHandler jsonDiagnosticHandler(&context, *diagnosticsOs);

    return performActions(os, sourceMgr, &context, config);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(
      *sourceMgr, &context, config.verifyDiagnosticsLevel());

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, sourceMgr, &context, config);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

static LogicalResult printRegisteredDialects(DialectRegistry &registry) {
  llvm::outs() << "Available Dialects: ";
  interleave(registry.getDialectNames(), llvm::outs(), ",");
  llvm::outs() << "\n";
  return success();
}

static LogicalResult printRegisteredPassesAndReturn() {
  printRegisteredPasses();
  return success();
}

LogicalResult mlir::WaterOptMain(llvm::raw_ostream &outputStream,
                                 llvm::raw_ostream *diagnosticsStream,
                                 std::unique_ptr<llvm::MemoryBuffer> buffer,
                                 DialectRegistry &registry,
                                 const MlirOptMainConfig &config) {
  if (config.shouldShowDialects())
    return printRegisteredDialects(registry);

  if (config.shouldListPasses())
    return printRegisteredPassesAndReturn();

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPoolInterface *threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    return processBuffer(os, diagnosticsStream, std::move(chunkBuffer), config,
                         registry, threadPool);
  };
  return splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream,
                               config.inputSplitMarker(),
                               config.outputSplitMarker());
}

LogicalResult mlir::WaterOptMain(int argc, char **argv,
                                 llvm::StringRef inputFilename,
                                 llvm::StringRef outputFilename,
                                 llvm::StringRef diagnosticsFilename,
                                 DialectRegistry &registry) {

  InitLLVM y(argc, argv);

  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

  if (config.shouldShowDialects())
    return printRegisteredDialects(registry);

  if (config.shouldListPasses())
    return printRegisteredPassesAndReturn();

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  std::unique_ptr<ToolOutputFile> diagnostics = nullptr;
  if (!diagnosticsFilename.empty()) {
    diagnostics = openOutputFile(diagnosticsFilename, &errorMessage);
    if (!diagnostics) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
    diagnostics->keep();
  }

  if (failed(WaterOptMain(output->os(),
                          diagnostics ? &diagnostics->os() : nullptr,
                          std::move(file), registry, config)))
    return failure();

  // Keep the output file if the invocation of WaterOptMain was successful.
  output->keep();
  return success();
}

std::tuple<std::string, std::string, std::string>
mlir::registerAndParseWaterCLIOptions(int argc, char **argv,
                                      llvm::StringRef toolName,
                                      DialectRegistry &registry) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<std::string> diagnosticsFile(
      "diagnostics-file", cl::desc("Diagnostics filename"),
      cl::value_desc("filename"), cl::init(""));

  // Register any command line options.
  MlirOptMainConfig::registerCLOptions(registry);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  return std::make_tuple(inputFilename.getValue(), outputFilename.getValue(),
                         diagnosticsFile.getValue());
}

LogicalResult mlir::WaterOptMain(int argc, char **argv,
                                 llvm::StringRef toolName,
                                 DialectRegistry &registry) {

  // Register and parse command line options.
  auto [inputFilename, outputFilename, diagnosticsFilename] =
      registerAndParseWaterCLIOptions(argc, argv, toolName, registry);

  return WaterOptMain(argc, argv, inputFilename, outputFilename,
                      diagnosticsFilename, registry);
}

//===----------------------------------------------------------------------===//
// End adapted from upstream LLVM.
//===----------------------------------------------------------------------===//
