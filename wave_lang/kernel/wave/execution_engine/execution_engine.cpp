// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "execution_engine.h"

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetMachine.h>

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Target/LLVMIR/Export.h>

#define DEBUG_TYPE "wave-execution-engine"

static void initializeLLVMTarget() {
  static bool initOnce = []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  }();
  (void)initOnce;
}

/// Wrap a string into an llvm::StringError.
static llvm::Error makeStringError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// A simple object cache following Lang's LLJITWithObjectCache example.
class wave::ExecutionEngine::SimpleObjectCache : public llvm::ObjectCache {
public:
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef objBuffer) override {
    cachedObjects[m->getModuleIdentifier()] =
        llvm::MemoryBuffer::getMemBufferCopy(objBuffer.getBuffer(),
                                             objBuffer.getBufferIdentifier());
  }

  std::unique_ptr<llvm::MemoryBuffer>
  getObject(const llvm::Module *m) override {
    auto i = cachedObjects.find(m->getModuleIdentifier());
    if (i == cachedObjects.end()) {
      LLVM_DEBUG(llvm::dbgs() << "No object for " << m->getModuleIdentifier()
                              << " in cache. Compiling.\n");
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "Object for " << m->getModuleIdentifier()
                            << " loaded from cache.\n");
    return llvm::MemoryBuffer::getMemBuffer(i->second->getMemBufferRef());
  }

  /// Dump cached object to output file `filename`.
  llvm::Error dumpToObjectFile(llvm::StringRef outputFilename) {
    // Set up the output file.
    std::string errorMessage;
    auto file = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      return makeStringError(errorMessage);
    }

    // Dump the object generated for a single module to the output file.
    assert(cachedObjects.size() == 1 && "Expected only one object entry.");
    auto &cachedObject = cachedObjects.begin()->second;
    file->os() << cachedObject->getBuffer();
    file->keep();
    return llvm::Error::success();
  }

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cachedObjects;
};

// Setup LLVM target triple from the current machine.
static void setupModule(llvm::Module &M, llvm::TargetMachine &TM) {
  M.setDataLayout(TM.createDataLayout());
  M.setTargetTriple(TM.getTargetTriple());
  for (auto &&func : M.functions()) {
    if (!func.hasFnAttribute("target-cpu"))
      func.addFnAttr("target-cpu", TM.getTargetCPU());

    if (!func.hasFnAttribute("target-features")) {
      auto featStr = TM.getTargetFeatureString();
      if (!featStr.empty())
        func.addFnAttr("target-features", featStr);
    }
  }
}

namespace {
class CustomCompiler : public llvm::orc::SimpleCompiler {
public:
  using Optimizer = std::function<llvm::Error(llvm::Module *)>;
  using Transformer = std::function<llvm::Error(llvm::Module &)>;
  using AsmPrinter = std::function<void(llvm::StringRef)>;

  CustomCompiler(Transformer t, AsmPrinter a,
                 std::unique_ptr<llvm::TargetMachine> TM,
                 llvm::ObjectCache *ObjCache = nullptr)
      : SimpleCompiler(*TM, ObjCache),
        optimizer(mlir::makeOptimizingTransformer(
            static_cast<unsigned>(TM->getOptLevel()),
            /*sizeLevel*/ 0, TM.get())),
        TM(std::move(TM)), transformer(std::move(t)), printer(std::move(a)) {}

  llvm::Expected<CompileResult> operator()(llvm::Module &M) override {
    if (transformer) {
      if (auto error = transformer(M))
        return error;
    }

    setupModule(M, *TM);
    if (auto error = optimizer(&M))
      return error;

    if (printer) {
      llvm::SmallVector<char, 0> buffer;
      llvm::raw_svector_ostream os(buffer);

      llvm::legacy::PassManager PM;
      if (TM->addPassesToEmitFile(PM, os, nullptr,
                                  llvm::CodeGenFileType::AssemblyFile))
        return makeStringError("Target does not support Asm emission");

      PM.run(M);
      printer(llvm::StringRef(buffer.data(), buffer.size()));
    }

    return llvm::orc::SimpleCompiler::operator()(M);
  }

private:
  Optimizer optimizer;
  std::shared_ptr<llvm::TargetMachine> TM;
  Transformer transformer;
  AsmPrinter printer;
};
} // namespace

wave::ExecutionEngine::ExecutionEngine(const ExecutionEngineOptions &options)
    : cache(options.enableObjectCache ? std::make_unique<SimpleObjectCache>()
                                      : nullptr),
      gdbListener(options.enableGDBNotificationListener
                      ? llvm::JITEventListener::createGDBRegistrationListener()
                      : nullptr),
      perfListener(nullptr) {
  if (options.enablePerfNotificationListener) {
    if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
      perfListener = listener;
    else if (auto *listener =
                 llvm::JITEventListener::createIntelJITEventListener())
      perfListener = listener;
  }

  // Initialize LLVM native target (only once per process)
  initializeLLVMTarget();

  auto tmBuilder =
      llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());

  // Get the target triple from the builder
  auto targetTriple = tmBuilder.getTargetTriple();

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator =
      [this, targetTriple](llvm::orc::ExecutionSession &session)
      -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
    auto GetMemMgr = [](const llvm::MemoryBuffer &) {
      return std::make_unique<llvm::SectionMemoryManager>();
    };

    auto objectLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        session, GetMemMgr);

    // Register JIT event listeners if they are enabled.
    if (gdbListener)
      objectLayer->registerJITEventListener(*gdbListener);
    if (perfListener)
      objectLayer->registerJITEventListener(*perfListener);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    // cf llvm/lib/ExecutionEngine/Orc/LLJIT.cpp LLJIT::createObjectLinkingLayer
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    return objectLayer;
  };

  // Callback to inspect the cache and recompile on demand. This follows Lang's
  // LLJITWithObjectCache example.
  auto compileFunctionCreator =
      [this, jitCodeGenOptLevel = options.jitCodeGenOptLevel,
       transformer = options.lateTransformer,
       asmPrinter = options.asmPrinter](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    if (jitCodeGenOptLevel)
      jtmb.setCodeGenOptLevel(*jitCodeGenOptLevel);
    auto tm = jtmb.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return std::make_unique<CustomCompiler>(transformer, asmPrinter,
                                            std::move(*tm), cache.get());
  };

  // Create the LLJIT by calling the LLJITBuilder with 2 callbacks.
  jit = llvm::cantFail(
      llvm::orc::LLJITBuilder()
          .setCompileFunctionCreator(compileFunctionCreator)
          .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
          .setJITTargetMachineBuilder(tmBuilder)
          .create());

  symbolMap = std::move(options.symbolMap);
  transformer = std::move(options.transformer);
}

wave::ExecutionEngine::~ExecutionEngine() {}

llvm::Expected<wave::ExecutionEngine::ModuleHandle>
wave::ExecutionEngine::loadModule(mlir::ModuleOp m) {
  assert(m);

  std::unique_ptr<llvm::LLVMContext> ctx(std::make_unique<llvm::LLVMContext>());
  auto llvmModule = mlir::translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    return makeStringError("could not convert to LLVM IR");

  // Add a ThreadSafeModule to the engine and return.
  llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
  if (transformer)
    cantFail(tsm.withModuleDo(
        [this](llvm::Module &module) { return transformer(module); }));

  llvm::orc::JITDylib *dylib;
  while (true) {
    auto uniqueName =
        (llvm::Twine("module") + llvm::Twine(uniqueNameCounter++)).str();
    if (jit->getJITDylibByName(uniqueName))
      continue;

    auto res = jit->createJITDylib(std::move(uniqueName));
    if (!res)
      return res.takeError();

    dylib = &res.get();
    break;
  }
  assert(dylib);

  auto dataLayout = jit->getDataLayout();
  dylib->addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          dataLayout.getGlobalPrefix())));

  if (symbolMap)
    cantFail(
        dylib->define(absoluteSymbols(symbolMap(llvm::orc::MangleAndInterner(
            dylib->getExecutionSession(), jit->getDataLayout())))));

  llvm::cantFail(jit->addIRModule(*dylib, std::move(tsm)));
  llvm::cantFail(jit->initialize(*dylib));
  return static_cast<ModuleHandle>(dylib);
}

mlir::DialectRegistry createMLIRContextRegistry();

llvm::Expected<wave::ExecutionEngine::ModuleHandle>
wave::ExecutionEngine::loadModuleFromBytecode(llvm::ArrayRef<char> bytecode) {
  // Create MLIR context on demand if not already created
  if (!mlirContext)
    mlirContext =
        std::make_unique<mlir::MLIRContext>(createMLIRContextRegistry());

  // Create memory buffer from bytecode
  auto memoryBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(bytecode.data(), bytecode.size()),
      /*BufferName=*/"bytecode",
      /*RequiresNullTerminator=*/false);

  // Deserialize MLIR module from bytecode
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(mlirContext.get())));

  mlir::ParserConfig config(mlirContext.get());
  if (mlir::failed(mlir::readBytecodeFile(memoryBuffer->getMemBufferRef(),
                                          module->getBody(), config)))
    return makeStringError("Failed to deserialize MLIR bytecode");

  // Load the deserialized module
  return loadModule(module.get());
}

llvm::Expected<wave::ExecutionEngine::ModuleHandle>
wave::ExecutionEngine::loadModuleFromText(llvm::StringRef mlirText) {
  // Create MLIR context on demand if not already created
  if (!mlirContext)
    mlirContext =
        std::make_unique<mlir::MLIRContext>(createMLIRContextRegistry());

  // Parse MLIR text
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlirText, mlirContext.get());

  if (!module)
    return makeStringError("Failed to parse MLIR text");

  // Load the parsed module
  return loadModule(module.get());
}

void wave::ExecutionEngine::releaseModule(ModuleHandle handle) {
  assert(handle);
  auto dylib = static_cast<llvm::orc::JITDylib *>(handle);
  llvm::cantFail(jit->deinitialize(*dylib));
  llvm::cantFail(jit->getExecutionSession().removeJITDylib(*dylib));
}

llvm::Expected<void *>
wave::ExecutionEngine::lookup(wave::ExecutionEngine::ModuleHandle handle,
                              llvm::StringRef name) const {
  assert(handle);
  auto dylib = static_cast<llvm::orc::JITDylib *>(handle);
  auto expectedSymbol = jit->lookup(*dylib, name);

  // JIT lookup may return an Error referring to strings stored internally by
  // the JIT. If the Error outlives the ExecutionEngine, it would have a
  // dangling reference, which is currently caught by an assertion inside JIT
  // thanks to hand-rolled reference counting. Rewrap the error message into a
  // string before returning. Alternatively, ORC JIT should consider copying
  // the string into the error message.
  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError(os.str());
  }

  if (void *fptr = expectedSymbol->toPtr<void *>())
    return fptr;

  return makeStringError("looked up function is null");
}

llvm::Error wave::ExecutionEngine::dumpToObjectFile(llvm::StringRef filename) {
  if (cache == nullptr)
    return makeStringError("cannot dump ExecutionEngine object code to file: "
                           "object cache is disabled");

  return cache->dumpToObjectFile(filename);
}
