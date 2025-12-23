// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>

#include <functional>
#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace llvm {
template <typename T> class Expected;
class JITEventListener;

namespace orc {
class LLJIT;
class MangleAndInterner;
} // namespace orc
} // namespace llvm

namespace wave {
struct ExecutionEngineOptions {
  /// `jitCodeGenOptLevel`, when provided, is used as the optimization level for
  /// target code generation.
  std::optional<llvm::CodeGenOptLevel> jitCodeGenOptLevel = std::nullopt;

  /// If `enableObjectCache` is set, the JIT compiler will create one to store
  /// the object generated for the given module. The contents of the cache can
  /// be dumped to a file via the `dumpToObjectfile` method.
  bool enableObjectCache = false;

  /// If `enableGDBNotificationListener` is true, the JIT compiler will
  /// notify the llvm's global GDB notification listener.
  bool enableGDBNotificationListener = true;

  /// If `enablePerfNotificationListener` is true, the JIT compiler will notify
  /// the llvm's global Perf notification listener.
  bool enablePerfNotificationListener = true;

  /// Register symbols with this ExecutionEngine.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;

  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  std::function<llvm::Error(llvm::Module &)> transformer;

  /// If `lateTransformer` is provided, it will be called on the LLVM module
  /// just before final code generation and can be used, e.g., for reporting or
  /// optimization.
  std::function<llvm::Error(llvm::Module &)> lateTransformer;

  /// If `asmPrinter` is provided, it will be called to print resulting assembly
  /// just before final code generation.
  std::function<void(llvm::StringRef)> asmPrinter;
};

class ExecutionEngine {
  class SimpleObjectCache;

public:
  using ModuleHandle = void *;

  ExecutionEngine(const ExecutionEngineOptions &options);
  ~ExecutionEngine();

  /// Compiles given module, adds it to execution engine and run its
  /// constructors if any.
  llvm::Expected<ModuleHandle> loadModule(mlir::ModuleOp m);

  /// Deserializes MLIR bytecode from a memory buffer, compiles it, and loads
  /// it into the execution engine.
  llvm::Expected<ModuleHandle>
  loadModuleFromBytecode(llvm::ArrayRef<char> bytecode);

  /// Parses MLIR text from a string, compiles it, and loads it into the
  /// execution engine.
  llvm::Expected<ModuleHandle> loadModuleFromText(llvm::StringRef mlirText);

  /// Runs module destructors and removes it from execution engine.
  void releaseModule(ModuleHandle handle);

  /// Looks up the original function with the given name and returns a
  /// pointer to it. Propagates errors in case of failure.
  llvm::Expected<void *> lookup(ModuleHandle handle,
                                llvm::StringRef name) const;

  /// Dump object code to output file `filename`.
  llvm::Error dumpToObjectFile(llvm::StringRef filename);

private:
  /// MLIR context for deserializing bytecode.
  std::unique_ptr<mlir::MLIRContext> mlirContext;

  /// Ordering of llvmContext and jit is important for destruction purposes: the
  /// jit must be destroyed before the context.
  llvm::LLVMContext llvmContext;

  /// Underlying LLJIT.
  std::unique_ptr<llvm::orc::LLJIT> jit;

  /// Underlying cache.
  std::unique_ptr<SimpleObjectCache> cache;

  /// GDB notification listener.
  llvm::JITEventListener *gdbListener;

  /// Perf notification listener.
  llvm::JITEventListener *perfListener;

  /// Callback to get additional symbol definitions.
  std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)> symbolMap;

  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  std::function<llvm::Error(llvm::Module &)> transformer;

  /// Id for unique module name generation.
  int uniqueNameCounter = 0;
};
} // namespace wave
