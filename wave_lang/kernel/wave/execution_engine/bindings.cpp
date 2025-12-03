// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/Support/Error.h>

#include <mlir/CAPI/IR.h>

#include "execution_engine.h"

namespace nb = nanobind;

// Helper to convert llvm::Expected to Python (throw on error)
template <typename T>
static T unwrapExpected(llvm::Expected<T> expected, const char *context) {
  if (!expected) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::logAllUnhandledErrors(expected.takeError(), os);
    throw std::runtime_error(std::string(context) + ": " + os.str());
  }
  return std::move(*expected);
}

static void unwrapError(llvm::Error error, const char *context) {
  if (!error)
    return;

  std::string errorMessage;
  llvm::raw_string_ostream os(errorMessage);
  llvm::logAllUnhandledErrors(std::move(error), os);
  throw std::runtime_error(std::string(context) + ": " + os.str());
}

// Nanobind module definition for Python bindings
NB_MODULE(wave_execution_engine, m) {
  m.doc() = "LLVM ExecutionEngine bindings for Wave JIT compilation";

  // Bind ExecutionEngineOptions
  nb::class_<wave::ExecutionEngineOptions>(m, "ExecutionEngineOptions")
      .def(nb::init<>(), "Create default ExecutionEngineOptions")
      .def_rw("enable_object_cache",
              &wave::ExecutionEngineOptions::enableObjectCache,
              "Enable object cache for compiled code")
      .def_rw("enable_gdb_notification_listener",
              &wave::ExecutionEngineOptions::enableGDBNotificationListener,
              "Enable GDB notification listener")
      .def_rw("enable_perf_notification_listener",
              &wave::ExecutionEngineOptions::enablePerfNotificationListener,
              "Enable Perf notification listener")
      .def(
          "set_symbol_map",
          [](wave::ExecutionEngineOptions &self,
             const std::map<std::string, uintptr_t> &symbols) {
            // Convert Python dict to C++ symbolMap function
            self.symbolMap = [symbols](llvm::orc::MangleAndInterner mangle) {
              llvm::orc::SymbolMap symbolMap;
              for (const auto &[name, address] : symbols) {
                auto mangledName = mangle(name);
                auto flags = llvm::JITSymbolFlags::Exported |
                             llvm::JITSymbolFlags::Callable;
                symbolMap[mangledName] = llvm::orc::ExecutorSymbolDef(
                    llvm::orc::ExecutorAddr(address), flags);
              }
              return symbolMap;
            };
          },
          nb::arg("symbols"),
          R"(Set symbol map from a dictionary of symbol names to addresses.

Args:
    symbols: Dictionary mapping symbol names (str) to addresses (int)

Example:
    options.set_symbol_map({'my_function': 0x12345678}))");

  // Bind ExecutionEngine class
  nb::class_<wave::ExecutionEngine>(m, "ExecutionEngine",
                                    nb::is_weak_referenceable())
      .def(nb::init<const wave::ExecutionEngineOptions &>(), nb::arg("options"),
           R"(Create a new ExecutionEngine with the given options.

Args:
    options: ExecutionEngineOptions to configure the engine)")
      .def(
          "load_module",
          [](wave::ExecutionEngine &self, MlirModule cModule) {
            auto module = unwrap(cModule);
            auto handle = unwrapExpected(self.loadModule(module),
                                         "Failed to load module");
            return reinterpret_cast<uintptr_t>(handle);
          },
          nb::arg("module"),
          R"(Compile and load an MLIR module into the execution engine.

Args:
    module: MLIR module (MlirModule from MLIR C API)

Returns:
    Module handle as integer

Raises:
    RuntimeError: If compilation or loading fails)")
      .def(
          "load_module_from_bytecode",
          [](wave::ExecutionEngine &self, nb::bytes bytecode) {
            // Convert Python bytes to ArrayRef<char>
            llvm::ArrayRef<char> data(bytecode.c_str(), bytecode.size());
            auto handle = unwrapExpected(self.loadModuleFromBytecode(data),
                                         "Failed to load module from bytecode");
            return reinterpret_cast<uintptr_t>(handle);
          },
          nb::arg("bytecode"),
          R"(Deserialize MLIR bytecode and load it into the execution engine.

Args:
    bytecode: MLIR module serialized as bytecode (bytes)

Returns:
    Module handle as integer

Raises:
    RuntimeError: If deserialization, compilation or loading fails)")
      .def(
          "load_module_from_text",
          [](wave::ExecutionEngine &self, const std::string &mlirText) {
            auto handle = unwrapExpected(
                self.loadModuleFromText(llvm::StringRef(mlirText)),
                "Failed to load module from text");
            return reinterpret_cast<uintptr_t>(handle);
          },
          nb::arg("mlir_text"),
          R"(Parse MLIR text and load it into the execution engine.

Args:
    mlir_text: MLIR module as text string

Returns:
    Module handle as integer

Raises:
    RuntimeError: If parsing, compilation or loading fails)")
      .def(
          "release_module",
          [](wave::ExecutionEngine &self, uintptr_t handle) {
            self.releaseModule(reinterpret_cast<void *>(handle));
          },
          nb::arg("handle"),
          R"(Release a loaded module from the execution engine.

Args:
    handle: Module handle returned from load_module)")
      .def(
          "lookup",
          [](const wave::ExecutionEngine &self, uintptr_t handle,
             const std::string &name) {
            auto ptr = unwrapExpected(
                self.lookup(reinterpret_cast<void *>(handle), name),
                "Failed to lookup function");
            return reinterpret_cast<uintptr_t>(ptr);
          },
          nb::arg("handle"), nb::arg("name"),
          R"(Look up a function in a loaded module.

Args:
    handle: Module handle returned from load_module
    name: Name of the function to look up

Returns:
    Function address as integer

Raises:
    RuntimeError: If function lookup fails)")
      .def(
          "dump_to_object_file",
          [](wave::ExecutionEngine &self, const std::string &filename) {
            unwrapError(self.dumpToObjectFile(filename),
                        "Failed to dump object file");
          },
          nb::arg("filename"),
          R"(Dump compiled object code to a file.

Args:
    filename: Path to output file

Raises:
    RuntimeError: If dumping fails)");
}
