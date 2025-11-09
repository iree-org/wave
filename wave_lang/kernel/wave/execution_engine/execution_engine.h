// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVE_EXECUTION_ENGINE_H
#define WAVE_EXECUTION_ENGINE_H

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations for LLVM types (to be implemented)
// These will be replaced with actual LLVM includes when implementing
namespace llvm {
  class Module;
  class ExecutionEngine;
  class LLVMContext;
}

namespace mlir {
  class MLIRContext;
  class ModuleOp;
}

namespace wave {

/// LLVM ExecutionEngine wrapper for Wave JIT compilation
class WaveExecutionEngine {
public:
  WaveExecutionEngine();
  ~WaveExecutionEngine();

  // Disable copy and move operations
  WaveExecutionEngine(const WaveExecutionEngine&) = delete;
  WaveExecutionEngine& operator=(const WaveExecutionEngine&) = delete;
  WaveExecutionEngine(WaveExecutionEngine&&) = delete;
  WaveExecutionEngine& operator=(WaveExecutionEngine&&) = delete;

  /// Initialize the execution engine with MLIR module
  /// @param mlir_module_str MLIR module as a string
  /// @throws std::runtime_error if initialization fails or already initialized
  void initialize(const std::string& mlir_module_str);

  /// Load a pre-compiled LLVM IR module
  /// @param ir_str LLVM IR as a string
  /// @throws std::runtime_error if loading fails
  void load_llvm_ir(const std::string& ir_str);

  /// Lookup and invoke a function by name
  /// @param func_name Name of the function to invoke
  /// @param args Vector of arguments as uint64_t values
  /// @throws std::runtime_error if engine not initialized or function not found
  void invoke(const std::string& func_name, const std::vector<uint64_t>& args);

  /// Get pointer to a function by name
  /// @param func_name Name of the function
  /// @return Address of the function
  /// @throws std::runtime_error if engine not initialized or function not found
  uintptr_t get_function_address(const std::string& func_name);

  /// Check if engine is initialized
  /// @return true if initialized, false otherwise
  bool is_initialized() const;

  /// Optimize the module with given optimization level
  /// @param opt_level Optimization level (0-3, default 2)
  /// @throws std::runtime_error if engine not initialized
  void optimize(int opt_level = 2);

  /// Dump the current LLVM IR module for debugging
  /// @return LLVM IR as a string
  /// @throws std::runtime_error if engine not initialized
  std::string dump_llvm_ir() const;

private:
  void cleanup();

  bool initialized_;

  // TODO: Add private members for:
  // std::unique_ptr<llvm::LLVMContext> llvm_context_;
  // std::unique_ptr<llvm::Module> llvm_module_;
  // std::unique_ptr<llvm::ExecutionEngine> execution_engine_;
  // std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

/// Initialize LLVM and MLIR infrastructure
/// Must be called before creating any WaveExecutionEngine instances
/// @throws std::runtime_error if initialization fails
void initialize_llvm_mlir();

} // namespace wave

#endif // WAVE_EXECUTION_ENGINE_H
