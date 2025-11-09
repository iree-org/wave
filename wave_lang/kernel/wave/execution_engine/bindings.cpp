// Copyright 2025 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "execution_engine.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Nanobind module definition for Python bindings
NB_MODULE(wave_execution_engine, m) {
  m.doc() = "LLVM ExecutionEngine bindings for Wave JIT compilation";

  // Bind the WaveExecutionEngine class
  nb::class_<wave::WaveExecutionEngine>(m, "ExecutionEngine")
      .def(nb::init<>(),
           "Create a new WaveExecutionEngine instance")
      .def("initialize", &wave::WaveExecutionEngine::initialize,
           nb::arg("mlir_module_str"),
           "Initialize the execution engine with an MLIR module string.\n\n"
           "Args:\n"
           "    mlir_module_str: MLIR module as a string\n\n"
           "Raises:\n"
           "    RuntimeError: If initialization fails or already initialized")
      .def("load_llvm_ir", &wave::WaveExecutionEngine::load_llvm_ir,
           nb::arg("ir_str"),
           "Load a pre-compiled LLVM IR module.\n\n"
           "Args:\n"
           "    ir_str: LLVM IR as a string\n\n"
           "Raises:\n"
           "    RuntimeError: If loading fails")
      .def("invoke", &wave::WaveExecutionEngine::invoke,
           nb::arg("func_name"), nb::arg("args"),
           "Invoke a function by name with the given arguments.\n\n"
           "Args:\n"
           "    func_name: Name of the function to invoke\n"
           "    args: List of arguments as uint64_t values\n\n"
           "Raises:\n"
           "    RuntimeError: If engine not initialized or function not found")
      .def("get_function_address", &wave::WaveExecutionEngine::get_function_address,
           nb::arg("func_name"),
           "Get the address of a function by name.\n\n"
           "Args:\n"
           "    func_name: Name of the function\n\n"
           "Returns:\n"
           "    Address of the function as an integer\n\n"
           "Raises:\n"
           "    RuntimeError: If engine not initialized or function not found")
      .def("is_initialized", &wave::WaveExecutionEngine::is_initialized,
           "Check if the execution engine is initialized.\n\n"
           "Returns:\n"
           "    True if initialized, False otherwise")
      .def("optimize", &wave::WaveExecutionEngine::optimize,
           nb::arg("opt_level") = 2,
           "Optimize the module with the given optimization level.\n\n"
           "Args:\n"
           "    opt_level: Optimization level (0-3, default 2)\n"
           "               0 = No optimization\n"
           "               1 = Basic optimizations\n"
           "               2 = Standard optimizations (default)\n"
           "               3 = Aggressive optimizations\n\n"
           "Raises:\n"
           "    RuntimeError: If engine not initialized or invalid opt_level")
      .def("dump_llvm_ir", &wave::WaveExecutionEngine::dump_llvm_ir,
           "Dump the current LLVM IR module as a string.\n\n"
           "Returns:\n"
           "    LLVM IR as a string\n\n"
           "Raises:\n"
           "    RuntimeError: If engine not initialized");

  // Bind the global initialization function
  m.def("initialize_llvm_mlir", &wave::initialize_llvm_mlir,
        "Initialize LLVM and MLIR infrastructure.\n\n"
        "Must be called before creating any WaveExecutionEngine instances.\n\n"
        "Raises:\n"
        "    RuntimeError: If initialization fails");
}
