# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.19...3.27)
project(wave_execution_engine)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR in: ${MLIR_DIR}")
message(STATUS "Found nanobind in: ${nanobind_ROOT}")


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_library(wave_runtime_helpers SHARED
  buffer_utils.cpp
)

target_link_libraries(wave_runtime_helpers PRIVATE
  Python::Python
)

add_library(wave_hip_runtime SHARED
  wave_hip_runtime.cpp
)

# Add other runtime dir so we can include hip_types.h
target_include_directories(wave_hip_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../runtime")

nanobind_add_module(wave_execution_engine NB_STATIC
  bindings.cpp
  execution_engine.cpp
  register_dialects.cpp
)

# Disable RTTI for execution_engine.cpp to avoid typeinfo symbol dependencies
if(UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  set_source_files_properties(execution_engine.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# Include current directory for header files
target_include_directories(wave_execution_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(wave_execution_engine PRIVATE
  Python::Python
  Python::Module

  LLVM${LLVM_NATIVE_ARCH}AsmParser
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVM${LLVM_NATIVE_ARCH}Info
  LLVMExecutionEngine
  LLVMOrcJIT
  LLVMTarget
  MLIRExecutionEngineUtils
  MLIRLLVMDialect
  MLIRToLLVMIRTranslationRegistration
)

set_target_properties(wave_execution_engine PROPERTIES LINK_WHAT_YOU_USE TRUE)

# Force all symbols to be resolved at compile time (Linux/Unix only)
if(UNIX AND NOT APPLE)
  target_link_options(wave_execution_engine PRIVATE
    -Wl,--no-undefined
    -Wl,-z,defs
    # Use linker script to hide all symbols except Python module entry point
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/wave_execution_engine.lds
  )
# TODO: If we want to support OSX we will need a sligthly different incantation
# https://github.com/numba/numba-mlir/blob/fd9609d81dab96b0658392886ebc470ea2fa4eeb/numba_mlir/numba_mlir/mlir_compiler/CMakeLists.txt#L71
endif()

install(TARGETS wave_execution_engine wave_runtime_helpers wave_hip_runtime LIBRARY DESTINATION .)
