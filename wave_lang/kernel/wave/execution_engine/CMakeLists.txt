# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.19...3.27)
project(wave_execution_engine)

# Skip building on macOS
if(APPLE)
  message(STATUS "Skipping wave_execution_engine build on ${CMAKE_SYSTEM_NAME}")
  return()
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

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

# Build the core parts of nanobind once
nanobind_build_library(nanobind STATIC)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR in: ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Compile an extension library
add_library(wave_execution_engine MODULE
  execution_engine.cpp
  bindings.cpp
  buffer_utils.cpp
)

# Disable RTTI for execution_engine.cpp to avoid typeinfo symbol dependencies
# Only for GCC/Clang on Unix-like systems
if(UNIX AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  set_source_files_properties(execution_engine.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# Include current directory for header files
target_include_directories(wave_execution_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against nanobind
target_link_libraries(wave_execution_engine PRIVATE nanobind)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Link against LLVM and MLIR libraries
target_link_libraries(wave_execution_engine PRIVATE
  Python::Python
  Python::Module

  LLVM${LLVM_NATIVE_ARCH}AsmParser
  LLVM${LLVM_NATIVE_ARCH}CodeGen
  LLVM${LLVM_NATIVE_ARCH}Desc
  LLVMOrcJIT
  LLVMTarget
  MLIRCAPIDebug
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPITransforms
  MLIRPass
  MLIRRegisterAllDialects
  MLIRToLLVMIRTranslationRegistration
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${mlir_extension_libs}
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
endif()

# Enable size optimizations
nanobind_opt_size(wave_execution_engine)

# Enable link time optimization
nanobind_lto(wave_execution_engine)

# Set the default symbol visibility to 'hidden'
nanobind_set_visibility(wave_execution_engine)

# Strip unneeded symbols and debug info from the binary (only active in release builds)
nanobind_strip(wave_execution_engine)

# Disable the stack protector
nanobind_disable_stack_protector(wave_execution_engine)

# Set the Python extension suffix
nanobind_extension(wave_execution_engine)

# Set important linker flags
nanobind_link_options(wave_execution_engine)

install(TARGETS wave_execution_engine DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
