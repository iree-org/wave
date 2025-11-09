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

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)

# TODO: Add LLVM and MLIR dependencies when implementing
# find_package(LLVM REQUIRED CONFIG)
# find_package(MLIR REQUIRED CONFIG)
#
# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
# message(STATUS "Found MLIR in: ${MLIR_DIR}")
#
# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})
#
# add_definitions(${LLVM_DEFINITIONS})

# Compile an extension library
add_library(wave_execution_engine MODULE
  execution_engine.cpp
  bindings.cpp
)

# Include current directory for header files
target_include_directories(wave_execution_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against nanobind
target_link_libraries(wave_execution_engine PRIVATE nanobind)

# TODO: Link against LLVM and MLIR libraries when implementing
# target_link_libraries(wave_execution_engine PRIVATE
#   MLIRIR
#   MLIRParser
#   MLIRExecutionEngine
#   MLIRTargetLLVMIRExport
#   MLIRLLVMDialect
#   LLVMCore
#   LLVMSupport
#   LLVMExecutionEngine
#   LLVMJIT
#   LLVMOrcJIT
#   LLVMRuntimeDyld
# )

set_target_properties(wave_execution_engine PROPERTIES LINK_WHAT_YOU_USE TRUE)

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
