# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)
project(waveasm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-------------------------------------------------------------------------------
# LLVM Version Pinning
#-------------------------------------------------------------------------------

# Pin to a specific LLVM commit for reproducible builds.
# This SHA corresponds to LLVM 19.x and is known to work with wave-asm.
set(WAVEASM_LLVM_SHA "c75d371f57608b01bfee12092e707c99124b5341" CACHE STRING
    "LLVM commit SHA to use for building wave-asm")

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#-------------------------------------------------------------------------------
# Test Configuration
#-------------------------------------------------------------------------------

set(WAVEASM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(WAVEASM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(WAVEASM_TOOLS_DIR ${CMAKE_CURRENT_BINARY_DIR}/tools/waveasm-translate)

#-------------------------------------------------------------------------------
# Directory Configuration
#-------------------------------------------------------------------------------

# Enable CTest for integration tests
enable_testing()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
