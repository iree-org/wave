# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Integration tests that compile to binary and run on GPU
# These tests require ROCm runtime to be available

find_package(hip QUIET)

if(hip_FOUND)
  message(STATUS "HIP found, enabling GPU functional tests")

  # Add integration test executable
  add_executable(waveasm-functional-test
    functional_test.cpp
  )

  target_include_directories(waveasm-functional-test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
  )

  target_link_libraries(waveasm-functional-test
    hip::host
  )

  # Add CTest test
  add_test(NAME waveasm-functional-test
    COMMAND waveasm-functional-test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Mark test as requiring GPU (skip if no device available)
  set_tests_properties(waveasm-functional-test PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "gpu"
  )

  # Add functional test to check-waveasm dependencies
  if(TARGET check-waveasm)
    add_dependencies(check-waveasm waveasm-functional-test)
  endif()

else()
  message(STATUS "HIP not found, skipping GPU functional tests")
endif()
