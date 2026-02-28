# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Configure lit tests
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WAVEASM_TEST_DEPENDS
  waveasm-translate
)

add_lit_testsuite(check-waveasm "Running the waveasm regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${WAVEASM_TEST_DEPENDS}
)

set_target_properties(check-waveasm PROPERTIES FOLDER "WAVEASM Tests")

# Add integration tests subdirectory (HIP-based GPU execution tests)
add_subdirectory(Integration)

# Create a unified target that runs all tests (lit + CTest)
add_custom_target(check-waveasm-all
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-waveasm
  COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure -L gpu
  COMMENT "Running all WAVEASM tests (lit + GPU functional)"
  DEPENDS waveasm-translate
)

set_target_properties(check-waveasm-all PROPERTIES FOLDER "WAVEASM Tests")
