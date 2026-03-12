# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(waveasm-translate)

# Install clang++ and ld.lld from LLVM for runtime assembler support.
# LLVM uses symlink chains (clang++ -> clang -> clang-NN), so we resolve
# to the real binary and install it directly under the expected name.
find_program(CLANG_EXECUTABLE clang++ HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLD_EXECUTABLE ld.lld HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

if(CLANG_EXECUTABLE)
  get_filename_component(CLANG_REAL ${CLANG_EXECUTABLE} REALPATH)
  install(PROGRAMS ${CLANG_REAL} DESTINATION bin RENAME clang++)
else()
  message(WARNING "clang++ not found in LLVM tools dir — assembler support will be unavailable")
endif()

if(LLD_EXECUTABLE)
  get_filename_component(LLD_REAL ${LLD_EXECUTABLE} REALPATH)
  install(PROGRAMS ${LLD_REAL} DESTINATION bin RENAME ld.lld)
else()
  message(WARNING "ld.lld not found in LLVM tools dir — linker support will be unavailable")
endif()
