# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(waveasm-translate)

# Install clang++ and ld.lld from LLVM for runtime assembler support.
find_program(CLANG_PLUS_PLUS clang++ HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLD_EXECUTABLE ld.lld HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

if(CLANG_PLUS_PLUS)
  install(PROGRAMS ${CLANG_PLUS_PLUS} DESTINATION bin)
else()
  message(WARNING "clang++ not found in LLVM tools dir — assembler support will be unavailable")
endif()

if(LLD_EXECUTABLE)
  install(PROGRAMS ${LLD_EXECUTABLE} DESTINATION bin)
else()
  message(WARNING "ld.lld not found in LLVM tools dir — linker support will be unavailable")
endif()
