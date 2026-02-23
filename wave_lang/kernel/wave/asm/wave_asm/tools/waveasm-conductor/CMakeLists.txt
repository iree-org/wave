# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(waveasm-conductor
  waveasm-conductor.cpp
)

llvm_update_compile_flags(waveasm-conductor)

target_link_libraries(waveasm-conductor
  PRIVATE
  MLIRWaveASMDialect
  MLIRWaveASMTransforms
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRIR
  LLVMSupport
)
