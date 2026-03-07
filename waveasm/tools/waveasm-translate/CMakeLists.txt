# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_llvm_executable(waveasm-translate
  waveasm-translate.cpp
)

llvm_update_compile_flags(waveasm-translate)

target_link_libraries(waveasm-translate
  PRIVATE
  LLVMCore
  LLVMSupport
  MLIRAffineDialect
  MLIRAMDGPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  MLIRVectorDialect
  MLIRWaveASMDialect
  MLIRWaveASMTarget
  MLIRWaveASMTransforms
)

install(TARGETS waveasm-translate
  RUNTIME DESTINATION bin
)
