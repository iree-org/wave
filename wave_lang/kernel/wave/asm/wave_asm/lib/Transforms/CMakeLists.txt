# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_mlir_dialect_library(MLIRWaveASMTransforms
  Liveness.cpp
  LinearScanRegAlloc.cpp
  LinearScanPass.cpp
  Ticketing.cpp
  HazardMitigation.cpp
  AssemblyEmitter.cpp
  TranslateFromMLIR.cpp
  ScopedCSE.cpp
  Peephole.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/waveasm

  DEPENDS
  MLIRWaveASMPassesIncGen

  LINK_LIBS PUBLIC
  MLIRWaveASMDialect
  MLIRWaveASMTarget
  MLIRAffineDialect
  MLIRAMDGPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRIR
  MLIRSupport
)
