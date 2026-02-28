# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Include handler subdirectory to get HANDLERS_SRCS
add_subdirectory(handlers)

# Prepend handlers/ path to each handler source file
set(HANDLERS_FULL_PATHS)
foreach(src ${HANDLERS_SRCS})
  list(APPEND HANDLERS_FULL_PATHS handlers/${src})
endforeach()

add_mlir_dialect_library(MLIRWaveASMTransforms
  Liveness.cpp
  LinearScanRegAlloc.cpp
  LinearScanPass.cpp
  Ticketing.cpp
  HazardMitigation.cpp
  AssemblyEmitter.cpp
  MetadataEmitter.cpp
  LiteralMaterialization.cpp
  TranslateFromMLIR.cpp
  RegionBuilder.cpp
  ScopedCSE.cpp
  Peephole.cpp
  M0RedundancyElimination.cpp
  MemoryOffsetOptimization.cpp
  ScalePackElimination.cpp
  BufferLoadStrengthReduction.cpp
  LoopAddressPromotion.cpp
  ${HANDLERS_FULL_PATHS}

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
  MLIRMathDialect
  MLIRIR
  MLIRSupport
)
