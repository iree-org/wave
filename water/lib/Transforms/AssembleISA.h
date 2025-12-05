// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_LIB_TRANSFORMS_ASSEMBLEISA_H
#define WATER_LIB_TRANSFORMS_ASSEMBLEISA_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace mlir::water {

/// Initializes the LLVM AMDGPU target. Safe to call multiple times.
void initializeAMDGPUTarget();

/// Assembles ISA (assembly code) to HSACO (HSA Code Object) binary.
///
/// This function:
/// 1. Parses the ISA using LLVM MC infrastructure
/// 2. Assembles it to an ELF object file
/// 3. Links the object file using ld.lld to create an HSACO
///
/// \param op Operation for error reporting
/// \param isa Assembly code to assemble
/// \param targetMachine Target machine for MC infrastructure setup
/// \param toolkitPath Path to toolkit containing ld.lld
/// \return Binary data of the HSACO file, or failure
FailureOr<SmallVector<char, 0>>
assembleISAToHSACO(Operation *op, StringRef isa,
                   llvm::TargetMachine &targetMachine, StringRef toolkitPath);

} // namespace mlir::water

#endif // WATER_LIB_TRANSFORMS_ASSEMBLEISA_H
