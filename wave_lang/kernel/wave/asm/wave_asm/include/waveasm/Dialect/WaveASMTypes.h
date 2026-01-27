// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMTYPES_H
#define WaveASM_DIALECT_WAVEASMTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"

// Include the generated enum declarations
#include "waveasm/Dialect/WaveASMEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "waveasm/Dialect/WaveASMTypes.h.inc"

namespace waveasm {

// Helper functions for register type classification

/// Check if a type is a virtual register type
inline bool isVirtualRegType(mlir::Type type) {
  return mlir::isa<VRegType, SRegType>(type);
}

/// Check if a type is a physical register type
inline bool isPhysicalRegType(mlir::Type type) {
  return mlir::isa<PVRegType, PSRegType>(type);
}

/// Check if a type is any register type
inline bool isRegType(mlir::Type type) {
  return mlir::isa<VRegType, SRegType, PVRegType, PSRegType>(type);
}

/// Check if a type is a VGPR type (virtual or physical)
inline bool isVGPRType(mlir::Type type) {
  return mlir::isa<VRegType, PVRegType>(type);
}

/// Check if a type is an SGPR type (virtual or physical)
inline bool isSGPRType(mlir::Type type) {
  return mlir::isa<SRegType, PSRegType>(type);
}

/// Get the register class for a type
inline std::optional<RegClass> getRegClass(mlir::Type type) {
  if (mlir::isa<VRegType, PVRegType>(type))
    return RegClass::VGPR;
  if (mlir::isa<SRegType, PSRegType>(type))
    return RegClass::SGPR;
  return std::nullopt;
}

/// Get the register size for a type
inline int64_t getRegSize(mlir::Type type) {
  if (auto vreg = mlir::dyn_cast<VRegType>(type))
    return vreg.getSize();
  if (auto sreg = mlir::dyn_cast<SRegType>(type))
    return sreg.getSize();
  if (auto pvreg = mlir::dyn_cast<PVRegType>(type))
    return pvreg.getSize();
  if (auto psreg = mlir::dyn_cast<PSRegType>(type))
    return psreg.getSize();
  return 1;
}

/// Get register alignment for a type
inline int64_t getRegAlignment(mlir::Type type) {
  if (auto vreg = mlir::dyn_cast<VRegType>(type))
    return vreg.getAlignment();
  if (auto sreg = mlir::dyn_cast<SRegType>(type))
    return sreg.getAlignment();
  // Physical registers don't have alignment (already allocated)
  return 1;
}

/// Check if type is an immediate
inline bool isImmType(mlir::Type type) {
  return mlir::isa<ImmType>(type);
}

} // namespace waveasm

#endif // WaveASM_DIALECT_WAVEASMTYPES_H
