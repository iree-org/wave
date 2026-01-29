// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H
#define WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H

#include "waveasm/Dialect/WaveASMAttrs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Target Registry
//===----------------------------------------------------------------------===//

/// Check if a target ID is supported
static inline bool isTargetSupported(llvm::StringRef targetId) {
  return symbolizeTargetKind(targetId) != std::nullopt;
}

/// Get list of all supported target IDs
llvm::SmallVector<TargetKind> getSupportedTargets();

} // namespace waveasm

#endif // WaveASM_TARGET_AMDGCN_AMDGCNTARGET_H
