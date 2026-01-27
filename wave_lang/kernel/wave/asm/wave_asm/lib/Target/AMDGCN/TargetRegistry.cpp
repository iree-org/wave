// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

namespace waveasm {

// Forward declarations for target factory functions
std::unique_ptr<AMDGCNTarget> createGFX942Target();
std::unique_ptr<AMDGCNTarget> createGFX950Target();
std::unique_ptr<AMDGCNTarget> createGFX1250Target();

//===----------------------------------------------------------------------===//
// Target Registry Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<AMDGCNTarget> getAMDGCNTarget(llvm::StringRef targetId) {
  if (targetId == "gfx942")
    return createGFX942Target();
  if (targetId == "gfx950")
    return createGFX950Target();
  if (targetId == "gfx1250")
    return createGFX1250Target();

  // Unknown target
  return nullptr;
}

bool isTargetSupported(llvm::StringRef targetId) {
  return targetId == "gfx942" || targetId == "gfx950" || targetId == "gfx1250";
}

llvm::SmallVector<llvm::StringRef> getSupportedTargets() {
  return {"gfx942", "gfx950", "gfx1250"};
}

} // namespace waveasm
