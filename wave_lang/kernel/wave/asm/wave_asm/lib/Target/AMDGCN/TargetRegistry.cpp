// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "waveasm/Target/AMDGCN/AMDGCNTarget.h"

namespace waveasm {

//===----------------------------------------------------------------------===//
// Target Registry Implementation
//===----------------------------------------------------------------------===//

llvm::SmallVector<TargetKind> getSupportedTargets() {
  return llvm::to_vector(
      llvm::map_range(llvm::seq<uint32_t>(0, getMaxEnumValForTargetKind() + 1),
                      [](uint32_t i) { return static_cast<TargetKind>(i); }));
}

} // namespace waveasm
