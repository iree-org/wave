// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_UNIFORMITYANALYSIS_H
#define WATER_DIALECT_WAVE_TRANSFORMS_UNIFORMITYANALYSIS_H

namespace llvm {
struct LogicalResult;
}

namespace mlir {
class DataFlowSolver;
class Value;
} // namespace mlir

namespace wave {

/// Add uniformity analysis to the solver.
void addWaveUniformityAnalysis(mlir::DataFlowSolver &solver);

/// Check if a value is uniform across all threads in a wavefront.
bool isUniform(mlir::Value value, const mlir::DataFlowSolver &solver);

} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_UNIFORMITYANALYSIS_H
