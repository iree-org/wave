// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Pipelines.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/NormalForm/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

using namespace mlir;

void mlir::water::registerWaterPipelines() {
  PassPipelineRegistration<>(
      "water-middle-end-lowering",
      "Lower Wave dialect through normal forms to upstream MLIR dialects.",
      [](OpPassManager &pm) {
        pm.addPass(wave::createWaterWaveDetectNormalFormsPass());
        OpPassManager &nfPM = pm.nest<normalform::ModuleOp>();
        nfPM.addPass(wave::createWaterWavePropagateElementsPerThreadPass());
        nfPM.addPass(wave::createWaterWaveResolveDistributedAllocationsPass());
        nfPM.addPass(wave::createWaterWaveDetectNormalFormsPass());
        nfPM.addPass(wave::createLowerWaveToMLIRPass());
        pm.addPass(normalform::createLowerNormalFormModulePass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}
