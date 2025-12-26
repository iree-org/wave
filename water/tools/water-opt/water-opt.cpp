//===- water-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Tools/water-opt/WaterOptMain.h"
#include "water/Transforms/Passes.h"

using namespace mlir;

// Forward-declare test passes so we don't have a dependency on the test
// headers.
namespace mlir::water::test {
void registerAllPasses();
void registerWaterTestDialect(DialectRegistry &registry);
} // namespace mlir::water::test

int main(int argc, char **argv) {
  arith::registerArithIntRangeOptsPass();
  registerCSEPass();
  registerCanonicalizerPass();
  registerCompositeFixedPointPass();
  registerConvertAMDGPUToROCDLPass();
  registerConvertGpuOpsToROCDLOpsPass();
  registerConvertVectorToLLVMPass();
  registerGpuModuleToBinaryPass();
  registerGpuROCDLAttachTarget();
  registerGpuToLLVMConversionPass();
  registerLoopInvariantCodeMotionPass();
  registerLowerAffinePass();
  registerReconcileUnrealizedCastsPass();
  registerSCFToControlFlowPass();
  registerSymbolDCEPass();
  transform::registerTransformPasses();
  water::registerPasses();
  water::test::registerAllPasses();
  wave::registerPasses();

  DialectRegistry registry;
  registry.insert<
      // clang-format off
      LLVM::LLVMDialect,
      ROCDL::ROCDLDialect,
      affine::AffineDialect,
      amdgpu::AMDGPUDialect,
      arith::ArithDialect,
      cf::ControlFlowDialect,
      func::FuncDialect,
      gpu::GPUDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      transform::TransformDialect,
      vector::VectorDialect,
      wave::WaveDialect
      // clang-format on
      >();

  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);

  arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);

  ROCDL::registerROCDLTargetInterfaceExternalModels(registry);

  registerGPUDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);

  water::test::registerWaterTestDialect(registry);

  return asMainReturnCode(
      WaterOptMain(argc, argv, "water optimizer driver\n", registry));
}
