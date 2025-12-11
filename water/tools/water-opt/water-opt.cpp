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

// Forward-declare test passes so we don't have a dependency on the test
// headers.
namespace mlir::water::test {
void registerAllPasses();
void registerWaterTestDialect(DialectRegistry &registry);
} // namespace mlir::water::test

int main(int argc, char **argv) {
  mlir::arith::registerArithIntRangeOptsPass();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCompositeFixedPointPass();
  mlir::registerConvertAMDGPUToROCDLPass();
  mlir::registerConvertGpuOpsToROCDLOpsPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerGpuModuleToBinaryPass();
  mlir::registerGpuROCDLAttachTarget();
  mlir::registerGpuToLLVMConversionPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLowerAffinePass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerSymbolDCEPass();
  mlir::transform::registerTransformPasses();
  mlir::water::registerPasses();
  mlir::water::test::registerAllPasses();
  wave::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<
      // clang-format off
      mlir::LLVM::LLVMDialect,
      mlir::ROCDL::ROCDLDialect,
      mlir::affine::AffineDialect,
      mlir::amdgpu::AMDGPUDialect,
      mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect,
      mlir::gpu::GPUDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::transform::TransformDialect,
      mlir::vector::VectorDialect,
      wave::WaveDialect
      // clang-format on
      >();

  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::memref::registerTransformDialectExtension(registry);

  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);

  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry);

  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);

  mlir::water::test::registerWaterTestDialect(registry);

  return mlir::asMainReturnCode(
      WaterOptMain(argc, argv, "water optimizer driver\n", registry));
}
