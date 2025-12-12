// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#define GEN_PASS_DEF_LOWERWAVETOMLIRPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

struct LowerWaveToMLIRPass
    : public ::impl::LowerWaveToMLIRPassBase<LowerWaveToMLIRPass> {
  using LowerWaveToMLIRPassBase::LowerWaveToMLIRPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        // clang-format off
      affine::AffineDialect,
      arith::ArithDialect,
      math::MathDialect,
      gpu::GPUDialect,
      amdgpu::AMDGPUDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      vector::VectorDialect
        // clang-format on
        >();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // TODO: require index expressions to be present
    if (failed(wave::verifyNormalFormPassPrecondition(
            wave::WaveNormalForm::AllTypesSpecified |
                wave::WaveNormalForm::MemoryOnlyTypes,
            op, getPassName())))
      return signalPassFailure();

    ConversionTarget target(*ctx);
    target.addLegalDialect<
        // clang-format off
      affine::AffineDialect,
      arith::ArithDialect,
      func::FuncDialect,
      math::MathDialect,
      gpu::GPUDialect,
      amdgpu::AMDGPUDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      vector::VectorDialect
        // clang-format on
        >();
    target.addIllegalOp<wave::AddOp, wave::AllocateOp, wave::CastOp,
                        wave::DivOp, wave::Exp2Op, wave::ExtractSliceOp,
                        wave::IterateOp, wave::MmaOp, wave::MulOp, wave::ReadOp,
                        wave::RegisterOp, wave::WriteOp, wave::YieldOp>();

    // Mark functions as illegal if they have Wave tensor types in their
    // signature.
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return llvm::none_of(op.getFunctionType().getInputs(),
                           llvm::IsaPred<wave::WaveTensorType>) &&
             llvm::none_of(op.getFunctionType().getResults(),
                           llvm::IsaPred<wave::WaveTensorType>);
    });
    ConversionConfig config;
    config.allowPatternRollback = false;

    auto *waveDialect = getContext().getLoadedDialect<wave::WaveDialect>();
    WalkResult walkResult =
        getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
          // We shouldn't hit standalone Wave dialect operations as we are
          // walking in preorder.
          // TODO: consider turning this into a normalform.
          if (op->getDialect() == waveDialect) {
            op->emitError() << "wave dialect operation with no hyperparameters "
                               "provided by any ancestor";
            return WalkResult::interrupt();
          }

          auto hyperparam = op->getAttrOfType<wave::WaveHyperparameterAttr>(
              wave::WaveDialect::kHyperparameterAttrName);
          if (!hyperparam)
            return WalkResult::advance();

          wave::WaveTypeConverter typeConverter(hyperparam);
          RewritePatternSet patterns(ctx);

          // Add function signature conversion patterns.
          mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
              patterns, typeConverter);

          wave::populateWaveMiscellaneousOpsLoweringPatterns(typeConverter,
                                                             patterns);
          wave::populateWaveBinaryOpLoweringPatterns(typeConverter, patterns);
          wave::populateWaveAllocateOpLoweringPatterns(typeConverter, patterns);
          wave::populateWaveMmaLoweringPatterns(typeConverter, patterns);
          wave::populateWaveReadWriteLoweringPatterns(typeConverter, patterns);
          wave::populateWaveUnaryFPOpLoweringPatterns(typeConverter, patterns);

          if (failed(applyPartialConversion(op, target, std::move(patterns),
                                            config))) {
            op->emitError() << "failed to convert starting at this operation";
            return WalkResult::interrupt();
          }

          op->removeAttr(wave::WaveDialect::kHyperparameterAttrName);
          op->removeAttr(wave::WaveDialect::kWaveConstraintsAttrName);
          return WalkResult::skip();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    if (failed(wave::clearNormalFormPassPostcondition(op)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> wave::createLowerWaveToMLIRPass() {
  return std::make_unique<LowerWaveToMLIRPass>();
}
