// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::water {
#define GEN_PASS_DEF_WATERMEMREFLOWERINGPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

namespace {
struct ConvertUnrealizedConversionCast
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getInputs();
    if (inputs.size() != castOp.getNumResults())
      return rewriter.notifyMatchFailure(
          castOp, "expected number of operands to match number of results");

    for (auto [input, result] : llvm::zip(inputs, castOp.getResults())) {
      Type resultType = typeConverter->convertType(result.getType());
      if (resultType != input.getType())
        return rewriter.notifyMatchFailure(castOp,
                                           "failed to convert result type");
    }

    rewriter.replaceOp(castOp, inputs);
    return success();
  }
};

struct ConvertMemrefCast : public OpConversionPattern<memref::CastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value source = adaptor.getSource();
    Type resultType = typeConverter->convertType(castOp.getResult().getType());
    if (resultType != source.getType())
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert result type");

    rewriter.replaceOp(castOp, source);
    return success();
  }
};

struct ConvertMemrefLoad : public OpConversionPattern<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = cast<MemRefType>(loadOp.getMemRefType());

    // Only handle 0D memrefs.
    if (memrefType.getRank() != 0)
      return rewriter.notifyMatchFailure(loadOp, "only 0D memrefs supported");

    Value ptr = adaptor.getMemref();
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp, typeConverter->convertType(memrefType.getElementType()), ptr,
        loadOp.getAlignment().value_or(0), false, loadOp.getNontemporal());
    return success();
  }
};

struct ConvertMemrefStore : public OpConversionPattern<memref::StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memrefType = cast<MemRefType>(storeOp.getMemRefType());

    // Only handle 0D memrefs.
    if (memrefType.getRank() != 0)
      return rewriter.notifyMatchFailure(storeOp, "only 0D memrefs supported");

    Value ptr = adaptor.getMemref();
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        storeOp, adaptor.getValue(), ptr, storeOp.getAlignment().value_or(0),
        false, storeOp.getNontemporal());
    return success();
  }
};

struct ConvertMemrefView : public OpConversionPattern<memref::ViewOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::ViewOp viewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType<LLVM::LLVMPointerType>(
        viewOp.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(viewOp,
                                         "failed to convert result type");

    Value offset = adaptor.getByteShift();
    if (!isa<IntegerType>(offset.getType())) {
      Type i64 = rewriter.getIntegerType(64);
      offset =
          arith::IndexCastOp::create(rewriter, viewOp.getLoc(), i64, offset);
    }

    Type i8 = rewriter.getIntegerType(8);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(viewOp, resultType, i8,
                                             adaptor.getSource(), offset);
    return success();
  }
};

struct ConvertGPULaunchFunc : public OpConversionPattern<gpu::LaunchFuncOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp launchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<gpu::KernelDim3> clusterSize;
    if (launchOp.hasClusterSize())
      clusterSize =
          gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                          adaptor.getClusterSizeZ()};

    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        launchOp, launchOp.getKernelAttr(),
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                        adaptor.getGridSizeZ()},
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                        adaptor.getBlockSizeZ()},
        adaptor.getDynamicSharedMemorySize(), adaptor.getKernelOperands(),
        adaptor.getAsyncObject(), clusterSize);

    return success();
  }
};

/// Type converter for memref to LLVM lowering.
/// Converts contiguous memref types to LLVM pointer types.
class MemrefToLLVMTypeConverter : public TypeConverter {
public:
  MemrefToLLVMTypeConverter(MLIRContext *ctx) {
    // Keep all other types unchanged.
    addConversion([](Type type) { return type; });

    // Convert memref types to LLVM pointers.
    addConversion([this, ctx](MemRefType type) -> std::optional<Type> {
      // Only convert contiguous memrefs (no layout or identity layout).
      if (type.getLayout() && !type.getLayout().isIdentity())
        return std::nullopt;

      // Convert memory space attribute.
      unsigned addressSpace = 0;
      if (Attribute memorySpace = type.getMemorySpace()) {
        std::optional<Attribute> convertedSpace =
            convertTypeAttribute(type, memorySpace);
        if (!convertedSpace)
          return std::nullopt;

        if (!(*convertedSpace)) { // Conversion to default is 0.
          addressSpace = 0;
        } else if (auto explicitSpace =
                       dyn_cast_if_present<IntegerAttr>(*convertedSpace)) {
          addressSpace = explicitSpace.getInt();
        } else {
          return std::nullopt;
        }
      }

      // Convert to LLVM pointer in the memref's memory space.
      return LLVM::LLVMPointerType::get(ctx, addressSpace);
    });
  }
};

class MemrefLoweringPass
    : public water::impl::WaterMemrefLoweringPassBase<MemrefLoweringPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = &getContext();

    // Set up type converter.
    MemrefToLLVMTypeConverter typeConverter(ctx);

    // Set up conversion target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect>();

    // Mark gpu.launch_func as dynamically illegal if it has memref operands.
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([&](gpu::LaunchFuncOp op) {
      return llvm::none_of(op.getKernelOperands(), [](Value operand) {
        return isa<MemRefType>(operand.getType());
      });
    });

    // Mark operations with memref types as illegal.
    target.addDynamicallyLegalDialect<memref::MemRefDialect>(
        [&](Operation *op) {
          // Operation is legal if all its types are converted.
          return typeConverter.isLegal(op);
        });

    // Add conversion patterns.
    RewritePatternSet patterns(ctx);

    patterns.add<ConvertUnrealizedConversionCast, ConvertMemrefCast,
                 ConvertMemrefLoad, ConvertMemrefStore, ConvertMemrefView,
                 ConvertGPULaunchFunc>(typeConverter, ctx);

    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                        typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
        return typeConverter.isSignatureLegal(
            cast<FunctionType>(funcOp.getFunctionType()));

      return typeConverter.isLegal(op);
    });

    // Apply partial conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
