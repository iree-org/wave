// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

struct ConvertMemrefLoad : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

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
  using OpConversionPattern::OpConversionPattern;

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

    // Mark operations with memref types as illegal.
    target.addDynamicallyLegalDialect<memref::MemRefDialect>(
        [&](Operation *op) {
          // Operation is legal if all its types are converted.
          return typeConverter.isLegal(op);
        });

    // Add conversion patterns.
    RewritePatternSet patterns(ctx);

    patterns.add<ConvertMemrefLoad, ConvertMemrefStore>(typeConverter, ctx);

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
