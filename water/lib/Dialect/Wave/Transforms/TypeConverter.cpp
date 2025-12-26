// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-tensor-type-converter"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

wave::WaveTypeConverter::WaveTypeConverter(
    wave::WaveHyperparameterAttr hyperparameters)
    : hyperparameters(hyperparameters) {
  // Catch-all noop conversion. This will be called last.
  addConversion([](Type t) { return t; });

  addConversion([this](wave::WaveTensorType tensorType) -> Type {
    ArrayRef<Attribute> symbols(tensorType.getShape().begin(),
                                tensorType.getShape().end());
    return convertTensorFromComponents(symbols,
                                       /*shape=*/{},
                                       tensorType.getElementType(),
                                       tensorType.getAddressSpaceValue());
  });

  addConversion([this](VectorType vectorType) -> Type {
    if (vectorType.isScalable() || vectorType.getRank() != 1)
      return nullptr;

    Type elementType = convertType(vectorType.getElementType());
    if (elementType == vectorType.getElementType())
      return vectorType;

    return VectorType::get(vectorType.getShape(), elementType);
  });

  addSourceMaterialization([](OpBuilder &builder, wave::WaveTensorType waveType,
                              ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, waveType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
}

Type wave::WaveTypeConverter::convertTensorFromComponents(
    llvm::ArrayRef<Attribute> symbols, AffineMap shape, Type elementType,
    wave::WaveAddressSpace addressSpace) const {
  if (any_of(symbols, llvm::IsaPred<wave::WaveIndexSymbolAttr>))
    return nullptr;

  std::optional<SmallVector<int64_t>> staticShape =
      shape ? wave::evaluateMapWithHyperparams(shape, symbols, hyperparameters)
            : wave::resolveSymbolNames(symbols, hyperparameters);

  if (!staticShape)
    return nullptr;

  elementType = convertType(elementType);
  if (!elementType)
    return nullptr;

  switch (addressSpace) {
  case wave::WaveAddressSpace::Unspecified:
    LLVM_DEBUG(DBGS() << "address spaces must have been specified\n");
    return nullptr;

  case wave::WaveAddressSpace::Global: {
    // GPU global memory (device memory)
    auto globalMemoryAddressSpace = gpu::AddressSpaceAttr::get(
        elementType.getContext(), gpu::AddressSpace::Global);
    return MemRefType::get(*staticShape, elementType,
                           /*layout=*/MemRefLayoutAttrInterface{},
                           globalMemoryAddressSpace);
  }

  case wave::WaveAddressSpace::Shared: {
    // GPU shared memory
    auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
        elementType.getContext(), gpu::AddressSpace::Workgroup);
    return MemRefType::get(*staticShape, elementType,
                           /*layout=*/MemRefLayoutAttrInterface{},
                           workgroupMemoryAddressSpace);
  }

  case wave::WaveAddressSpace::Register:
    return nullptr;
  }

  llvm_unreachable("unsupported address space");
}
