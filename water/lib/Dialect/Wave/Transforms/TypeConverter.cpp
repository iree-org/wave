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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-tensor-type-converter"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

wave::WaveTypeConverter::WaveTypeConverter(
    wave::WaveHyperparameterAttr hyperParameters)
    : hyperParameters(hyperParameters) {
  // Catch-all noop conversion. This will be called last.
  addConversion([](Type t) { return t; });

  addConversion([this](wave::WaveTensorType tensorType) -> Type {
    return convertTensorFromComponents(tensorType.getShape(),
                                       /*shape=*/{},
                                       tensorType.getElementType(),
                                       tensorType.getAddressSpaceValue());
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

mlir::Type wave::WaveTypeConverter::convertTensorFromComponents(
    llvm::ArrayRef<wave::WaveSymbolAttr> symbols, mlir::AffineMap shape,
    mlir::Type elementType, wave::WaveAddressSpace addressSpace) const {
  std::optional<SmallVector<int64_t>> symbolValues =
      wave::resolveSymbolNames(symbols, hyperParameters);
  if (!symbolValues)
    return nullptr;

  std::optional<SmallVector<int64_t>> staticShape =
      shape ? wave::evaluateMapWithSymbols(shape, *symbolValues) : symbolValues;
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
    // For register space, use vector type (registers are handled by LLVM)
    return VectorType::get(*staticShape, elementType);
  }

  llvm_unreachable("unsupported address space");
}
