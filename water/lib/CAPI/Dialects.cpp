// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir-c/AffineMap.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Support/TypeID.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Wave, wave, ::wave::WaveDialect)

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveSymbolAttrGet(MlirContext mlirCtx,
                                    MlirStringRef symbolNameStrRef) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef symbolName = unwrap(symbolNameStrRef);
  return wrap(wave::WaveSymbolAttr::get(ctx, symbolName));
}

MlirTypeID mlirWaveSymbolAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveSymbolAttr>());
}

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveIndexMappingAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveIndexMappingAttr>(unwrap(attr));
}

MlirAttribute mlirWaveIndexMappingAttrGet(MlirContext mlirCtx,
                                          MlirAttribute *symbolNames,
                                          MlirAffineMap start,
                                          MlirAffineMap step,
                                          MlirAffineMap stride) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);

  // Convert C array of MlirAttribute to vector of WaveSymbolAttr.
  unsigned numSymbols = mlirAffineMapGetNumSymbols(start);
  assert(mlirAffineMapGetNumSymbols(step) == numSymbols &&
         "expected start and step to have the same number of dimensions");
  assert(mlirAffineMapGetNumSymbols(stride) == numSymbols &&
         "expected start and stride to have the same number of dimensions");
  llvm::SmallVector<wave::WaveSymbolAttr> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) {
        return llvm::cast<wave::WaveSymbolAttr>(unwrap(attr));
      });

  return wrap(wave::WaveIndexMappingAttr::get(ctx, symbolAttrs, unwrap(start),
                                              unwrap(step), unwrap(stride)));
}

MlirTypeID mlirWaveIndexMappingAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveIndexMappingAttr>());
}

//===---------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveHyperparameterAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveHyperparameterAttr>(unwrap(attr));
}

MlirAttribute mlirWaveHyperparameterAttrGet(MlirAttribute mapping) {
  auto dictAttr = llvm::cast<mlir::DictionaryAttr>(unwrap(mapping));

  mlir::MLIRContext *ctx = dictAttr.getContext();

  assert(llvm::all_of(dictAttr,
                      [](const mlir::NamedAttribute &namedAttr) {
                        return llvm::isa<mlir::IntegerAttr>(
                            namedAttr.getValue());
                      }) &&
         "expected mapping to contain only integer values");

  return wrap(wave::WaveHyperparameterAttr::get(ctx, dictAttr));
}

MlirTypeID mlirWaveHyperparameterAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveHyperparameterAttr>());
}

//===---------------------------------------------------------------------===//
// WaveAddressSpaceAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveAddressSpaceAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveAddressSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirWaveAddressSpaceAttrGet(MlirContext mlirCtx, uint32_t value) {
  return wrap(wave::WaveAddressSpaceAttr::get(
      unwrap(mlirCtx), static_cast<wave::WaveAddressSpace>(value)));
}

uint32_t mlirWaveAddressSpaceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<wave::WaveAddressSpaceAttr>(unwrap(attr)).getValue());
}

MlirTypeID mlirWaveAddressSpaceAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveAddressSpaceAttr>());
}

//===---------------------------------------------------------------------===//
// WaveWaveExprListAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveWaveExprListAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveExprListAttr>(unwrap(attr));
}

MlirAttribute mlirWaveWaveExprListAttrGet(MlirAttribute *symbolNames,
                                          MlirAffineMap map) {
  mlir::MLIRContext *ctx = unwrap(map).getContext();

  unsigned numSymbols = mlirAffineMapGetNumSymbols(map);
  llvm::SmallVector<wave::WaveSymbolAttr> symbolAttrs = llvm::map_to_vector(
      llvm::make_range(symbolNames, symbolNames + numSymbols),
      [](MlirAttribute attr) {
        return llvm::cast<wave::WaveSymbolAttr>(unwrap(attr));
      });

  return wrap(wave::WaveExprListAttr::get(ctx, symbolAttrs, unwrap(map)));
}

MlirTypeID mlirWaveWaveExprListAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveExprListAttr>());
}

//===---------------------------------------------------------------------===//
// WaveReadWriteBoundsAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveReadWriteBoundsAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveReadWriteBoundsAttr>(unwrap(attr));
}

MlirAttribute mlirWaveReadWriteBoundsAttrGet(MlirAttribute mapping) {
  auto dictAttr = llvm::cast<mlir::DictionaryAttr>(unwrap(mapping));

  mlir::MLIRContext *ctx = dictAttr.getContext();

  assert(llvm::all_of(dictAttr,
                      [](const mlir::NamedAttribute &namedAttr) {
                        return llvm::isa<wave::WaveExprListAttr>(
                            namedAttr.getValue());
                      }) &&
         "expected mapping to contain only WaveWaveExprListAttr values");

  return wrap(wave::WaveReadWriteBoundsAttr::get(ctx, dictAttr));
}

MlirTypeID mlirWaveReadWriteBoundsAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveReadWriteBoundsAttr>());
}
