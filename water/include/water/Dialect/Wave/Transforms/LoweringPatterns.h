// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
#define WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

namespace wave {

enum class WaveAddressSpace : uint32_t;
class WaveHyperparameterAttr;
class WaveSymbolAttr;

// Type converter handling Wave dialect types.
class WaveTypeConverter : public mlir::TypeConverter {
public:
  // Type converter is contextual in the given scope, where the scope defines
  // the hyper-parameters to be used. These feed into symbolic shapes that get
  // converted into static shapes based on hyperparameter values.
  explicit WaveTypeConverter(wave::WaveHyperparameterAttr hyperparameters);

  // Convert a Wave tensor type without constructing it first. This allows to
  // both avoid the cost of constructing a new type object and to handle more
  // general cases than the Wave tensor type currently allows for. If `shape` is
  // null, an identity shape from symbols is assumed.
  mlir::Type
  convertTensorFromComponents(llvm::ArrayRef<wave::WaveSymbolAttr> symbols,
                              mlir::AffineMap shape, mlir::Type elementType,
                              wave::WaveAddressSpace addressSpace) const;

  wave::WaveHyperparameterAttr getHyperparameters() const {
    return hyperparameters;
  }

private:
  wave::WaveHyperparameterAttr hyperparameters;
};

// Adds pattern that lowers `wave.register` to upstream MLIR ops.
void populateWaveRegisterLoweringPatterns(WaveTypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);

// Adds pattern that lowers wave binary ops to upstream MLIR ops.
void populateWaveBinaryOpLoweringPatterns(WaveTypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);

// Adds pattern that lowers 'wave.allocate' ops to upstream MLIR ops.
void populateWaveAllocateOpLoweringPatterns(WaveTypeConverter &typeConverter,
                                            mlir::RewritePatternSet &patterns);

// Adds pattern that lowers 'wave.read' and 'wave.write' ops to upstream MLIR
// ops.
void populateWaveReadWriteLoweringPatterns(WaveTypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns);

} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
