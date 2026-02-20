// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "water/Dialect/Wave/IR/WaveInterfaces.td"
include "water/Dialect/Wave/IR/WaveTypes.td"

#ifndef WATER_TEST_LIB_DIALECT_WATERTESTDIALECT
#define WATER_TEST_LIB_DIALECT_WATERTESTDIALECT

def WaterTestDialect : Dialect {
  let name = "water_test";
  let summary = "Dialect for testing Water";
  let cppNamespace = "::mlir::water::test";
  let useDefaultAttributePrinterParser = 1;
}

def WaveTensorOp
    : Op<WaterTestDialect, "wave_tensor",
         [WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait]> {
  let results = (outs WaveTensorType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def WaveFailPropagationOp
  : Op<WaterTestDialect, "wave_fail_propagation",
       [DeclareOpInterfaceMethods<WaveInferTypeOpInterface>]> {
  let arguments = (ins WaveTensorType:$in1,
                       WaveTensorType:$in2,
                       UnitAttr:$forward,
                       UnitAttr:$backward);
  let results = (outs WaveTensorType:$out1,
                      WaveTensorType:$out2);
  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` functional-type(operands, results)";
}

// Dummy op used to test WaveSymbolMappingAttr. When `expected_num_results` is
// provided the op verifier checks that every value in the mapping has exactly
// that many results.
def WaveSymbolMappingOp
    : Op<WaterTestDialect, "wave_symbol_mapping"> {
  let summary = "Test op that holds a WaveSymbolMappingAttr";
  let arguments = (ins
    WaveSymbolMappingAttr:$mapping,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$expected_num_results
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WATER_TEST_LIB_DIALECT_WATERTESTDIALECT
