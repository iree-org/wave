// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVEINTERFACES
#define WATER_DIALECT_WAVE_WAVEINTERFACES

include "mlir/IR/OpBase.td"

//-----------------------------------------------------------------------------
// Operation traits
//-----------------------------------------------------------------------------

def HasWaveIndexMapping : NativeOpTrait<"HasWaveIndexMapping"> {
    let cppNamespace = "::wave";
}

//-----------------------------------------------------------------------------
// WaveInferTypeOpInterface and implementation traits
//-----------------------------------------------------------------------------

def WaveInferTypeOpInterface : OpInterface<"WaveInferTypeOpInterface"> {
  let description = [{
    Interface capturing op-specific type inference rules inside the Wave
    dialect. Type inference proceeds forward and backward on a function in order
    to fully resolve shapes. Each operation may indicate how to update the types
    of its results based on the types of its operands and, conversely, the types
    of its operands based on the types of its results.
  }];
  let cppNamespace = "::wave";

  let methods = [
    InterfaceMethod<[{
      Propagate type information from operands to results. Result types may be
      updated in place. If propagation would lead to a type conflict, return
      failure and print additional information into the provided error stream.
      Otherwise return an indication whether result types were updated.
      }],
      "::llvm::FailureOr<::mlir::ChangeResult>", "propagateForward",
      (ins "::llvm::ArrayRef<::wave::WaveTensorType>":$operandTypes,
           "::llvm::MutableArrayRef<::wave::WaveTensorType>":$resultTypes,
           "::llvm::raw_ostream &":$errs)
    >,
    InterfaceMethod<[{
      Propagate type information from results to operands. Operand types may be
      updated in place. If propagation would lead to a type conflict, return
      failure and print additional information into the provided error stream.
      Otherwise return an indication whether operand types were updated.
      }],
      "::llvm::FailureOr<::mlir::ChangeResult>",
      "propagateBackward",
      (ins "::llvm::MutableArrayRef<::wave::WaveTensorType>":$operandTypes,
           "::llvm::ArrayRef<::wave::WaveTensorType>":$resultTypes,
           "::llvm::raw_ostream &":$errs)
    >
  ];
}

def IdentityTypeInferenceOpTrait
    : NativeOpTrait<"IdentityTypeInferenceOpTrait"> {
  let cppNamespace = "::wave";
}

def NoOpTypeInferenceOpTrait : NativeOpTrait<"NoOpTypeInferenceOpTrait"> {
  let cppNamespace = "::wave";
}

//-----------------------------------------------------------------------------
// Type verification traits
//-----------------------------------------------------------------------------

def CompatibleOperandsAndResultsOpTrait
    : NativeOpTrait<"CompatibleOperandsAndResultsOpTrait"> {
  let cppNamespace = "::wave";
}

def CompatibleOperandsAndResultsIgnoreSpaceOpTrait
    : NativeOpTrait<"CompatibleOperandsAndResultsIgnoreSpaceOpTrait"> {
  let cppNamespace = "::wave";
}

#endif // WATER_DIALECT_WAVE_WAVEINTERFACES
