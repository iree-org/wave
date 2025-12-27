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
// Elements-per-thread propagation interface and implementation traits
//-----------------------------------------------------------------------------

def WaveElementsPerThreadOpInterface
    : OpInterface<"WaveElementsPerThreadOpInterface"> {
  let description = [{
    Interface capturing op-specific rules for inferring and propagating the
    number of elements processed by each work item (thread) in the workgroup
    when lowering the Wave dialect programming model to the SIMT programming
    model. Each operation may indicate how to infer the elements per threads for
    its results based on the operation itself and the elements per thread values
    of its operands, and vice versa.

    Several common implementations are provided via traits.
  }];
  let methods = [
    InterfaceMethod<[{
      Propagate elements per threads information from operands to results.
      Expected to return failure when a conflict is detected after printing
      a descriptive error message to the error stream provided.
      }],
      "::llvm::FailureOr<mlir::ChangeResult>",
      "propagateElementsPerThreadForward",
      (ins "::llvm::ArrayRef<::wave::ElementsPerThreadLatticeValue>"
               :$operandElements,
           "::llvm::MutableArrayRef<::wave::ElementsPerThreadLatticeValue>"
               :$resultElements,
           "::llvm::raw_ostream &":$errs)
    >,
    InterfaceMethod<[{
      Propagate elements per threads information from results to operands.
      Expected to return failure when a conflict is detected after printing
      a descriptive error message to the error stream provided.
      }],
      "::llvm::FailureOr<mlir::ChangeResult>",
      "propagateElementsPerThreadBackward",
      (ins "::llvm::MutableArrayRef<::wave::ElementsPerThreadLatticeValue>"
               :$operandElements,
           "::llvm::ArrayRef<::wave::ElementsPerThreadLatticeValue>"
               :$resultElements,
           "::llvm::raw_ostream &":$errs)
    >,
  ];
  let cppNamespace = "::wave";
}

def NoOpElementsPerThreadOpTrait
    : NativeOpTrait<"NoOpElementsPerThreadOpTrait"> {
  let cppNamespace = "::wave";
}

def IdentityElementsPerThreadOpTrait
    : NativeOpTrait<"IdentityElementsPerThreadOpTrait"> {
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

//-----------------------------------------------------------------------------
// WaveInferIndexExprsOpInterface and implementation traits
//-----------------------------------------------------------------------------

def WaveInferIndexExprsOpInterface : OpInterface<"WaveInferIndexExprsOpInterface"> {
  let description = [{
    Interface capturing op-specific index expression inference rules inside the
    Wave dialect. Index expression inference proceeds forward and backward on a
    function in order to compute index expressions for all values. For legacy
    reasons, MMA and reduction operations capture index expressions for their
    operands in addition to results, and write operations capture a single index
    expression that is common to operations. Each operation may indicate how to
    propagate index expressions from its operands to its results and, conversely,
    from its results to its operands.
  }];
  let cppNamespace = "::wave";

  let methods = [
    InterfaceMethod<[{
      Propagate index expression information from operands to results. Result
      index expressions may be updated in place. If propagation would lead to a
      conflict, return failure and print additional information into the
      provided error stream. Otherwise return an indication whether result
      index expressions were updated.
      }],
      "::llvm::FailureOr<::mlir::ChangeResult>", "propagateIndexExprsForward",
      (ins "::llvm::ArrayRef<::wave::IndexExprsLatticeStorage>":$operandExprs,
           "::llvm::MutableArrayRef<::wave::IndexExprsLatticeStorage>":$resultExprs,
           "::wave::EmitErrorFn":$emitError)
    >,
    InterfaceMethod<[{
      Propagate index expression information from results to operands. Operand
      index expressions may be updated in place. If propagation would lead to a
      conflict, return failure and print additional information into the
      provided error stream. Otherwise return an indication whether operand
      index expressions were updated.
      }],
      "::llvm::FailureOr<::mlir::ChangeResult>",
      "propagateIndexExprsBackward",
      (ins "::llvm::MutableArrayRef<::wave::IndexExprsLatticeStorage>":$operandExprs,
           "::llvm::ArrayRef<::wave::IndexExprsLatticeStorage>":$resultExprs,
           "::wave::EmitErrorFn":$emitError)
    >,
    InterfaceMethod<[{
      Initialize index expression information for results of the current
      operation. By default, they are initialized to lattice bottom and will be
      updated by the analysis. Certain operations may want to initialize them
      to specific values, e.g., MMA operations will initialize result indices
      based on the targeted hardware layout. The initialization object contains
      pre-processed constraints relevant to the current operation. The name
      forward implies that this function is called from the forward analysis.

      If initialization is impossible or would immediately lead to a
      propagation conflict, report the error using the function provided and
      return a failure. Otherwise, update the `resultExprs` and return success.
      }],
      "::llvm::LogicalResult",
      "initializeIndexExprsForward",
      (ins "::llvm::MutableArrayRef<::wave::IndexExprsLatticeStorage>":$resultExprs,
           "const ::wave::IndexExprsAnalysisInit &":$initObject,
           "::wave::EmitErrorFn":$emitError),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::success();
      }]
    >,
    InterfaceMethod<[{
      Initialize index expression information for operands of the current
      operation. By default, they are initialized to lattice bottom and will be
      updated by the analysis. Certain operations may want to initialize them
      to specific values, e.g., MMA operations will initialize operand indices
      based on the targeted hardware layout. The initialization object contains
      pre-processed constraints relevant to the current operation. The name
      backward implies this function is called from the backward analysis.

      If initialization is impossible or would immediately lead to a
      propagation conflict, report the error using the function provided and
      return a failure. Otherwise, update the `resultExprs` and return success.
      }],
      "::llvm::LogicalResult",
      "initializeIndexExprsBackward",
      (ins "::llvm::MutableArrayRef<::wave::IndexExprsLatticeStorage>":$operandExprs,
           "const ::wave::IndexExprsAnalysisInit &":$initObject,
           "::wave::EmitErrorFn":$emitError),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::success();
      }]
    >,
    InterfaceMethod<[{
      Set the index attribute of the current operation based on (converged)
      operand and result index expressions. Even though the index expression is
      conceptually a property of a value, it is typically attached to the
      operation producing these values. Some operations may require special
      handling, e.g., operations with regions that may also want to capture
      index expressions of their block arguments or certain operations like MMA
      that also expect operand index expressions for legacy reasons.
      }],
      "::llvm::LogicalResult",
      "setIndexFromLattices",
      (ins "::llvm::ArrayRef<::wave::IndexExprsLatticeStorage>":$operandExprs,
           "::llvm::ArrayRef<::wave::IndexExprsLatticeStorage>":$resultExprs)
    >
  ];
}

def IdentityIndexExprsOpTrait
    : NativeOpTrait<"IdentityIndexExprsOpTrait"> {
  let cppNamespace = "::wave";
}

#endif // WATER_DIALECT_WAVE_WAVEINTERFACES
