// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveAttrs.td"
include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveInterfaces.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

//-----------------------------------------------------------------------------
// Base class for all Wave operations.
//-----------------------------------------------------------------------------

// Wave operations must have an optional attribute for index sequences.
class WaveOp<string mnemonic, list<Trait> traits = []>
    : Op<WaveDialect, mnemonic, !listconcat(traits, [HasWaveIndexMapping])> {
  dag commonArguments = (ins
    Arg<OptionalAttr<DictionaryAttr>, "Index expression">:$index
  );

  string commonArgumentsSyntax = "( `index` custom<WaveIndexDict>($index)^ )?";
}

//-----------------------------------------------------------------------------
// Base classes for arithmetic operations
//-----------------------------------------------------------------------------

class WaveArithmeticOpDoc {
  string baseDescription = [{
    Integer overflow and signedness as well as floating point fastmath semantics
    are currently **UNDEFINED**.

    All operands and results are expected to live in registers. Previous
    operations must bring them to registers if needed.
  }];
}

class UnaryWaveOp<string mnemonic>
    : WaveOp<mnemonic,
         [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
          WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
          CompatibleOperandsAndResultsOpTrait]>,
      WaveArithmeticOpDoc {
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Argument">:$argument
  ), commonArguments);

  let results = (outs
    Arg<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "$argument attr-dict `:` functional-type(operands, results)";
}

class BinaryWaveOp<string mnemonic>
    : WaveOp<mnemonic,
         [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
          WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
          CompatibleOperandsAndResultsOpTrait]>,
      WaveArithmeticOpDoc {
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Left-hand side">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side">:$rhs
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Result of the operation">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
}

//-----------------------------------------------------------------------------
// Arithmetic operations
//-----------------------------------------------------------------------------

def AddOp : BinaryWaveOp<"add"> {
  let summary = "Add two values";
  let description = baseDescription;
}

def MulOp : BinaryWaveOp<"mul"> {
  let summary = "Multiply two values";
  let description = baseDescription;
}

def DivOp : BinaryWaveOp<"div"> {
  let summary = "Divide two values";
  let description = baseDescription;
}

def Exp2Op : UnaryWaveOp<"exp2"> {
  let summary = "Compute 2 to the power of the argument";
  let description = baseDescription;
}

def MmaOp : WaveOp<"mma",
    [DeclareOpInterfaceMethods<WaveInferTypeOpInterface>]>,
            WaveArithmeticOpDoc {
  let summary = "Matrix multiply and accumulate";

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorInRegister, "Accumulator for addition">:$accumulator,
    Arg<WaveMmaKindAttr, "Kind of the MMA intrinsic to target">:$kind
  ), commonArguments);

  let results = (outs
    Arg<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
  let hasVerifier = 1;
}

//-----------------------------------------------------------------------------
// Control flow operations
//-----------------------------------------------------------------------------

def IterateOp : Op<WaveDialect, "iterate", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["areTypesCompatible", "getEntrySuccessorOperands"]>]> {
  let summary = "Executes the body repeatedly";
  let description = [{
    Intrinsically sequential iterative execution that is akin to a loop with
    bounds that are not yet specified. Instead, the iteration is understood to
    be performed along the symbolic dimension that will later be instantiated to
    a concrete value.

    Similarly to other loop-like constructs, this operation uses secondary
    induction variables for loop-carried values. The initial values are supplied
    as `iter_args` and the resulting values are produced by `result`.
  }];

  let arguments = (ins
    Arg<WaveSymbolAttr, "Iterator symbol">:$iterator,
    Arg<Variadic<WaveTensorType>, "Carried values">:$iter_args
  );

  let results = (outs
    Res<Variadic<WaveTensorType>, "Yielded values">:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat =
    "custom<SingleSymbol>($iterator) (`iter_args` `(` $iter_args^ `)`)?"
    "attr-dict-with-keyword regions `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def YieldOp : Op<WaveDialect, "yield",
    [Terminator, HasParent<"::wave::IterateOp">,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "Yields values from the current control flow context";

  let arguments = (ins
    Arg<Variadic<WaveTensorType>, "Yielded values">:$values
  );

  let assemblyFormat = "$values attr-dict `:` type($values)";
}

//-----------------------------------------------------------------------------
// Memory-related operations
//-----------------------------------------------------------------------------

def AllocateOp : WaveOp<"allocate"> {
  let summary = "Represents an allocation in an address space";
  let description = [{
    Allocates memory for a Wave tensor in the address space indicated by the
    result type, or creates a view into an existing parent buffer starting at
    the specified 'offset' if the parent is specified.

    The `distributed_shape` specifies the size of the data allocated in each
    workgroup in terms of workgroup sizes. This allocation may be larger than a
    slice of the overall tensor, i.e. be padded, at which point the additional
    values in the allocation are not expected to be used and their value is
    unspecified. This typically happens for bank conflict avoidance purposes.

    TODO: actually verify that the distributed shape is larger and not *smaller*
    than the total shape of the tensor. This can be done by restricting the
    shape to only use block sizes and static offsets rather than arbitrarily
    complex expressions.
  }];

  let arguments = !con((ins
    Arg<Optional<WaveTensorType>, "Optional parent buffer to view into">:$parent,
    Arg<WaveExprListAttr, "Distributed (physical) shape">:$distributed_shape,
    Arg<OptionalAttr<I64Attr>, "Byte offset into parent buffer">:$offset
  ), commonArguments);

  let results = (outs Res<WaveTensorType,"Allocated buffer">:$result);

  let assemblyFormat =
    " (`in` $parent^ `:` type($parent))? attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def ReadOp : WaveOp<"read", [
    WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
    WaveElementsPerThreadOpInterface, AttrBasedElementsPerThreadOpTrait,
    CompatibleOperandsAndResultsIgnoreSpaceOpTrait]> {
  let summary = "Reads from memory";
  let description = [{
    Moves data from a memory-resident tensor to a register-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInMemory, "Tensor representing memory to read from">:$memory,
    Arg<OptionalAttr<I64Attr>,
        "Number of elements processed by each thread">:$elements_per_thread,
    Arg<OptionalAttr<WaveReadWriteBoundsAttr>,
        "Bound expressions for each symbolic dimension">:$bounds
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Read value">:$result
  );

  let assemblyFormat = "$memory " # commonArgumentsSyntax # "attr-dict `:` "
                       "functional-type(operands, results)";
  let hasVerifier = 1;
}

def RegisterOp : WaveOp<"register", [
    WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait,
    NoOpElementsPerThreadOpTrait]> {
  let summary = "Defines a tensor value known to be placed in a register";
  let description = [{
    Defines a new register-resident tensor initialized with the given scalar
    value. The (symbolic) shape of the tensor may be specified in the result
    type.
  }];

  let arguments = !con((ins
    Arg<Type<Or<[AnyInteger.predicate, AnyFloat.predicate]>>,
        "Scalar value to initialize the tensor elements">:$init
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Defined value">:$result
  );

  let assemblyFormat =
    "$init " # commonArgumentsSyntax # " attr-dict `:`"
    " custom<RegisterOpTypes>(type($init), type($result))";
  let hasVerifier = 1;
}

def WriteOp : WaveOp<"write", [
    WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait,
    WaveElementsPerThreadOpInterface, AttrBasedElementsPerThreadOpTrait,
    CompatibleOperandsAndResultsIgnoreSpaceOpTrait]> {
  let summary = "Writes into memory";
  let description = [{
    Moves data from a register-resident tensor into a memory-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to write">:$value_to_store,
    Arg<WaveTensorInMemory, "Tensor representing memory to write into">:$memory,
    Arg<OptionalAttr<I64Attr>,
        "Number of elements processed by each thread">:$elements_per_thread,
    Arg<OptionalAttr<WaveReadWriteBoundsAttr>,
        "Bound expressions for each symbolic dimension">:$bounds
  ), commonArguments);

  let assemblyFormat =
    "$value_to_store `,` $memory " # commonArgumentsSyntax # "attr-dict `:`"
    "qualified(type($value_to_store)) `,` qualified(type($memory))";

   let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
