// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveAttrs.td"
include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

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
    : Op<WaveDialect, mnemonic>,
      WaveArithmeticOpDoc {
  let arguments = (ins
    Arg<WaveTensorInRegister, "Argument">:$argument
  );

  let results = (outs
    Arg<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "$argument attr-dict `:` functional-type(operands, results)";
}

class BinaryWaveOp<string mnemonic>
    : Op<WaveDialect, mnemonic>,
      WaveArithmeticOpDoc {
  let arguments = (ins
    Arg<WaveTensorInRegister, "Left-hand side">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side">:$rhs
  );

  let results = (outs
    Res<WaveTensorInRegister, "Result of the operation">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
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

def MmaOp
    : Op<WaveDialect, "mma">,
      WaveArithmeticOpDoc {
  let summary = "Matrix multiply and accumulate";
  let description = baseDescription # [{
    The mandatory `kind` attribute indicates which configuration to use in
    emitting matrix core instructions. Operand and result tensors are expected
    to have the corresponding elemental types. In particular, the first
    component of the `kind`-s textual format indicates the accumulator and
    result elemental type and the last component indicates the LHS/RHS elemental
    type.
  }];

  let arguments = (ins
    Arg<WaveTensorInRegister, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorInRegister, "Accumulator for addition">:$accumulator,
    Arg<WaveMmaKindAttr, "Kind of the MMA intrinsic to target">:$kind
  );

  let results = (outs
    Res<WaveTensorInRegister, "Accumulated result">:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $accumulator attr-dict `:`"
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

def ReadOp : Op<WaveDialect, "read"> {
  let summary = "Reads from memory";
  let description = [{
    Moves data from a memory-resident tensor to a register-resident tensor
    preserving the shape.
  }];

  let arguments = (ins
    Arg<WaveTensorInMemory, "Tensor representing memory to read from">:$memory
  );

  let results = (outs
    Res<WaveTensorInRegister, "Read value">:$result
  );

  let assemblyFormat =
    "$memory attr-dict `:` functional-type(operands, results)";
}

def RegisterOp : Op<WaveDialect, "register"> {
  let summary = "Defines a tensor value known to be placed in a register";
  let description = [{
    Defines a new register-resident tensor initialized with the given scalar
    value. The (symbolic) shape of the tensor may be specified in the result
    type.
  }];

  let arguments = (ins
    Arg<Type<Or<[AnyInteger.predicate, AnyFloat.predicate]>>,
        "Scalar value to initialize the tensor elements">:$init
  );

  let results = (outs
    Res<WaveTensorInRegister, "Defined value">:$result
  );

  let assemblyFormat =
    "$init attr-dict `:` custom<RegisterOpTypes>(type($init), type($result))";
  let hasVerifier = 1;
}

def WriteOp : Op<WaveDialect, "write"> {
  let summary = "Writes into memory";
  let description = [{
    Moves data from a register-resident tensor into a memory-resident tensor
    preserving the shape.
  }];

  let arguments = (ins
    Arg<WaveTensorInRegister, "Value to write">:$value_to_store,
    Arg<WaveTensorInMemory, "Tensor representing memory to write into">:$memory
  );

  let assemblyFormat = "$value_to_store `,` $memory attr-dict `:`"
                       "type($value_to_store) `,` type($memory)";
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
