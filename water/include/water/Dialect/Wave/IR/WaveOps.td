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
// Type constraints for Wave operations.
//-----------------------------------------------------------------------------

// Named constraint for types supported by wave.iterate and wave.yield operations.
// Supports both WaveTensorType (before PropagateElementsPerThread pass) and
// 1D vectors (after PropagateElementsPerThread pass).
def WaveIterableType : AnyTypeOf<[WaveTensorType, VectorOfRank<[1]>],
                                 "wave tensor or 1d vector type">;

// Memory type for allocate results and read/write memory operands.
// Before ResolveDistributedAllocations: WaveTensorType with memory address space.
// After ResolveDistributedAllocations: MemRefType with resolved distributed shape.
def WaveMemoryType : Type<Or<[WaveTensorInMemory.predicate, AnyMemRef.predicate]>,
                          "wave tensor in memory or memref">;

//-----------------------------------------------------------------------------
// Base class for all Wave operations.
//-----------------------------------------------------------------------------

// Wave operations must have an optional attribute for index sequences.
class WaveOp<string mnemonic, list<Trait> traits = []>
    : Op<WaveDialect, mnemonic, !listconcat(traits, [HasWaveIndexMapping])> {
  dag commonArguments = (ins
    Arg<OptionalAttr<DictArrayAttr>, "Index expression">:$index
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
          CompatibleOperandsAndResultsOpTrait,
          WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]>,
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
          CompatibleOperandsAndResultsOpTrait,
          WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]>,
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
    [DeclareOpInterfaceMethods<WaveInferTypeOpInterface>,
     DeclareOpInterfaceMethods<WaveElementsPerThreadOpInterface>,
     DeclareOpInterfaceMethods<WaveInferIndexExprsOpInterface,
      ["initializeIndexExprsForward", "initializeIndexExprsBackward"]>]>,
            WaveArithmeticOpDoc {
  let summary = "Matrix multiply and accumulate";

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorInRegister, "Accumulator for addition">:$accumulator,
    Arg<OptionalAttr<WaveMmaKindAttr>, "Kind of the MMA intrinsic to target">:$kind
  ), commonArguments);

  let results = (outs
    Arg<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Compute the expected elements per thread for a specific operand of this MMA operation.
    /// Returns failure if no hardware constraints are available.
    llvm::FailureOr<unsigned> computeElementsPerThreadForOperand(unsigned operandIndex);
  }];
}

//-----------------------------------------------------------------------------
// Control flow operations
//-----------------------------------------------------------------------------

def IterateOp : Op<WaveDialect, "iterate", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["areTypesCompatible", "getEntrySuccessorOperands"]>,
    DeclareOpInterfaceMethods<WaveInferIndexExprsOpInterface>]> {
  let summary = "Executes the body repeatedly";
  let description = [{
    Intrinsically sequential iterative execution that is akin to a loop with
    bounds that are not yet specified. Instead, the iteration is understood to
    be performed along the symbolic dimension that will later be instantiated to
    a concrete value.

    Similarly to other loop-like constructs, this operation uses secondary
    induction variables for loop-carried values. The initial values are supplied
    as `iter_args` and the resulting values are produced by `result`.

    It optionally takes a list of captured values that are re-mapped to block
    arguments. Unlike induction variables, these values cannot be updated and are
    not returned from the loop. They are meant purely to indicate a distribution
    scheme (index expressions) that is aware of tiling constraints applicable to
    the loop.
  }];

  let arguments = (ins
    Arg<WaveSymbolAttr, "Iterator symbol">:$iterator,
    // Accept both WaveTensorType (before PropagateElementsPerThread) and 1D vectors (after).
    // We cannot use Arg<Variadic<WaveTensorInRegisters>> because Variadic
    // requires a Type and not TypeConstraint.
    Arg<Variadic<WaveIterableType>, "Carried values">:$iter_args,
    Arg<Variadic<WaveIterableType>, "Captured values">:$captures
  );

  let results = (outs
    // Results follow the same type constraints as inputs.
    Res<Variadic<WaveIterableType>, "Yielded values">:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat =
    "custom<SingleSymbol>($iterator) (`iter_args` `(` $iter_args^ `)`)?"
    "(`captures` `(` $captures^ `)`)?"
    "attr-dict-with-keyword regions `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    // Get the body block.
    ::mlir::Block *getLoopBody() {
      assert(llvm::hasSingleElement(getBody()));
      return &getBody().front();
    }

    // Get the block arguments corresponding to captured values.
    ::mlir::ValueRange getCaptureBlockArgs() {
      return getLoopBody()->getArguments().take_back(getCaptures().size());
    }

    // Rewrite the loop to turn all values defined above and used inside the
    // body into captures.
    void makeIsolated(::mlir::RewriterBase &rewriter);

    // Rewrite the loop to directly use values instead of capturing them.
    void makeNonIsolated(::mlir::RewriterBase &rewriter);
  }];
}

def YieldOp : Op<WaveDialect, "yield",
    [Terminator, HasParent<"::wave::IterateOp">,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "Yields values from the current control flow context";

  let arguments = (ins
    // Must match the type constraints of wave.iterate results.
    Arg<Variadic<WaveIterableType>, "Yielded values">:$values
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
    Arg<Optional<WaveMemoryType>, "Optional parent buffer to view into">:$parent,
    Arg<WaveExprListAttr, "Distributed (physical) shape">:$distributed_shape,
    Arg<OptionalAttr<I64Attr>, "Byte offset into parent buffer">:$offset
  ), commonArguments);

  let results = (outs Res<WaveMemoryType, "Allocated buffer">:$result);

  let assemblyFormat =
    " (`in` $parent^ `:` type($parent))? attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def ExtractSliceOp : WaveOp<"extract_slice", [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait, CompatibleOperandsAndResultsOpTrait, NoOpElementsPerThreadOpTrait]> {
  let summary = "Extracts a subvector from an n-D tensor";
  let description = [{
    Extracts an n-D subvector from an n-D tensor using k-D offset, size, and
    stride wave expression attributes, where k <= n. When k < n, the first k
    dimensions are sliced while the remaining (n-k) dimensions are preserved.
  }];

  let arguments = (ins
    Arg<WaveTensorInRegister, "Memory to extract slice from">:$memory,
    Arg<WaveExprListAttr>:$offset,
    Arg<WaveExprListAttr>:$size,
    Arg<WaveExprListAttr>:$stride
  );

  let results = (outs
    Res<WaveTensorInRegister, "Extracted values">:$result
  );

  let assemblyFormat = "$memory attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def ReadOp : WaveOp<"read", [
    WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
    DeclareOpInterfaceMethods<WaveElementsPerThreadOpInterface>,
    CompatibleOperandsAndResultsIgnoreSpaceOpTrait,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Reads from memory";
  let description = [{
    Moves data from a memory-resident tensor to a register-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveMemoryType, "Memory to read from">:$memory,
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
    NoOpElementsPerThreadOpTrait,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
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
    DeclareOpInterfaceMethods<WaveElementsPerThreadOpInterface>,
    CompatibleOperandsAndResultsIgnoreSpaceOpTrait,
    DeclareOpInterfaceMethods<WaveInferIndexExprsOpInterface>]> {
  let summary = "Writes into memory";
  let description = [{
    Moves data from a register-resident tensor into a memory-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to write">:$value_to_store,
    Arg<WaveMemoryType, "Memory to write into">:$memory,
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

//-----------------------------------------------------------------------------
// Miscellaneous operations
//-----------------------------------------------------------------------------

def CastOp : WaveOp<"cast", [
    WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
    WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Cast a register-resident tensor from one type to another";
  let description = [{
    Change the type of elements in the register-resident tensor without
    changing its shape if specified. Supports integer, floating point casts
    and combinations of both. Widening integer casts are sign-extending.
  }];
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to cast">:$value_to_cast
  ), commonArguments);
  let results = (outs
    Res<WaveTensorInRegister, "Casted value">:$result
  );

  let assemblyFormat =
    "$value_to_cast " # commonArgumentsSyntax # " attr-dict `:`"
    "qualified(type($value_to_cast)) `to` qualified(type($result))";

  let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
