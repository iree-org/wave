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

class ReductionWaveOp<string mnemonic>
    : WaveOp<mnemonic,
        [WaveInferTypeOpInterface, ReductionTypeInferenceOpTrait,
         WaveElementsPerThreadOpInterface, ReductionElementsPerThreadOpTrait,
         RequiresSidewaysBackwardPropagationOpTrait, WaveReductionOpTrait]>,
      WaveArithmeticOpDoc {
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Input tensor to reduce">:$input,
    Arg<WaveTensorInRegister, "Initial value for the reduction">:$init,
    Arg<WaveReductionScopeAttr, "Scope of the reduction">:$scope,
    Arg<OptionalAttr<WaveSymbolAttr>, "Reduction axis">:$axis
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Reduced result">:$result
  );

  let assemblyFormat =
      "$input `init` `(` $init `)` (`along` custom<SingleSymbol>($axis)^)? "
      "$scope " # commonArgumentsSyntax # " attr-dict `:`"
      "functional-type(operands, results)";
}

def SumOp : ReductionWaveOp<"sum"> {
  let summary = "Reduce a tensor by summing along a specified axis";
  let description = [{
    Reduces the input tensor by summing its elements along the specified
    symbolic axis. The initial value is used as the starting point for the
    reduction.
  }] # baseDescription;
}

def MaxElementOp : ReductionWaveOp<"max_element"> {
  let summary = "Reduce a tensor by taking the maximum along a specified axis";
  let description = [{
    Reduces the input tensor by taking the maximum of its elements along the
    specified symbolic axis. The initial value is used as the starting point
    for the reduction.
  }] # baseDescription;
}

//-----------------------------------------------------------------------------
// Arithmetic operations
//-----------------------------------------------------------------------------

def AddOp : BinaryWaveOp<"add"> {
  let summary = "Add two values";
  let description = baseDescription;
}

def SubOp : BinaryWaveOp<"sub"> {
  let summary = "Subtract two values";
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

def MaxOp : BinaryWaveOp<"max"> {
  let summary = "Get the maximum of two values";
  let description = baseDescription;
}

def MinOp : BinaryWaveOp<"min"> {
  let summary = "Get the minimum of two values";
  let description = baseDescription;
}

def Exp2Op : UnaryWaveOp<"exp2"> {
  let summary = "Compute 2 to the power of the argument";
  let description = baseDescription;
}

def ReciprocalOp : UnaryWaveOp<"reciprocal"> {
  let summary = "Compute the reciprocal (1/x) of the argument";
  let description = baseDescription;
  let hasVerifier = 1;
}

def SelectOp : WaveOp<"select",
    [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
     WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
     WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Select between two values based on a condition";
  let description = [{
    Selects between two values based on a condition. The selection is element-wise, that is,
    the condition is a tensor co-indexed with the two values.
  }];
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Condition">:$condition,
    Arg<WaveTensorInRegister, "First value">:$lhs,
    Arg<WaveTensorInRegister, "Second value">:$rhs
  ), commonArguments);
  let results = (outs
    Res<WaveTensorInRegister, "Result">:$result
  );
  let assemblyFormat =
    "$condition `,` $lhs `,` $rhs " # commonArgumentsSyntax # "attr-dict `:` "
    "functional-type(operands, results)";
  let hasVerifier = 1;
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

  let assemblyFormat = "$values attr-dict (`:` type($values)^)?";
}

//-----------------------------------------------------------------------------
// Memory-related operations
//-----------------------------------------------------------------------------

def AllocateOp : WaveOp<"allocate", [
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
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
    Arg<WaveExprListAttrOf<[WaveSymbolAttr]>,
        "Distributed (physical) shape">:$distributed_shape,
    Arg<OptionalAttr<I64Attr>, "Byte offset into parent buffer">:$offset,
    Arg<DefaultValuedAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>, "0">,
      "Padding added to the last dimension of distributed shape in number of elements">:$padding,
    Arg<OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>, [{
      Padding added to the end of an allocation in number of elements,
      does not affect the distributed shape
    }]
    >:$tail_padding
  ), commonArguments);

  let results = (outs Res<WaveMemoryType, "Allocated buffer">:$result);

  let assemblyFormat =
    " (`in` $parent^ `:` type($parent))? attr-dict `:` type($result)";

  let hasVerifier = 1;
}

def ExtractOp : WaveOp<"extract",
    [DeclareOpInterfaceMethods<WaveElementsPerThreadOpInterface>]> {
  let summary = "Extracts a single element from a vector at the given index";
  let description = [{
    This is an internal operation that appears during expansion/lowering and
    is not intended for use in user code. As such, it intentionally does not
    implement the interfaces for type and index dataflow analyses as it only
    appears after those. It does, however, implement the elements per thread
    interface as it is needed for lowering to vectors. User code should use
    `extract_slice` instead.

    When lowered, extracts a single element from the vector that corresponds
    to lowered register-resident tensor, producing a 1-element vector.
    Therefore, the result tensor is indexed by one less dimension that the
    source tensor.

    TODO: we don't have a good way of verifying this, but the shape dimension
    removed from the result must be the only one with non-unit index step
    (second element of the index expression) when such a
    symbol exists (and presumably the last / the one mapped to thread X
    otherwise).

    TODO: this operation leaks the internal abstraction of vector registers.
    Consider regrouping it with `reshape` into a single relayout operation.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Vector to extract from">:$source,
    Arg<WaveExprListAttr, "Index position to extract">:$position
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Extracted 1-element vector">:$result
  );

  let assemblyFormat =
    "$source `[` $position `]` " # commonArgumentsSyntax # "attr-dict "
    "`:` functional-type(operands, results)";
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
    Arg<WaveExprListAttrOf<[WaveSymbolAttr]>>:$offset,
    Arg<WaveExprListAttrOf<[WaveSymbolAttr]>>:$size,
    Arg<WaveExprListAttrOf<[WaveSymbolAttr]>>:$stride
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
        "Bound expressions for each symbolic dimension">:$bounds,
    Arg<OptionalAttr<WaveSymbolArrayAttr>,
        "Ordered dimension symbols from memory type shape">:$ordered_syms
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Read value">:$result
  );

  let assemblyFormat = "$memory " # commonArgumentsSyntax # "attr-dict `:` "
                       "functional-type(operands, results)";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Set ordered_syms from an ArrayRef of WaveSymbolAttr.
    void setOrderedSyms(llvm::ArrayRef<::wave::WaveSymbolAttr> syms) {
      llvm::SmallVector<::mlir::Attribute> attrs(syms.begin(), syms.end());
      setOrderedSymsAttr(::mlir::ArrayAttr::get(getContext(), attrs));
    }
  }];
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
    DeclareOpInterfaceMethods<WaveInferIndexExprsOpInterface>,
    RequiresSidewaysBackwardPropagationOpTrait]> {
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
        "Bound expressions for each symbolic dimension">:$bounds,
    Arg<OptionalAttr<WaveSymbolArrayAttr>,
        "Ordered dimension symbols from memory type shape">:$ordered_syms
  ), commonArguments);

  let assemblyFormat =
    "$value_to_store `,` $memory " # commonArgumentsSyntax # "attr-dict `:`"
    "qualified(type($value_to_store)) `,` qualified(type($memory))";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Set ordered_syms from an ArrayRef of WaveSymbolAttr.
    void setOrderedSyms(llvm::ArrayRef<::wave::WaveSymbolAttr> syms) {
      llvm::SmallVector<::mlir::Attribute> attrs(syms.begin(), syms.end());
      setOrderedSymsAttr(::mlir::ArrayAttr::get(getContext(), attrs));
    }
  }];
}

//-----------------------------------------------------------------------------
// Shuffle operation
//-----------------------------------------------------------------------------

def ShuffleOp : WaveOp<"shuffle", [
    WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
    WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Shuffle data between threads in a wave/warp";
  let description = [{
    Performs a GPU shuffle operation to exchange data between threads in a
    wave/warp. The shuffle mode determines how the data is exchanged:
    - `xor`: Exchange with thread whose lane ID differs by XOR of offset.
    - `down`: Get value from thread with higher lane ID (by offset).
    - `up`: Get value from thread with lower lane ID (by offset).
    - `idx`: Get value from thread with absolute lane ID given by offset.

    The `offset` specifies the lane offset or target lane depending on mode.
    The `width` specifies the logical warp width for the shuffle operation.

    Example:
    ```mlir
    %result = wave.shuffle xor %value, 1, 64 : (vector<1xf16>) -> vector<1xf16>
    ```
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to shuffle">:$value,
    Arg<I32Attr, "Shuffle offset">:$offset,
    Arg<I32Attr, "Logical warp width">:$width,
    Arg<WaveShuffleModeAttr, "Shuffle mode">:$mode
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Shuffled result">:$result
  );

  let assemblyFormat =
    "$mode $value `,` $offset `,` $width " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
}

//-----------------------------------------------------------------------------
// Miscellaneous operations
//-----------------------------------------------------------------------------

def ApplyExprOp : WaveOp<"apply_expr",
    [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait,
     WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
     CompatibleOperandsAndResultsShapeOpTrait,
     WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]>,
    WaveArithmeticOpDoc {
  let summary = "Apply a symbolic expression to one or more register values";
  let description = [{
    Applies a symbolic expression to one or more register-resident tensor
    values. The expression is specified as a `WaveExprListAttr` containing an
    affine map. Symbols in the expression correspond to one of the operands or
    to an external symbolic constant. An optional `combinator` may be provided
    to combine multiple results of the affine map together in a non-affine way.
    If not provided, the map must have a single result. If a comparison
    combinator is provided, the map must have exactly two results. For min/max
    combinators, the map must have at least one result.

    All operands must have the same type and indexing dimensions. The result
    type is the same as the operand types except for comparison combinators
    where it may be an arbitrary integer type, though only the last bit of
    this type will be set if the comparison is true.

    TODO(#901): tighten the constraint on comparison result to be i1, which is
    currently impossible because python typing is broken.

    Operand symbols are represented using `WaveOperandAttr` (`#wave.operand`)
    with the operand index (0 for first operand, 1 for second, etc.). In
    expressions they appear as `_Operand_0`, `_Operand_1`, etc. External
    symbolic constants use `WaveSymbolAttr` (`#wave.symbol`).
  }] # baseDescription;

  // Accept both WaveTensorType (before PropagateElementsPerThread) and
  // 1D vectors (after). We cannot use Variadic<WaveTensorInRegister> because
  // Variadic requires a Type and not a TypeConstraint.
  let arguments = !con((ins
    Arg<Variadic<WaveIterableType>, "Input registers">:$arguments,
    Arg<WaveExprListAttrOf<[WaveSymbolAttr, WaveOperandAttr]>, "Expression to apply">:$expr,
    Arg<OptionalAttr<WaveApplyExprCombinatorAttr>, "Combinator to apply">:$combinator
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "`(` $arguments `)` ($combinator^)? $expr " # commonArgumentsSyntax
    # "attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

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

def BroadcastOp : WaveOp<"broadcast", [
    DeclareOpInterfaceMethods<WaveInferTypeOpInterface>,
    DeclareOpInterfaceMethods<WaveElementsPerThreadOpInterface>,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Broadcast a tensor to a larger shape by replicating values";
  let description = [{
    Broadcasts the source tensor to the result shape by replicating values
    along the broadcast dimensions, which must be the trailing dimensions
    of the result shape.

    When either operand or result are underspecified, the broadcast dimension
    must be explicitly provided. Otherwise, it is inferred from the difference
    between their respective symbolic shapes.

    Developer remark: unless one of the dimensions introduced by broadcasting
    is indexed by a symbol that is mapped to thread X, this operation does not
    change elements-per-thread.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Source tensor to broadcast">:$source,
    Arg<OptionalAttr<WaveSymbolArrayAttr>, "Dimensions being broadcast">:$broadcast_dims
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Broadcasted result">:$result
  );

  let assemblyFormat =
    "$source (`dims` custom<SymbolArray>($broadcast_dims)^)?" #
    commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Infer broadcast dimensions from types. Expects both source and result
    // types to be fully specified tensor types.
    ::llvm::SmallVector<::wave::WaveSymbolAttr> inferBroadcastDims();
  }];
}

def SelfIndexOp : WaveOp<"self_index", [
    WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait,
    NoOpElementsPerThreadOpTrait,
    WaveInferIndexExprsOpInterface, IdentityIndexExprsOpTrait]> {
  let summary = "Produce a vector of indices along a specified dimension";
  let description = [{
    Creates a register-resident tensor containing the index values for each
    thread along the specified symbolic dimension. The index computation uses
    the index mapping associated with the operation: for each element position
    i, the value is `start + i * stride`, where start and stride come from
    the index mapping for the specified dimension.

    The result type must be a 1-dimensional tensor (or vector) with an integer
    element type, indexed by the specified dimension.

    Example:
    ```mlir
    %idx = wave.self_index @N index {...} : !wave.tensor<[N], i32, register>
    ```
  }];

  let arguments = !con((ins
    Arg<WaveSymbolAttr, "Dimension to index over">:$dim,
    Arg<OptionalAttr<I64Attr>,
        "Number of elements processed by each thread">:$elements_per_thread
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Index values">:$result
  );

  let assemblyFormat =
    "custom<SingleSymbol>($dim) " # commonArgumentsSyntax # " attr-dict `:`"
    " type($result)";
  let hasVerifier = 1;
}

def PermuteOp : WaveOp<"permute", [
    DeclareOpInterfaceMethods<WaveInferTypeOpInterface>,
    WaveElementsPerThreadOpInterface, IdentityElementsPerThreadOpTrait,
    DeclareOpInterfaceMethods<WaveInferIndexExprsOpInterface>]> {
  let summary = "Permute the dimensions of a register-resident tensor";
  let description = [{
    Reorders the symbolic dimensions of a register-resident tensor according
    to the result type shape. This operation is primarily a semantic marker
    that affects how index expressions are transformed during compilation.
    At lowering time, the operation is a pass-through since the actual data
    layout in registers remains unchanged - only the interpretation of which
    dimension each element belongs to changes.

    For example, permuting a tensor with shape [B, M, N] to [M, N, B] swaps
    the strides associated with the symbolic dimensions in the index
    expressions.

    The result type, when fully-specified, must be a permutation of the input
    shape (same symbols in a different order). If the result type is not
    fully-specified, its shape will be inferred from the input type during
    type inference.
  }];
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to permute">:$value
  ), commonArguments);
  let results = (outs
    Res<WaveTensorInRegister, "Permuted value">:$result
  );

  let assemblyFormat =
    "$value " # commonArgumentsSyntax # " attr-dict `:`"
    "qualified(type($value)) `to` qualified(type($result))";

  let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
