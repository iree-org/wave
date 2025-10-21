// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVETYPES
#define WATER_DIALECT_WAVE_WAVETYPES

include "water/Dialect/Wave/IR/WaveAttrs.td"
include "water/Dialect/Wave/IR/WaveDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

//-----------------------------------------------------------------------------
// Tensor type
//-----------------------------------------------------------------------------

def WaveTensorType : TypeDef<WaveDialect, "WaveTensor"> {
  let mnemonic = "tensor";
  let description = [{
    Tensor with symbolic shapes used in Wave dialects. The exact shape may be
    unknown in the earlier stages and will be inferred later. Tensors may have
    an address space indicating whether the data is expected to live in a
    certain location, including local data store (shared memory) and registers.
  }];

  let parameters = (ins
    // TODO: extend this to a larger scope of symbol expressions as needed.
    // TODO: consider supporting yet-uninferred dimensions or groups of
    // dimensions, e.g., we may know that the trailing dimensions must be
    // certain symbols but not know the expressions for leading dimensions or
    // maybe even their number.

    ArrayRefParameter<
      "::wave::WaveSymbolAttr",
      "Shape of the tensor when fully inferred, missing otherwise">:$shape,
    TypeParameter<"bool", "">:$fully_specified,
    TypeParameter<"::mlir::Type", "Type of the tensor elements">:$element_type,
    DefaultValuedParameter<
        "::wave::WaveAddressSpaceAttr",
        "::wave::WaveAddressSpaceAttr::get("
            "$_ctxt, ::wave::WaveAddressSpace::Unspecified)",
        "Where does the tensor live">:$address_space
  );

  let assemblyFormat =
      "`<` custom<TensorShape>($shape, $fully_specified) `of` $element_type "
      "(`,` $address_space^)? `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Return the rank of the fully-specified tensor.
    size_t getRank() {
      assert(getFullySpecified() &&
             "requested rank of an under-specified tensor");
      return getShape().size();
    }

    // Return the resolved shape of the fully-specified tensor.
    std::optional<llvm::SmallVector<int64_t>> getResolvedShape(wave::WaveHyperparameterAttr hyperparameter) const;

    // Return the address space.
    ::wave::WaveAddressSpace getAddressSpaceValue() const {
      return getAddressSpace().getValue();
    }

    // Return a tensor with the same element type and address space and the
    // shape copied from the other tensor.
    ::wave::WaveTensorType copyShapeFrom(::wave::WaveTensorType other) {
      if (other.getElementType() == getElementType() &&
          other.getAddressSpace() == getAddressSpace())
        return other;

      return ::wave::WaveTensorType::get(
          getContext(), other.getShape(), other.getFullySpecified(),
          getElementType(), getAddressSpace());
    }
  }];
}

//-----------------------------------------------------------------------------
// Predicates and constraints for tensor types with specific address spaces
//-----------------------------------------------------------------------------

def WaveTensorAddressSpaceUnspecified
  : CPred<"::llvm::cast<::wave::WaveTensorType>($_self).getAddressSpaceValue()"
          " == ::wave::WaveAddressSpace::Unspecified">;
def WaveTensorAddressSpaceGlobal
  : CPred<"::llvm::cast<::wave::WaveTensorType>($_self).getAddressSpaceValue()"
          " == ::wave::WaveAddressSpace::Global">;
def WaveTensorAddressSpaceShared
  : CPred<"::llvm::cast<::wave::WaveTensorType>($_self).getAddressSpaceValue()"
          " == ::wave::WaveAddressSpace::Shared">;
def WaveTensorAddressSpaceRegister
  : CPred<"::wave::isaTensorInRegister($_self)">;

def WaveTensorInRegister : TypeConstraint<
  Or<[WaveTensorAddressSpaceRegister,
      And<[WaveTensorType.predicate, WaveTensorAddressSpaceUnspecified]>,
      IsFixedVectorOfRankPred<[1]>]>,
  "Wave tensor in register or a 1D vector",
  "::mlir::Type"
>;

def WaveTensorInMemory : TypeConstraint<
  And<[WaveTensorType.predicate,
       Or<[WaveTensorAddressSpaceUnspecified,
           WaveTensorAddressSpaceGlobal,
           WaveTensorAddressSpaceShared]>]>,
  "Wave tensor in memory",
  WaveTensorType.cppType
>;

#endif // WATER_DIALECT_WAVE_WAVETYPES
