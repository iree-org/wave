// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace normalform;

#define GET_OP_CLASSES
#include "water/Dialect/NormalForm/IR/NormalFormOps.cpp.inc"

//-----------------------------------------------------------------------------
// ModuleOp
//-----------------------------------------------------------------------------

void normalform::ModuleOp::build(OpBuilder &builder, OperationState &state,
                                 ArrayRef<NormalFormAttrInterface> normalForms,
                                 std::optional<llvm::StringRef> name) {
  state.addRegion()->emplaceBlock();
  ArrayRef<Attribute> attributeArray =
      ArrayRef<Attribute>(normalForms.begin(), normalForms.end());
  ArrayAttr normalFormsArray = builder.getArrayAttr(attributeArray);
  // Use the tablegen-generated attribute name "normal_forms".
  state.addAttribute(getNormalFormsAttrName(state.name), normalFormsArray);
  if (name) {
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));
  }
}

/// Construct a module from the given context.
normalform::ModuleOp
normalform::ModuleOp::create(Location loc,
                             ArrayRef<NormalFormAttrInterface> normalForms,
                             std::optional<StringRef> name) {
  OpBuilder builder(loc->getContext());
  return ModuleOp::create(builder, loc, normalForms, name);
}

LogicalResult
normalform::ModuleOp::verifyNormalForm(NormalFormAttrInterface normalForm,
                                       bool emitDiagnostics) {
  Operation *root = getOperation();

  SmallPtrSet<Type, 16> seenTypes;
  SmallPtrSet<Attribute, 16> seenAttrs;
  Location loc = root->getLoc();
  AttrTypeWalker walker;

  auto emitLocError = [&]() {
    InFlightDiagnostic diag = mlir::emitError(loc);
    if (!emitDiagnostics)
      diag.abandon();

    return diag;
  };

  auto visitType = [&](Type type) {
    auto [it, inserted] = seenTypes.insert(type);
    if (!inserted)
      return WalkResult::skip();

    if (llvm::failed(normalForm.verifyType(emitLocError, type)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  };

  auto visitAttr = [&](Attribute attr) {
    auto [it, inserted] = seenAttrs.insert(attr);
    if (!inserted)
      return WalkResult::skip();

    if (llvm::failed(normalForm.verifyAttribute(emitLocError, attr)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  };

  walker.addWalk(visitType);
  walker.addWalk(visitAttr);

  auto visitOp = [&](Operation *op) {
    loc = op->getLoc();

    // TODO: skip when we reach another normalform.module which has normalform
    // in its attributes
    if (llvm::failed(normalForm.verifyOperation(emitLocError, op)))
      return WalkResult::interrupt();

    for (OpResult result : op->getResults()) {
      loc = result.getLoc();
      WalkResult walkResult = walker.walk(result.getType());
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();
    }

    for (mlir::Region &region : op->getRegions()) {
      for (mlir::Block &block : region) {
        for (mlir::BlockArgument arg : block.getArguments()) {
          loc = arg.getLoc();
          WalkResult walkResult = walker.walk(arg.getType());
          if (walkResult.wasInterrupted())
            return WalkResult::interrupt();
        }
      }
    }

    for (NamedAttribute attr : op->getAttrs()) {
      WalkResult walkResult = walker.walk(attr.getValue());
      if (walkResult.wasInterrupted())
        return WalkResult::interrupt();
    }

    return WalkResult::advance();
  };

  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(visitOp);

  return llvm::failure(walkResult.wasInterrupted());
}

bool normalform::ModuleOp::inferNormalForms(
    ArrayRef<NormalFormAttrInterface> normalForms) {
  ArrayRef<Attribute> currentNormalForms = getNormalFormsAttr().getValue();
  SetVector<Attribute> normalFormSet;
  normalFormSet.insert_range(currentNormalForms);

  bool changed = false;
  for (NormalFormAttrInterface nf : normalForms) {
    if (normalFormSet.contains(nf))
      continue;

    if (llvm::succeeded(verifyNormalForm(nf, /*emitDiagnostics*/ false))) {
      normalFormSet.insert(nf);
      changed = true;
    }
  }

  if (!changed)
    return false;

  OpBuilder builder(getContext());
  ArrayAttr newNormalFormsAttr =
      builder.getArrayAttr(normalFormSet.getArrayRef());
  setNormalFormsAttr(newNormalFormsAttr);
  return true;
}

bool normalform::ModuleOp::addNormalForms(
    ArrayRef<NormalFormAttrInterface> normalForms) {
  if (normalForms.empty())
    return false;

  ArrayRef<Attribute> currentNormalForms = getNormalFormsAttr().getValue();
  SetVector<Attribute> normalFormSet;
  normalFormSet.insert_range(currentNormalForms);

  bool changed = false;
  for (NormalFormAttrInterface nf : normalForms)
    changed |= normalFormSet.insert(nf);

  if (!changed)
    return false;

  OpBuilder builder(getContext());
  ArrayAttr newNormalFormsAttr =
      builder.getArrayAttr(normalFormSet.getArrayRef());
  setNormalFormsAttr(newNormalFormsAttr);
  return true;
}

bool normalform::ModuleOp::removeNormalForms(
    ArrayRef<NormalFormAttrInterface> normalForms) {
  if (normalForms.empty())
    return false;

  ArrayRef<Attribute> currentNormalForms = getNormalFormsAttr().getValue();
  SetVector<Attribute> normalFormSet;
  normalFormSet.insert_range(currentNormalForms);

  bool changed = false;
  for (NormalFormAttrInterface nf : normalForms)
    changed |= normalFormSet.remove(nf);

  if (!changed)
    return false;

  OpBuilder builder(getContext());
  ArrayAttr newNormalFormsAttr =
      builder.getArrayAttr(normalFormSet.getArrayRef());
  setNormalFormsAttr(newNormalFormsAttr);
  return true;
}

LogicalResult normalform::ModuleOp::verify() {
  // Verify that normal form attributes are unique (set semantics).
  ArrayAttr normalForms = getNormalFormsAttr();
  SmallPtrSet<Attribute, 4> seenForms;
  for (Attribute attr : normalForms) {
    auto [it, inserted] = seenForms.insert(attr);
    if (!inserted)
      return emitOpError() << "contains duplicate normal form attribute: "
                           << attr;
  }
  return llvm::success();
}

LogicalResult normalform::ModuleOp::verifyRegions() {
  ArrayRef<Attribute> normalFormsAttrs = getNormalForms().getValue();
  auto normalFormRange =
      llvm::map_range(normalFormsAttrs, llvm::CastTo<NormalFormAttrInterface>);

  for (NormalFormAttrInterface normalForm : normalFormRange) {
    if (llvm::failed(verifyNormalForm(normalForm, /*emitDiagnostics*/ true))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}
