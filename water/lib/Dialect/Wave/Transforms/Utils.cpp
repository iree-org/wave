// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Utils.h"
#include "mlir/IR/Attributes.h"
#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"
#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

wave::WaveHyperparameterAttr wave::getHyperparameters(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto hyperparams = current->getAttrOfType<WaveHyperparameterAttr>(
            WaveDialect::kHyperparameterAttrName))
      return hyperparams;
  }
  return nullptr;
}

llvm::LogicalResult wave::collectWaveConstraints(
    Operation *top, llvm::DenseMap<Operation *, Attribute> &constraints) {
  auto *waveDialect = top->getContext()->getLoadedDialect<wave::WaveDialect>();
  auto walkResult = top->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(
            wave::WaveDialect::kWaveConstraintsAttrName)) {
      constraints[op] = attr;
      return WalkResult::skip();
    }
    if (op->getDialect() == waveDialect) {
      op->emitError()
          << "wave dialect operation without constraints on an ancestor";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return llvm::failure();
  return llvm::success();
}

llvm::LogicalResult
wave::setNormalFormPassPostcondition(wave::WaveNormalForm form, Operation *root,
                                     bool preserve) {
  auto module = llvm::dyn_cast<normalform::ModuleOp>(root);
  if (!module)
    return root->emitError() << "expected normalform.module";

  wave::WaveNormalForm finalForm = form;
  auto normalforms =
      module.getNormalForms().getAsRange<normalform::NormalFormAttrInterface>();

  SmallVector<normalform::NormalFormAttrInterface> waveNormalForms =
      llvm::filter_to_vector(normalforms,
                             llvm::IsaPred<wave::WaveNormalFormAttr>);

  if (preserve) {
    // Merge all existing normal forms with the new form.
    for (auto nf : waveNormalForms) {
      wave::WaveNormalForm currentForm =
          cast<WaveNormalFormAttr>(nf).getValue();
      finalForm = finalForm | currentForm;
    }
  }

  if (!waveNormalForms.empty())
    module.removeNormalForms(waveNormalForms);

  module.addNormalForms(
      {wave::WaveNormalFormAttr::get(root->getContext(), finalForm)});

  // We rely on the pass manager to call verifyRegion on the normalform.module
  // after the pass

  return llvm::success();
}

llvm::LogicalResult wave::clearNormalFormPassPostcondition(Operation *root) {
  auto module = llvm::dyn_cast<normalform::ModuleOp>(root);
  if (!module)
    return root->emitError()
           << "expected << " << normalform::ModuleOp::getOperationName();

  auto normalforms =
      module.getNormalForms().getAsRange<normalform::NormalFormAttrInterface>();

  SmallVector<normalform::NormalFormAttrInterface> toRemove =
      llvm::filter_to_vector(normalforms, llvm::IsaPred<WaveNormalFormAttr>);

  module.removeNormalForms(toRemove);

  return llvm::success();
}

llvm::LogicalResult
wave::verifyNormalFormPassPrecondition(WaveNormalForm form, Operation *root,
                                       llvm::StringRef passName) {
  auto module = llvm::dyn_cast<normalform::ModuleOp>(root);
  if (!module)
    return root->emitError()
           << "expected << " << normalform::ModuleOp::getOperationName();

  ArrayRef<Attribute> normalforms = module.getNormalForms().getValue();
  WaveNormalForm expectedForm = WaveNormalForm::None;
  for (Attribute form : llvm::make_filter_range(
           normalforms, llvm::IsaPred<WaveNormalFormAttr>)) {
    expectedForm |= cast<WaveNormalFormAttr>(form).getValue();
  }

  if (wave::bitEnumContainsAll(expectedForm, form))
    return llvm::success();

  return root->emitError()
         << passName
         << " pass expects the root operation or its ancestor to guarantee the "
         << wave::stringifyEnum(form) << " normal form";
}
