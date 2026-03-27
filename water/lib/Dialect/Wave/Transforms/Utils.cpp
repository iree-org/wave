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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

llvm::LogicalResult
wave::setNormalFormPassPostcondition(ArrayRef<wave::WaveNormalForm> forms,
                                     Operation *root, bool preserve) {
  auto module = llvm::dyn_cast<normalform::ModuleOp>(root);
  if (!module)
    return root->emitError() << "expected normalform.module";

  llvm::DenseSet<wave::WaveNormalForm> finalForms(forms.begin(), forms.end());

  auto normalforms =
      module.getNormalForms().getAsRange<normalform::NormalFormAttrInterface>();

  SmallVector<normalform::NormalFormAttrInterface> waveNormalForms =
      llvm::filter_to_vector(normalforms,
                             llvm::IsaPred<wave::WaveNormalFormAttr>);

  if (preserve) {
    for (auto nf : waveNormalForms)
      finalForms.insert(cast<WaveNormalFormAttr>(nf).getValue());
  }

  if (!waveNormalForms.empty())
    module.removeNormalForms(waveNormalForms);

  SmallVector<wave::WaveNormalForm> sortedForms(finalForms.begin(),
                                                finalForms.end());
  llvm::sort(sortedForms, [](wave::WaveNormalForm a, wave::WaveNormalForm b) {
    return static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
  });

  SmallVector<normalform::NormalFormAttrInterface> newAttrs;
  newAttrs.reserve(sortedForms.size());

  for (wave::WaveNormalForm form : sortedForms)
    newAttrs.push_back(wave::WaveNormalFormAttr::get(root->getContext(), form));

  module.addNormalForms(newAttrs);

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

llvm::LogicalResult wave::verifyNormalFormPassPrecondition(
    ArrayRef<WaveNormalForm> forms, Operation *root, llvm::StringRef passName) {
  auto module = llvm::dyn_cast<normalform::ModuleOp>(root);
  if (!module)
    return root->emitError()
           << "expected << " << normalform::ModuleOp::getOperationName();

  ArrayRef<Attribute> normalforms = module.getNormalForms().getValue();
  llvm::DenseSet<WaveNormalForm> presentForms;
  for (Attribute attr :
       llvm::make_filter_range(normalforms, llvm::IsaPred<WaveNormalFormAttr>))
    presentForms.insert(cast<WaveNormalFormAttr>(attr).getValue());

  for (WaveNormalForm form : forms) {
    if (!presentForms.contains(form)) {
      return root->emitError()
             << passName
             << " pass expects the root operation or its ancestor to "
                "guarantee the "
             << wave::stringifyWaveNormalForm(form) << " normal form";
    }
  }

  return llvm::success();
}
