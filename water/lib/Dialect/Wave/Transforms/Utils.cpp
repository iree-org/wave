// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Utils.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Operation.h"
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
  wave::WaveNormalForm finalForm = form;

  if (preserve) {
    // Get current normal form and combine with new form.
    if (auto attr = root->getAttrOfType<wave::WaveNormalFormAttr>(
            wave::WaveDialect::kNormalFormAttrName)) {
      wave::WaveNormalForm currentForm = attr.getValue();
      finalForm = currentForm | form;
    }
  }

  if (llvm::failed(
          wave::detail::verifyNormalFormAttr(root, finalForm,
                                             /*emitDiagnostics=*/false))) {
    return emitError(root->getLoc())
           << "failed to produce code with the '" << wave::stringifyEnum(form)
           << "' normal form";
  }

  root->setAttr(wave::WaveDialect::kNormalFormAttrName,
                wave::WaveNormalFormAttr::get(root->getContext(), finalForm));

  return llvm::success();
}

llvm::LogicalResult wave::clearNormalFormPassPostcondition(Operation *root) {
  // Actually remove the normal form attribute instead of setting it to None
  root->removeAttr(wave::WaveDialect::kNormalFormAttrName);
  return llvm::success();
}

llvm::LogicalResult wave::verifyNormalFormPassPrecondition(
    wave::WaveNormalForm form, Operation *root, llvm::StringRef passName) {

  for (Operation *op = root; op != nullptr; op = op->getParentOp()) {
    auto attr = op->getAttrOfType<wave::WaveNormalFormAttr>(
        wave::WaveDialect::kNormalFormAttrName);
    if (!attr)
      continue;
    if (wave::bitEnumContainsAll(attr.getValue(), form))
      return llvm::success();
  }
  return root->emitError()
         << passName
         << " pass expects the root operation or its ancestor to guarantee the "
         << wave::stringifyEnum(form) << " normal form";
}
