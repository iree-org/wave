// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Utils.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Operation.h"

llvm::LogicalResult
wave::setNormalFormPassPostcondition(wave::WaveNormalForm form,
                                     mlir::Operation *root) {
  llvm::LogicalResult result =
      wave::detail::verifyNormalFormAttr(root, form, /*emitDiagnostics=*/false);
  if (llvm::succeeded(result))
    root->setAttr(wave::WaveDialect::kNormalFormAttrName,
                  wave::WaveNormalFormAttr::get(root->getContext(), form));
  return result;
}

llvm::LogicalResult
wave::verifyNormalFormPassPrecondition(wave::WaveNormalForm form,
                                       mlir::Operation *root,
                                       llvm::StringRef passName) {

  for (mlir::Operation *op = root; op != nullptr; op = op->getParentOp()) {
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
