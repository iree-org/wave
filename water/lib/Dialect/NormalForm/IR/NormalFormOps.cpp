// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/NormalForm/IR/NormalFormOps.h"
#include "water/Dialect/NormalForm/IR/NormalFormDialect.h"
#include "water/Dialect/NormalForm/IR/NormalFormInterfaces.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace normalform;

//-----------------------------------------------------------------------------
// ModuleOp
//-----------------------------------------------------------------------------

void normalform::ModuleOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &state,
                                 NormalFormAttrInterface normalForm,
                                 std::optional<llvm::StringRef> name) {
  state.addRegion()->emplaceBlock();
  state.addAttribute(NormalFormDialect::kNormalFormAttrName, normalForm);
  if (name) {
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));
  }
}

/// Construct a module from the given context.
normalform::ModuleOp
normalform::ModuleOp::create(mlir::Location loc,
                             NormalFormAttrInterface normalForm,
                             std::optional<llvm::StringRef> name) {
  OpBuilder builder(loc->getContext());
  return ModuleOp::create(builder, loc, normalForm, name);
}

llvm::LogicalResult normalform::ModuleOp::verifyRegions() {
  bool emitRemark = false;
  bool emitDiagnostics = true;

  NormalFormAttrInterface normalForm = getNormalFormAttr();
  Operation *op = getOperation();

  if (llvm::failed(normalForm.verifyOperation(getOperation()))) {
    return op->emitError() << "nornmal form verification failed";
  }

  //   if (llvm::failed(normalForm.verifyAttribute(Attribute()))) {
  //     return op->emitError() << "nornmal form verification failed";
  //   }

  //   if (llvm::failed(normalForm.verifyType(Type()))) {
  //     return op->emitError() << "nornmal form verification failed";
  //   }

  return llvm::success();
}
