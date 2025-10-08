// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Analysis/InUseForSpeculation.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::water;

LogicalResult InUseAnalysis::initialize(Operation *top) {
  if (failed(dataflow::AbstractSparseBackwardDataFlowAnalysis::initialize(top)))
    return failure();

  if (getSolverConfig().isInterprocedural())
    return success();

  top->walk([this](Operation *op) {
    if (!op->hasTrait<OpTrait::ReturnLike>())
      return;
    if (!isa<FunctionOpInterface>(op->getParentOp()))
      return;
    for (Value operand : op->getOperands()) {
      InUseLattice *lattice = getLatticeElement(operand);
      lattice->getValue().unsafeSet(InUseLatticeState::Unused);
      addDependency(lattice, getProgramPointAfter(op));
      propagateIfChanged(lattice, ChangeResult::Change);
    }
  });
  return success();
}

LogicalResult InUseAnalysis::visit(ProgramPoint *point) {
  bool needsSpecialHandling = [&] {
    if (point->isBlockStart())
      return false;
    Operation *op = point->getPrevOp();
    if (!op->hasTrait<OpTrait::ReturnLike>())
      return false;
    return isa_and_nonnull<FunctionOpInterface>(op->getParentOp());
  }();
  if (!needsSpecialHandling || getSolverConfig().isInterprocedural())
    return dataflow::AbstractSparseBackwardDataFlowAnalysis::visit(point);

  for (Value value : point->getPrevOp()->getOperands()) {
    InUseLattice *lattice = getLatticeElement(value);
    addDependency(lattice, point);
    propagateIfChanged(lattice, lattice->join(InUseLatticeState::Unused));
  }
  return success();
}

LogicalResult
InUseAnalysis::visitOperation(Operation *op, ArrayRef<InUseLattice *> operands,
                              ArrayRef<const InUseLattice *> results) {

  // Mark operands coming from a specific ODS group as used/unused. This is
  // hacky, but avoids introducing an interface.
  auto markODSGroupsAs = [&](auto op, ArrayRef<unsigned> groups,
                             InUseLatticeState state) {
    for (unsigned group : groups) {
      auto [firstIndex, indexLength] = op.getODSOperandIndexAndLength(group);
      for (unsigned i = 0; i < indexLength; ++i) {
        InUseLattice *lattice = operands[firstIndex + i];
        propagateIfChanged(lattice, lattice->join(state));
      }
    }
  };

  llvm::TypeSwitch<Operation *>(op)
      // Both load base and indices are used.
      .Case<memref::LoadOp, vector::LoadOp>([&](auto loadOp) {
        markODSGroupsAs(loadOp, {0, 1}, InUseLatticeState::Used);
      })
      // Stored values are considered unused, the rest is used.
      .Case<memref::StoreOp, vector::StoreOp>([&](auto storeOp) {
        markODSGroupsAs(storeOp, {0}, InUseLatticeState::Unused);
        markODSGroupsAs(storeOp, {1, 2}, InUseLatticeState::Used);
      })
      .Default([&](Operation *op) {
        // Be conservative for everything non-pure.
        if (!isPure(op)) {
          for (InUseLattice *operandLattice : operands) {
            propagateIfChanged(operandLattice, operandLattice->join(
                                                   InUseLatticeStorage::top()));
          }
          return;
        }
        // Pure operations just propagate.
        for (InUseLattice *operandLattice : operands) {
          ChangeResult change = ChangeResult::NoChange;
          for (const InUseLattice *resultLattice : results) {
            change |= operandLattice->join(*resultLattice);
          }
          propagateIfChanged(operandLattice, change);
        }
      });
  return success();
}

void InUseAnalysis::visitBranchOperand(OpOperand &opOperand) {
  InUseLattice *lattice = getLatticeElement(opOperand.get());
  // Non-forwarded operands of if's and for's are used in their control flow.
  llvm::TypeSwitch<Operation *>(opOperand.getOwner())
      .Case<scf::IfOp, scf::ForOp>([&](auto) {
        propagateIfChanged(lattice, lattice->join(InUseLatticeState::Used));
      })
      .Default([&](Operation *) {
        propagateIfChanged(lattice, lattice->join(InUseLatticeStorage::top()));
      });
}
