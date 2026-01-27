// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Ticketing Pass - Insert s_waitcnt instructions for memory synchronization
//
// This pass implements a ticket-based tracking system for VMEM and LGKM
// memory operations. Each memory operation is assigned a monotonically
// increasing ticket. When a value is used, the pass computes the minimum
// wait threshold and inserts s_waitcnt if required.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace waveasm;

namespace {

//===----------------------------------------------------------------------===//
// Ticketing System
//===----------------------------------------------------------------------===//

/// Tracks memory operation tickets for waitcnt insertion
class Ticketing {
public:
  Ticketing() = default;

  /// Issue a new VMEM ticket, returns the ticket number
  /// Also invalidates any previous wait tracking since new loads are pending
  int64_t nextVmemTicket() {
    ++lastVmemTicket;
    // A new load after a previous wait means we need to wait again
    // Reset the wait tracking to reflect this
    lastVmemWait = std::nullopt;
    return lastVmemTicket;
  }

  /// Issue a new LGKM ticket, returns the ticket number
  /// Also invalidates any previous wait tracking since new loads are pending
  int64_t nextLgkmTicket() {
    ++lastLgkmTicket;
    // A new load after a previous wait means we need to wait again
    lastLgkmWait = std::nullopt;
    return lastLgkmTicket;
  }

  /// Get current VMEM ticket (last issued)
  int64_t getLastVmemTicket() const { return lastVmemTicket; }

  /// Get current LGKM ticket (last issued)
  int64_t getLastLgkmTicket() const { return lastLgkmTicket; }

  /// Compute the vmcnt threshold needed to wait for a given ticket.
  /// Returns std::nullopt if no wait is needed (already satisfied by
  /// a previous stricter wait).
  std::optional<int64_t> computeVmemWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastVmemTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastVmemTicket - requiredTicket);

    // Check if a previous wait already covers this
    if (lastVmemWait.has_value() && *lastVmemWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Compute the lgkmcnt threshold needed to wait for a given ticket.
  std::optional<int64_t> computeLgkmWait(int64_t requiredTicket) {
    if (requiredTicket < 0 || lastLgkmTicket < 0)
      return std::nullopt;

    int64_t threshold = std::max(int64_t(0), lastLgkmTicket - requiredTicket);

    if (lastLgkmWait.has_value() && *lastLgkmWait <= threshold)
      return std::nullopt;

    return threshold;
  }

  /// Record that a vmem wait was emitted with the given threshold
  void observeVmemWait(int64_t threshold) {
    if (!lastVmemWait.has_value() || threshold < *lastVmemWait)
      lastVmemWait = threshold;
  }

  /// Record that a lgkm wait was emitted with the given threshold
  void observeLgkmWait(int64_t threshold) {
    if (!lastLgkmWait.has_value() || threshold < *lastLgkmWait)
      lastLgkmWait = threshold;
  }

  /// Reset wait tracking (e.g., at control flow boundaries)
  void resetWaits() {
    lastVmemWait = std::nullopt;
    lastLgkmWait = std::nullopt;
  }

private:
  int64_t lastVmemTicket = -1;
  int64_t lastLgkmTicket = -1;
  std::optional<int64_t> lastVmemWait;
  std::optional<int64_t> lastLgkmWait;
};

//===----------------------------------------------------------------------===//
// Memory Operation Classification
//===----------------------------------------------------------------------===//

enum class MemOpKind {
  None,
  VmemLoad,   // buffer_load, global_load, flat_load
  VmemStore,  // buffer_store, global_store, flat_store
  LgkmLoad,   // ds_read, s_load
  LgkmStore   // ds_write
};

/// Classify an operation by its memory type using MLIR op types
MemOpKind classifyMemOp(Operation *op) {
  // VMEM loads (buffer, global, flat)
  if (isa<BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3,
          BUFFER_LOAD_DWORDX4, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE,
          BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT,
          GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3,
          GLOBAL_LOAD_DWORDX4, GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE,
          GLOBAL_LOAD_USHORT, GLOBAL_LOAD_SSHORT,
          FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX3,
          FLAT_LOAD_DWORDX4,
          // Gather-to-LDS: VMEM load directly to LDS (still uses VMEM path)
          BUFFER_LOAD_DWORD_LDS, BUFFER_LOAD_DWORDX4_LDS>(op))
    return MemOpKind::VmemLoad;

  // VMEM stores (buffer, global, flat)
  if (isa<BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX3,
          BUFFER_STORE_DWORDX4, BUFFER_STORE_BYTE, BUFFER_STORE_SHORT,
          GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX3,
          GLOBAL_STORE_DWORDX4, GLOBAL_STORE_BYTE, GLOBAL_STORE_SHORT,
          FLAT_STORE_DWORD, FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX3,
          FLAT_STORE_DWORDX4>(op))
    return MemOpKind::VmemStore;

  // LDS loads (ds_read)
  if (isa<DS_READ_B32, DS_READ_B64, DS_READ_B128,
          DS_READ2_B32, DS_READ2_B64,
          DS_READ_U8, DS_READ_I8, DS_READ_U16, DS_READ_I16>(op))
    return MemOpKind::LgkmLoad;

  // LDS stores (ds_write)
  if (isa<DS_WRITE_B32, DS_WRITE_B64, DS_WRITE_B128,
          DS_WRITE2_B32, DS_WRITE2_B64,
          DS_WRITE_B8, DS_WRITE_B16>(op))
    return MemOpKind::LgkmStore;

  // Scalar loads (s_load) - also use LGKM counter
  if (isa<S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4,
          S_LOAD_DWORDX8, S_LOAD_DWORDX16>(op))
    return MemOpKind::LgkmLoad;

  return MemOpKind::None;
}

//===----------------------------------------------------------------------===//
// InsertWaitcnt Pass
//===----------------------------------------------------------------------===//

struct InsertWaitcntPass
    : public PassWrapper<InsertWaitcntPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertWaitcntPass)

  InsertWaitcntPass() = default;
  InsertWaitcntPass(bool insertAfterLoads)
      : insertAfterLoads(insertAfterLoads) {}

  StringRef getArgument() const override { return "waveasm-insert-waitcnt"; }

  StringRef getDescription() const override {
    return "Insert s_waitcnt instructions for memory synchronization";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each program
    module.walk([&](ProgramOp program) { processProgram(program); });
  }

private:
  bool insertAfterLoads = false;

  // Statistics
  unsigned numVmemOps = 0;
  unsigned numLgkmOps = 0;
  unsigned numWaitcntInserted = 0;

  void processProgram(ProgramOp program) {
    Ticketing ticketing;

    // Map from Value to the ticket that produces it
    llvm::DenseMap<Value, std::pair<MemOpKind, int64_t>> valueTickets;

    // Walk through operations in order
    program.walk([&](Operation *op) {
      // Skip the program op itself
      if (isa<ProgramOp>(op))
        return;

      // Check if this is a memory operation
      MemOpKind kind = classifyMemOp(op);

      if (kind != MemOpKind::None) {
        // For stores, check operands BEFORE incrementing ticket counter
        // This ensures correct wait thresholds are computed
        if (kind == MemOpKind::VmemStore || kind == MemOpKind::LgkmStore) {
          std::optional<int64_t> neededVmcnt;
          std::optional<int64_t> neededLgkmcnt;

          for (Value operand : op->getOperands()) {
            auto it = valueTickets.find(operand);
            if (it == valueTickets.end())
              continue;

            auto [opKind, opTicket] = it->second;

            if (opKind == MemOpKind::VmemLoad) {
              auto wait = ticketing.computeVmemWait(opTicket);
              if (wait.has_value()) {
                if (!neededVmcnt.has_value() || *wait < *neededVmcnt)
                  neededVmcnt = wait;
              }
            } else if (opKind == MemOpKind::LgkmLoad) {
              auto wait = ticketing.computeLgkmWait(opTicket);
              if (wait.has_value()) {
                if (!neededLgkmcnt.has_value() || *wait < *neededLgkmcnt)
                  neededLgkmcnt = wait;
              }
            }
          }

          // Insert waitcnt before the store if needed
          if (neededVmcnt.has_value() || neededLgkmcnt.has_value()) {
            OpBuilder builder(op->getContext());
            builder.setInsertionPoint(op);

            if (neededVmcnt.has_value() && neededLgkmcnt.has_value()) {
              builder.create<S_WAITCNT>(
                  op->getLoc(),
                  builder.getI32IntegerAttr(*neededVmcnt),
                  builder.getI32IntegerAttr(*neededLgkmcnt),
                  IntegerAttr());
              ticketing.observeVmemWait(*neededVmcnt);
              ticketing.observeLgkmWait(*neededLgkmcnt);
            } else if (neededVmcnt.has_value()) {
              builder.create<S_WAITCNT_VMCNT>(op->getLoc(),
                                             builder.getI32IntegerAttr(*neededVmcnt));
              ticketing.observeVmemWait(*neededVmcnt);
            } else {
              builder.create<S_WAITCNT_LGKMCNT>(op->getLoc(),
                                               builder.getI32IntegerAttr(*neededLgkmcnt));
              ticketing.observeLgkmWait(*neededLgkmcnt);
            }
            numWaitcntInserted++;
          }
        }

        // Now increment the ticket counter for this memory op
        // NOTE: Only LOADS increment the counter because vmcnt/lgkmcnt only
        // track outstanding loads, not stores. Stores don't affect these counters.
        int64_t ticket = -1;
        if (kind == MemOpKind::VmemLoad) {
          ticket = ticketing.nextVmemTicket();
          numVmemOps++;
        } else if (kind == MemOpKind::LgkmLoad) {
          ticket = ticketing.nextLgkmTicket();
          numLgkmOps++;
        } else {
          // Stores don't get tickets (they don't affect vmcnt/lgkmcnt)
          numVmemOps++;
        }

        // Map results to tickets (for loads)
        for (Value result : op->getResults()) {
          valueTickets[result] = {kind, ticket};
        }

        // For conservative mode, insert wait immediately after loads
        if (insertAfterLoads &&
            (kind == MemOpKind::VmemLoad || kind == MemOpKind::LgkmLoad)) {
          OpBuilder builder(op->getContext());
          builder.setInsertionPointAfter(op);

          if (kind == MemOpKind::VmemLoad) {
            builder.create<S_WAITCNT_VMCNT>(op->getLoc(),
                                           builder.getI32IntegerAttr(0));
            ticketing.observeVmemWait(0);
          } else {
            builder.create<S_WAITCNT_LGKMCNT>(op->getLoc(),
                                             builder.getI32IntegerAttr(0));
            ticketing.observeLgkmWait(0);
          }
          numWaitcntInserted++;
        }
        return;
      }

      // For non-memory operations, check if any operands require waits
      std::optional<int64_t> neededVmcnt;
      std::optional<int64_t> neededLgkmcnt;

      for (Value operand : op->getOperands()) {
        auto it = valueTickets.find(operand);
        if (it == valueTickets.end())
          continue;

        auto [opKind, ticket] = it->second;

        if (opKind == MemOpKind::VmemLoad) {
          auto wait = ticketing.computeVmemWait(ticket);
          if (wait.has_value()) {
            if (!neededVmcnt.has_value() || *wait < *neededVmcnt)
              neededVmcnt = wait;
          }
        } else if (opKind == MemOpKind::LgkmLoad) {
          auto wait = ticketing.computeLgkmWait(ticket);
          if (wait.has_value()) {
            if (!neededLgkmcnt.has_value() || *wait < *neededLgkmcnt)
              neededLgkmcnt = wait;
          }
        }
      }

      // Insert waitcnt if needed
      if (neededVmcnt.has_value() || neededLgkmcnt.has_value()) {
        OpBuilder builder(op->getContext());
        builder.setInsertionPoint(op);

        // Use combined s_waitcnt if both are needed
        if (neededVmcnt.has_value() && neededLgkmcnt.has_value()) {
          builder.create<S_WAITCNT>(
              op->getLoc(),
              builder.getI32IntegerAttr(*neededVmcnt),
              builder.getI32IntegerAttr(*neededLgkmcnt),
              IntegerAttr());
          ticketing.observeVmemWait(*neededVmcnt);
          ticketing.observeLgkmWait(*neededLgkmcnt);
        } else if (neededVmcnt.has_value()) {
          builder.create<S_WAITCNT_VMCNT>(op->getLoc(),
                                         builder.getI32IntegerAttr(*neededVmcnt));
          ticketing.observeVmemWait(*neededVmcnt);
        } else {
          builder.create<S_WAITCNT_LGKMCNT>(op->getLoc(),
                                           builder.getI32IntegerAttr(*neededLgkmcnt));
          ticketing.observeLgkmWait(*neededLgkmcnt);
        }
        numWaitcntInserted++;
      }

      // Reset waits at control flow boundaries (labels, branches)
      if (isa<LabelOp>(op) || op->getName().getStringRef().contains("branch")) {
        ticketing.resetWaits();
      }
    });
  }
};

} // namespace

namespace waveasm {

std::unique_ptr<mlir::Pass> createWAVEASMInsertWaitcntPass() {
  return std::make_unique<InsertWaitcntPass>();
}

std::unique_ptr<mlir::Pass> createWAVEASMInsertWaitcntPass(bool insertAfterLoads) {
  return std::make_unique<InsertWaitcntPass>(insertAfterLoads);
}

} // namespace waveasm
