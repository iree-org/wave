// Test --mlir-edit-ir-{before,after}[-all] instrumentation.
//
// When stdin is empty / EOF, the interactive prompt returns immediately and
// the IR round-trips through write-to-file / re-parse / replace unchanged.

// --- 1. --mlir-edit-ir-after fires only for the named pass ----------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --mlir-edit-ir-after=canonicalize -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=AFTER %s

// AFTER-NOT: === mlir-edit-ir before
// AFTER:     === mlir-edit-ir after canonicalize ===
// AFTER-NOT: === mlir-edit-ir after cse ===

// --- 2. --mlir-edit-ir-before fires only for the named pass ---------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --mlir-edit-ir-before=cse -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=BEFORE %s

// BEFORE-NOT: === mlir-edit-ir before canonicalize ===
// BEFORE:     === mlir-edit-ir before cse ===
// BEFORE-NOT: === mlir-edit-ir after

// --- 3. --mlir-edit-ir-after-all fires for every pass ---------------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --mlir-edit-ir-after-all -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=AFTER_ALL %s

// AFTER_ALL-NOT: === mlir-edit-ir before
// AFTER_ALL:     === mlir-edit-ir after canonicalize ===
// AFTER_ALL:     === mlir-edit-ir after cse ===

// --- 4. --mlir-edit-ir-before-all fires for every pass --------------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --mlir-edit-ir-before-all -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=BEFORE_ALL %s

// BEFORE_ALL-NOT: === mlir-edit-ir after
// BEFORE_ALL:     === mlir-edit-ir before canonicalize ===
// BEFORE_ALL:     === mlir-edit-ir before cse ===

// --- 5. Round-trip: pipeline result is identical with and without edit -----
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --mlir-edit-ir-after=canonicalize \
// RUN:   | FileCheck -check-prefix=ROUNDTRIP %s

// ROUNDTRIP-LABEL: func.func @foo
// ROUNDTRIP-NEXT:    return %arg0 : i32

module {
  func.func @foo(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %add = arith.addi %arg0, %c0 : i32
    return %add : i32
  }
}
