// Test --water-edit-ir-{before,after}[-all] instrumentation.
//
// When stdin is empty / EOF, the interactive prompt returns immediately and
// the IR round-trips through write-to-file / re-parse / replace unchanged.

// --- 1. --water-edit-ir-after fires only for the named pass ----------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-after=canonicalize -o /dev/null \
// RUN:   | FileCheck -check-prefix=AFTER %s

// AFTER-NOT: === water-edit-ir before
// AFTER:     === water-edit-ir after canonicalize ===
// AFTER-NOT: === water-edit-ir after cse ===

// --- 2. --water-edit-ir-before fires only for the named pass ---------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-before=cse -o /dev/null \
// RUN:   | FileCheck -check-prefix=BEFORE %s

// BEFORE-NOT: === water-edit-ir before canonicalize ===
// BEFORE:     === water-edit-ir before cse ===
// BEFORE-NOT: === water-edit-ir after

// --- 3. --water-edit-ir-after-all fires for every pass ---------------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-after-all -o /dev/null \
// RUN:   | FileCheck -check-prefix=AFTER_ALL %s

// AFTER_ALL-NOT: === water-edit-ir before
// AFTER_ALL:     === water-edit-ir after canonicalize ===
// AFTER_ALL:     === water-edit-ir after cse ===

// --- 4. --water-edit-ir-before-all fires for every pass --------------------
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-before-all -o /dev/null \
// RUN:   | FileCheck -check-prefix=BEFORE_ALL %s

// BEFORE_ALL-NOT: === water-edit-ir after
// BEFORE_ALL:     === water-edit-ir before canonicalize ===
// BEFORE_ALL:     === water-edit-ir before cse ===

// --- 5. Round-trip: pipeline result is identical with and without edit -----
// RUN: echo "" | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-after=canonicalize \
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
