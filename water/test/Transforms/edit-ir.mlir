// Test --water-edit-ir-{before,after}[-all] instrumentation.
//
// Each edit-IR stop reads one line from stdin; pressing Enter (or a piped
// newline) continues, 'q' or EOF aborts. The -all tests pipe two newlines
// because the pipeline has two passes.

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
// RUN: printf '\n\n' | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-after-all -o /dev/null \
// RUN:   | FileCheck -check-prefix=AFTER_ALL %s

// AFTER_ALL-NOT: === water-edit-ir before
// AFTER_ALL:     === water-edit-ir after canonicalize ===
// AFTER_ALL:     === water-edit-ir after cse ===

// --- 4. --water-edit-ir-before-all fires for every pass --------------------
// RUN: printf '\n\n' | water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-before-all -o /dev/null \
// RUN:   | FileCheck -check-prefix=BEFORE_ALL %s

// BEFORE_ALL-NOT: === water-edit-ir after
// BEFORE_ALL:     === water-edit-ir before canonicalize ===
// BEFORE_ALL:     === water-edit-ir before cse ===

// --- 5. Abort: 'q' at the first stop aborts the rest of the pipeline ------
// RUN: echo "q" | not water-opt %s -mlir-disable-threading=true \
// RUN:   -pass-pipeline='builtin.module(canonicalize,cse)' \
// RUN:   --water-edit-ir-after-all -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=ABORT %s

// ABORT:     === water-edit-ir after canonicalize ===
// ABORT-NOT: === water-edit-ir after cse ===

// --- 6. Round-trip: pipeline result is identical with and without edit -----
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
