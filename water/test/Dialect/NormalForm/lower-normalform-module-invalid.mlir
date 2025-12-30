// RUN: water-opt %s -lower-normalform-module --split-input-file --verify-diagnostics

//-----------------------------------------------------------------------------
// Test that multiple top-level normalform.module operations are rejected.
//-----------------------------------------------------------------------------

// expected-error @below {{expected at most one top-level normalform.module, found 2}}
module {
  normalform.module [] {
    func.func @foo() {
      return
    }
  }
  normalform.module [] {
    func.func @bar() {
      return
    }
  }
}
