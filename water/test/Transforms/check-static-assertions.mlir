// RUN: water-opt %s --water-check-static-assertions --verify-diagnostics

func.func @fail() {
  %false = arith.constant false
  // expected-error @below {{assertion known to be false}}
  cf.assert %false, "assertion"
  return
}
