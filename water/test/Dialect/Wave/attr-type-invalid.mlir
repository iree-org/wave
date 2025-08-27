// RUN: water-opt %s --allow-unregistered-dialect --water-test-wave-dialect-constructors --split-input-file --verify-diagnostics

// expected-error @below {{expected element type to be integer, index or floating point scalar}}
func.func private @unspecified_tensor() -> !wave.tensor<any of !wave.tensor<any of bf16>>

// -----

// expected-error @below {{shape not expected for non-fully specified tensors}}
"wave_test.create_tensor"() {fully_specified = false, shape = [@A, @B]} : () -> ()
