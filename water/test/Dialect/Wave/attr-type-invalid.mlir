// RUN: water-opt %s --allow-unregistered-dialect --water-test-wave-dialect-functions --split-input-file --verify-diagnostics

// expected-error @below {{expected element type to be integer, index or floating point scalar}}
func.func private @unspecified_tensor() -> !wave.tensor<any of !wave.tensor<any of bf16>>

// -----

// expected-error @below {{shape not expected for non-fully specified tensors}}
"wave_test.create_tensor"() {fully_specified = false, shape = [@A, @B]} : () -> ()

// -----

// expected-error @below {{"wave.hyperparameters" expects a WaveHyperparameterAttr}}
module attributes {wave.hyperparameters = 1} {}

// -----

// expected-error @below {{"wave.elements_per_thread" expects an IntegerAttr}}
module attributes {wave.elements_per_thread = "abc"} {}

// -----

// expected-error @below {{unexpected wave dialect attribute "wave.unexpected"}}
module attributes {wave.unexpected = 42} {}

// -----

// expected-error @below {{symbols names starting with '_' are reserved for internal use}}
module attributes {wave_test.symbol = #wave.symbol<"_A">}

// -----

// expected-error @below {{start map should have exactly one result, got 2}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0, s0 + 1)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{step map should have exactly one result, got 2}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<()[s0] -> (s0, s0 + 1)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{stride map should have exactly one result, got 2}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<()[s0] -> (s0, s0 + 1)>
} : () -> ()

// -----

// expected-error @below {{start map should have the same number of symbols as given to the attribute, got 2 symbols for 1 symbols}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0, s1] -> (s0)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{step map should have the same number of symbols as given to the attribute, got 2 symbols for 1 symbols}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<()[s0, s1] -> (s0)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{stride map should have the same number of symbols as given to the attribute, got 2 symbols for 1 symbols}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<()[s0, s1] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{start map should have no dimensions, got 1 dimensions}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<(d0)[s0] -> (s0)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{step map should have no dimensions, got 1 dimensions}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<(d0)[s0] -> (s0)>,
  stride = affine_map<()[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{stride map should have no dimensions, got 1 dimensions}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>],
  start = affine_map<()[s0] -> (s0)>,
  step = affine_map<()[s0] -> (s0)>,
  stride = affine_map<(d0)[s0] -> (s0)>
} : () -> ()

// -----

// expected-error @below {{duplicate symbol: #wave.index_symbol<WG0>}}
"wave_test.create_index_mapping"() {
  symbols = [#wave.index_symbol<WG0>, #wave.index_symbol<WG0>],
  start = affine_map<()[s0, s1] -> (s0)>,
  step = affine_map<()[s0, s1] -> (s0)>,
  stride = affine_map<()[s0, s1] -> (s0)>
} : () -> ()
