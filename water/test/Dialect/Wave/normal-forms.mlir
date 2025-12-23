// RUN: water-opt %s --water-wave-infer-types --water-wave-propagate-elements-per-thread | FileCheck %s

// CHECK: module attributes {wave.normal_form = #wave.normal_form<full_types,memory_only_types>}
module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  func.func @test_multiple_forms_in_sequence(%mem: !wave.tensor<[@M] of f32, <global>>) attributes {wave.hyperparameters = #wave.hyperparameters<{M = 128}>} {
    %0 = arith.constant 0.0 : f32
    %reg = wave.register %0 : !wave.tensor<[@M] of f32, <register>>
    wave.write %reg, %mem { elements_per_thread = 8 } : !wave.tensor<[@M] of f32, <register>>, !wave.tensor<[@M] of f32, <global>>
    return
  }
}
