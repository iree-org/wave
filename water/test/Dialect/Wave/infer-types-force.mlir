// RUN: water-opt %s --water-wave-infer-types='force=1' --split-input-file --verify-diagnostics

module attributes {wave.normal_form = #wave.normal_form<full_func_boundary>} {
  // expected-error @below {{type conflict was detected for argument #0 of block #0 in region #0}}
  func.func @iterate_conflict(%arg0: !wave.tensor<[@A] of f32>) {
    // expected-error @below {{type conflict was detected for result #0}}
    %0 = wave.iterate @I iter_args(%arg0) {
    // expected-error @below {{type conflict was detected for argument #0 of block #0 in region #0}}
    ^bb0(%arg1: !wave.tensor<[@A] of f32>):
      wave.yield %arg1 : !wave.tensor<[@A] of f32>
    } : (!wave.tensor<[@A] of f32>) -> (!wave.tensor<any of f32>)
    // expected-error @below {{type conflict was detected for result #0}}
    wave.iterate @I iter_args(%0) {
    // expected-error @below {{type conflict was detected for argument #0 of block #0 in region #0}}
    ^bb0(%arg1: !wave.tensor<any of f32>):
      wave.yield %arg1 : !wave.tensor<any of f32>
    } : (!wave.tensor<any of f32>) -> (!wave.tensor<[@B] of f32>)
    return
  }
}
