// RUN: water-opt %s

func.func @gemm(%a: !wave.tensor<[@M, @K] of bf16, <shared>>,
               %b: !wave.tensor<[@N, @K] of bf16, <shared>>,
               %c: !wave.tensor<[@M, @N] of f32, <global>>) {

  %0 = arith.constant 0.0 : f32
  %c_reg = wave.register %0 : !wave.tensor<[@M, @N] of f32>

  %mma_result = wave.iterate @K iter_args(%c_reg) {
  ^bb0(%acc: !wave.tensor<[@M, @N] of f32>):

    %a_reg = wave.read %a : (!wave.tensor<[@M, @K] of bf16, <shared>>) -> !wave.tensor<[@M, @K] of bf16>
    %b_reg = wave.read %b : (!wave.tensor<[@N, @K] of bf16, <shared>>) -> !wave.tensor<[@N, @K] of bf16>

    %inner_acc = wave.mma %a_reg, %b_reg, %acc {kind = #wave.mma_kind<f32_32x32x16_bf16>} :
      (!wave.tensor<[@M, @K] of bf16>, !wave.tensor<[@N, @K] of bf16>, !wave.tensor<[@M, @N] of f32>) -> !wave.tensor<[@M, @N] of f32>

    wave.yield %inner_acc : !wave.tensor<[@M, @N] of f32>
  } : (!wave.tensor<[@M, @N] of f32>)-> (!wave.tensor<[@M, @N] of f32>)

  wave.write %mma_result, %c : !wave.tensor<[@M, @N] of f32> , !wave.tensor<[@M, @N] of f32, <global>>

  return
}
