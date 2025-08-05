# RUN: python %s | FileCheck %s

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.templates.speculative_decoding import (
    get_speculative_sampling_kernel,
)
from wave_lang.kernel.wave.utils.general_utils import (
    run_test,
)


@run_test
def test_speculative_decoding():
    # Get the kernel and its hyperparameters
    kernel, hyperparams, _ = get_speculative_sampling_kernel(
        batch_size=10,
        num_speculative_tokens=3,
        threshold_single=0.01,
        threshold_acc=0.01,
        num_draft_tokens=6,
        vocab_size=64,
        seq_len=12,
    )

    # Create the kernel with the hyperparameters
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, kernel)
    print(kernel.asm)

    # CHECK: builtin.module {
    # CHECK-LABEL: func.func @speculative_sampling(

    # Check the function args
    # CHECK-SAME: %arg0: !stream.binding
    # CHECK-SAME: %arg3: !stream.binding
    # CHECK-SAME: %arg13: i32

    # CHECK: arith.constant dense<63> : vector<1xi32>

    # --- Typical operations (representative):
    # CHECK: gpu.block_id  y upper_bound 10
    # CHECK: gpu.thread_id  x upper_bound 64

    # CHECK: vector.load
    # CHECK: vector.store
    # CHECK: scf.while
    # CHECK: vector.extract
    # CHECK: scf.condition
    # CHECK: scf.while
    # CHECK: ^bb0
    # CHECK: arith.divf
    # CHECK: arith.cmpf
    # CHECK: arith.ori
    # CHECK: arith.xori
    # CHECK: vector.extract
    # CHECK: scf.if
    # CHECK: scf.if
    # CHECK: vector.load
    # CHECK: vector.store
    # CHECK: scf.yield
    # CHECK: scf.yield

    # --- Reduction and arithmetic patterns:
    # CHECK: gpu.shuffle  up
    # CHECK: arith.addf
    # CHECK: arith.select
    # CHECK: arith.minsi
    # CHECK: gpu.shuffle  xor
    # CHECK: vector.store
    # CHECK: return

    # CHECK-LABEL: func.func @isolated_benchmark$async(
    # CHECK: hal.tensor.import wait(
    # CHECK: flow.dispatch @speculative_sampling::@speculative_sampling(
    # CHECK-SAME: %0, %1, %2, %3
    # CHECK: hal.tensor.barrier join
    # CHECK: hal.tensor.export
    # CHECK: return
