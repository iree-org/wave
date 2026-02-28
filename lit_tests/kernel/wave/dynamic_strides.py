# RUN: python %s | FileCheck %s

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.utils.general_utils import run_test


@run_test
def test_dynamic_strides_gemm():
    shape = (1024, 1024, 1024)
    gemm, hyperparams, dynamic_symbols = get_gemm_kernel(
        shape,
        dynamic_dims=False,
        mfma_variant=MMAType.F32_16x16x16_F16,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
        wave_runtime=True,
        compile_to_mlir=True,
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # Export must accept 6 workload args (3 dynamic dim placeholders + 6 stride args when wave_runtime;
    # for this gemm with no dynamic dims, all 6 are stride args).

    # CHECK: stream.executable.export public @gemm workgroups(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index)

    # Kernel func: 3 bindings + 6 index (stride) arguments.
    # CHECK: func.func @gemm(%{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index)

    # First buffer: reinterpret_cast uses stride arg for leading dim, 1 for innermost (vector.load requirement).
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1024, 1024], strides: [%arg3, 1]
    # CHECK-SAME: memref<f16> to memref<1024x1024xf16, strided<[?, 1]>>

    # Second buffer: strides [%arg5, 1]
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1024, 1024], strides: [%arg5, 1]
    # CHECK-SAME: memref<f16> to memref<1024x1024xf16, strided<[?, 1]>>

    # Third buffer: strides [%arg7, 1]
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1024, 1024], strides: [%arg7, 1]
    # CHECK-SAME: memref<f32> to memref<1024x1024xf32, strided<[?, 1]>>

    # vector.load on input memrefs must show strided<[?, 1]> type (dynamic strides).
    # CHECK: vector.load %reinterpret_cast{{.*}} : memref<1024x1024xf16, strided<[?, 1]>>
    # CHECK: vector.load %reinterpret_cast_0{{.*}} : memref<1024x1024xf16, strided<[?, 1]>>

    # Output is linearized using dynamic strides from extract_strided_metadata, then stored to 1D view.
    # CHECK: memref.extract_strided_metadata %reinterpret_cast_1 : memref<1024x1024xf32, strided<[?, 1]>>
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [536870910], strides: [1]
    # CHECK: vector.store {{.*}} %reinterpret_cast_3{{.*}} : memref<536870910xf32, strided<[1], offset: ?>>
