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

    # With dynamic_dims=False there are no dynamic dim placeholders; export has 3 stride args only.
    # CHECK: stream.executable.export public @gemm workgroups(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index)

    # Kernel func: 3 bindings + 3 index (stride) arguments (one leading stride per buffer).
    # CHECK: func.func @gemm(%{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: !stream.binding, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index)

    # F32 output buffer keeps a 2-D view with dynamic leading stride (%arg5); this appears first in @gemm.
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1024, 1024], strides: [%arg5, 1]
    # CHECK-SAME: memref<f32> to memref<1024x1024xf32, strided<[?, 1]>>

    # Input f16 bindings are linearized to 1-D; vector.load uses a single offset index.
    # CHECK-COUNT-2: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1073741822], strides: [1] : memref<f16> to memref<1073741822xf16, strided<[1]>>
    # CHECK: vector.load %reinterpret_cast_1{{.*}} : memref<1073741822xf16, strided<[1]>>, vector<8xf16>
    # CHECK: vector.load %reinterpret_cast_2{{.*}} : memref<1073741822xf16, strided<[1]>>, vector<8xf16>

    # Output is linearized using dynamic strides from extract_strided_metadata, then stored to 1D view.
    # CHECK: memref.extract_strided_metadata %reinterpret_cast : memref<1024x1024xf32, strided<[?, 1]>>
    # CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [536870910], strides: [1]
    # CHECK: vector.store {{.*}} %reinterpret_cast_3{{.*}} : memref<536870910xf32, strided<[1], offset: ?>>

    # Host dispatch still passes three stride indices (ABI); linearization elides %arg3/%arg4 from f16 reinterprets.
    # CHECK: flow.dispatch @gemm::@gemm[%arg3, %arg4, %arg5]
