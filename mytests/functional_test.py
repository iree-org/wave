import torch
from wave_lang.kernel.wave.iree_utils import generate_iree_ref
from wave_lang.kernel.wave.utils.torch_utils import (
        device_randn,
        device_randint,
        device_zeros,
        device_ones,

)
from wave_lang.kernel.wave.templates.gemm import get_gemm_kernel
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.templates.vanilla_attention import *

shape=AttentionShape(num_query_heads=8, num_kv_heads=8, query_seq_len=128, head_size_kv=128, head_size=64, kv_seq_len=256)
mfma_variant_0 = (tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16, ) * 2
attention_kernel, hp, a = get_vanilla_attention_kernel(shape, mfma_variant_0, False)

options = WaveCompileOptions(subs=hp, canonicalize=True, run_bench=False, schedule=SchedulingType.NONE, use_scheduling_barriers=False,func_name="test", target="gfx1201")

attention = wave_compile(options, attention_kernel)

write_files="./wmma_f32_16x16x16_f16.mlir"
with open(write_files, "w") as fh: fh.write(attention.asm)


shape = (64,64,64)
mfma_variant_1 = tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16
gemm_kernel, hp, symbols = get_gemm_kernel(shape, False, mfma_variant_1, torch.float16, TPW=32, per_wave_process_shape = (32, 32, 32))

options = WaveCompileOptions(subs=hp, canonicalize=True, dynamic_symbols=symbols, run_bench=False, schedule=SchedulingType.NONE, use_scheduling_barriers=False,func_name="test", target="gfx1201")

gemm = wave_compile(options, gemm_kernel)

a = device_ones(shape[0], shape[2], dtype=torch.float16)
b = device_ones(shape[1], shape[2], dtype=torch.float16)
c = device_zeros(shape[0], shape[1], dtype=torch.float32)
gemm(a, b, c)

iree_ref = device_zeros(shape[0], shape[1], dtype=torch.float32)
generate_iree_ref("mmt", [a, b], [iree_ref], options)

print("finish")
