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

num_query_heads = 8
num_kv_heads = 8
query_seq_len = 128
head_size_kv = 128
head_size = 64
kv_seq_len = 256

q_shape = (num_query_heads, query_seq_len, head_size)
k_shape = (num_kv_heads, kv_seq_len, head_size)
v_shape = (num_kv_heads, kv_seq_len, head_size_kv)
o_shape = (num_query_heads, query_seq_len, head_size_kv)

shape=AttentionShape(num_query_heads=num_query_heads,
                     num_kv_heads=num_kv_heads,
                     query_seq_len=query_seq_len,
                     head_size_kv=head_size_kv,
                     head_size=head_size,
                     kv_seq_len=kv_seq_len)

mfma_variant = (tkw.MMAType.RDNA4_WAVE32_F32_16x16x16_F16, ) * 2

attention_kernel, hp, a = get_vanilla_attention_kernel(shape, mfma_variant, dynamic_dims = False)
options = WaveCompileOptions(subs=hp, canonicalize=True, run_bench=False, schedule=SchedulingType.NONE, use_scheduling_barriers=False,func_name="test", target="gfx1201")
attention = wave_compile(options, attention_kernel)

q = device_randn(q_shape, dtype = torch.float16)
k = device_randn(k_shape, dtype = torch.float16)
v = device_randn(v_shape, dtype = torch.float16)
o = device_zeros(o_shape, dtype = torch.float32)

attention(q, k, v, o)
torch_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)

torch.allclose(o, torch_ref, check_dtype=False)

print("finish")
