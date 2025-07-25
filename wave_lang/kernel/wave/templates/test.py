from torch.testing import assert_close
import torch

from wave_lang.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.compile_options import WaveCompileOptions
from wave_lang.kernel.wave.templates.transpose_conv import get_transpose_conv2d
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_randn

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw

def upsample_with_zeros(x, stride_h, stride_w):
    N, C, H, W = x.shape
    H_out = H * stride_h
    W_out = W * stride_w
    out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    # out[:, :, ::stride_h, ::stride_w] = x
    for ni in range(N):
        for hi in range(H):
            for wi in range(W):
                for ci in range(C):
                    out[ni, ci, hi * stride_h, wi * stride_w] = x[ni, ci, hi, wi]
    return out

def upsample_with_zeros(x, upsamp_stride):
    """Helper for upsampling, spreads elements out by upsamp_stride in h and w dims."""
    N, C, H, W = x.shape
    H_out = H * upsamp_stride
    W_out = W * upsamp_stride
    out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    out[:, :, ::upsamp_stride, ::upsamp_stride] = x
    return out


if __name__ == "__main__":
    use_random = True
    print_asm = False
    n, h, w, c = 16, 48, 48, 288
    nf, hf, wf = 288, 3, 3
    cf = c
    upsamp_stride = 2
    padding = 0
    output_padding = 0
    m = ((h * upsamp_stride - hf) +1) * ((w * upsamp_stride - wf) + 1) * n
    k = hf * wf * c

    # Input and filter
    
    x = device_randn(n, c, h, w, dtype=torch.float16)
    we = device_randn(nf, cf, hf, wf, dtype=torch.float16)
    if not use_random:
        x = x.fill_(1.0)
        we = we.fill_(0.25)
    we_flipped = torch.flip(we, dims=[2, 3])
    # Reference manual transposed conv
    x_up = upsample_with_zeros(x, upsamp_stride)
    convRef = torch.nn.Conv2d(c, nf, hf, stride=1, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we_flipped)
    out_ref = convRef(x_up).detach().to(torch.float32)

    mask_holder = torch.zeros(m, k, device=x.device, dtype=x.dtype)

    layout = "nchw_fchw" 
    #layout = "nhwc_hwcf"

    if layout == "nchw_fchw":
        pass  # Nothing
    elif layout == "nhwc_hwcf":
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        we = torch.permute(we, (2, 3, 1, 0)).contiguous()
        out_ref = torch.permute(out_ref, (0, 2, 3, 1)).contiguous()
    else:
        raise ValueError(f"Invalid layout: {layout}")
    # Get compiled IREE kernel
    trans_conv, hyperparams = get_transpose_conv2d(
        layout=layout,
        n=n,
        h=h,
        w=w,
        c=c,
        hf=hf,
        wf=wf,
        nf=nf,
        upsamp_stride=upsamp_stride,
        conv_stride=1,
        mem_space=GLOBAL_ADDRESS_SPACE,
        input_dtype=tkl.f16,
        output_dtype=tkl.f32,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        wave_runtime=True,
        print_mlir=True,
        iree_launch_async=False
    )
    options = set_default_run_config(options)
    trans_conv = wave_compile(options, trans_conv)
    if print_asm:
        print(trans_conv.asm)

    out = torch.zeros_like(out_ref)
    trans_conv(x, we, upsamp_stride, out)
    # out = out * -1
    # Print results
    # print(mask_holder)
    # print("Input (x):")
    # print(x[0, 0])
    # print(x.shape)

    # print("\nWeight:")
    # print(we[0, 0])
    # print("\nWeight Flipped:")
    # print(we_flipped[0, 0])
 
    # print("\nManual Transposed Convolution Output:")
    # print(out_ref)
    # print(out_ref.shape)
    # print(f"\nWave Result:\n{out}")
    # print(out.shape)
    # assert_close(out, out_ref, rtol=1e-02, atol=1e-02)
    # print("Results are the Same!")

