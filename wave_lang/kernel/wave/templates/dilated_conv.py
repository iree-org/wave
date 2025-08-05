from typing import Any, Optional

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.dtype import DataType
from wave_lang.kernel.lang.global_symbols import (
    SHARED_ADDRESS_SPACE,
    GLOBAL_ADDRESS_SPACE,
)

def get_dilated_conv2d(
    layout: str,
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    stride: int,
    dilation: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    ratio_m: Optional[int] = None,
    ratio_n: Optional[int] = None,
) -> tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]:
    """This Kernel computes dilated convolution with specified dilation rate.
    
    Dilated convolution samples the input at dilated intervals, effectively increasing the receptive field without increasing the number of parameters.
    
    Parameters:
        layout (str): Either "nchw_fchw" or "nhwc_hwcf" based on the ordering of the dims of the input tensors.
        n (int): Number of input (Batch size).
        h (int): Height of input matrix.
        w (int): Width of input matrix.
        c (int): Number of channels in input matrix.
        hf (int): Height of weight (filter) matrix.
        wf (int): Width of weight (filter) matrix.
        nf (int): Number of filters.
        stride (int): Convolution stride distance.
        dilation (int): Dilation rate for dilated convolution. dilation=1 is regular convolution.
        input_dtype (DataType): Input and filter datatype, currently only supports tkl.f16.
        output_dtype (DataType): Output matrix datatype, currently only supports tkl.f32.
        mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
        block_m (int | None): M dim tile size.
        block_n (int | None): N dim tile size.
        block_k (int | None): K dim tile size.
        ratio_m (int | None): Divider for M dim tile size and number of waves.
        ratio_n (int | None): Divider for N dim tile size and number of waves.

    Returns:
        output (tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]): Wave kernel to be compiled and hyperparameters.
    """
    # Input Checks
    assert input_dtype == tkl.f16, f"Unsupported input dtype: {input_dtype}"
    assert output_dtype == tkl.f32, f"Unsupported output dtype: {output_dtype}"
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    
    padding = 0  # only pad=0 is supported for now

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF
    DILATION = sym.DILATION

    # Calculate effective kernel size with dilation
    effective_hf = (hf - 1) * dilation + 1
    effective_wf = (wf - 1) * dilation + 1
    
    H_OUT = (H + 2 * padding - effective_hf) // stride + 1
    W_OUT = (W + 2 * padding - effective_wf) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    # Index mapping for dilated convolution
    # The key difference is multiplying kernel offsets by dilation rate
    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j % C,
            H: (i % SZ_OUT) % W_OUT * stride + ((j // C) % WF) * DILATION,
            W: (i % SZ_OUT) // W_OUT * stride + ((j // C) // WF) * DILATION,
        },
        outputs={M: i, K: j},
    )

    # Weight mapping remains the same as regular convolution
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j % C, HF: (j // C) % WF, WF: (j // C) // WF},
        outputs={NF: i, K: j},
    )
    
    # Output mapping remains the same
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    if layout == "nchw_fchw":
        x_type = tkl.Memory[N, C, H, W, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, NF, H_OUT, W_OUT, GLOBAL_ADDRESS_SPACE, output_dtype]
    elif layout == "nhwc_hwcf":
        x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, output_dtype]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if block_m is None:
        block_m = 64

    if block_n is None:
        block_n = 128

    if block_k is None:
        block_k = 32

    if ratio_m is None:
        ratio_m = 2

    if ratio_n is None:
        ratio_n = 2
    
    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N / ratio_n)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_m, ratio_n, 1),
        )
    ]

    @tkw.wave(constraints)
    def dilated_conv(
        x: x_type,
        we: we_type,
        dilation_rate: tkl.i32,
        out: out_type,
    ):
        # Set dilation symbol with dilation_rate value
        tkw.set_symbol(DILATION, dilation_rate)
        
        # Create reduction loop Register
        c_reg = tkl.Register[M, NF, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, NF, output_dtype],
        ) -> tkl.Register[M, NF, output_dtype]:
            # Read input matrix with dilated sampling
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            
            # Read filter (same as regular convolution)
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            
            # Compute mma for MxK x KxNF
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Write result using output mapping
        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    symbols = {
        N: n,
        C: c,
        W: w,
        H: h,
        NF: nf,
        WF: wf,
        HF: hf,
        DILATION: dilation,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: mem_space,
    }
    
    return dilated_conv, symbols
