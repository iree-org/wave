import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.compile as compiler
import wave_lang.kernel.wave.utils.torch_utils as torch_utils


def main():
    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,  # usually a constant for AMD GPUs
            waves_per_block=(1, 1, 1),
            # ratio of workgroup tile size (BLOCK_M) to the wave tile size (BLOCK_M)
            # hence invariant: workgroup tile size >= wave tile size
            # distribute rows among threads
            # each thread reads entire row
            vector_shapes={tkl.sym.M: 64, tkl.sym.N: 64},
        ),
        # workgroup tile size in the M dimension
        tkw.WorkgroupConstraint(tkl.sym.M, tkl.sym.BLOCK_M, 0),
        tkw.WorkgroupConstraint(tkl.sym.N, tkl.sym.BLOCK_N, 1),
        # wave tile size in the M dimension
        tkw.WaveConstraint(tkl.sym.M, tkl.sym.BLOCK_M),
        tkw.WaveConstraint(tkl.sym.N, tkl.sym.BLOCK_N),
    ]

    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            tkl.sym.M: tkw.IndexMapping.iterator(0),
            tkl.sym.N: tkw.IndexMapping.iterator(1),
        },
        outputs={
            tkl.sym.M: tkw.IndexMapping.iterator(0),
            tkl.sym.N: tkw.IndexMapping.iterator(1),
        },
    )

    @tkw.wave(constraints)
    def kernel_def(
        inp: tkl.Memory[tkl.sym.M, tkl.sym.N, tkl.sym.GLOBAL_ADDRESS_SPACE, tkl.i8],
        out: tkl.Memory[tkl.sym.N, tkl.sym.M, tkl.sym.GLOBAL_ADDRESS_SPACE, tkl.i8],
    ):
        inp_reg = tkw.read(inp, elements_per_thread=tkl.sym.BLOCK_N)
        tkw.write(inp_reg, out, mapping=mapping, elements_per_thread=tkl.sym.BLOCK_N)

    inp = torch_utils.device_randint(high=32, size=(256, 64), dtype=torch.int8)
    out = torch_utils.device_randint(high=32, size=(64, 256), dtype=torch.int8)

    hyper_params = {
        tkl.sym.M: 256,
        tkl.sym.N: 64,
        tkl.sym.BLOCK_M: 64,
        tkl.sym.BLOCK_N: 64,
    }

    default_params = tkw.utils.general_utils.get_default_scheduling_params()
    hyper_params.update(default_params)
    options = compiler.WaveCompileOptions(subs=hyper_params)
    options = tkw.utils.run_utils.set_default_run_config(options)

    kernel = compiler.wave_compile(options, kernel_def)
    kernel(inp, out)
    torch.testing.assert_close(out, inp.T)
    print("ok")


if __name__ == "__main__":
    main()
