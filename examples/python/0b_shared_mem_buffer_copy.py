import torch
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
import wave_lang.kernel.wave.compile as compiler


def main():
    constraints = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={
                tkl.sym.M: tkl.sym.THREADS_PER_WAVE_M,
                tkl.sym.N: tkl.sym.THREADS_PER_WAVE_N,
            },
        ),
        tkw.WaveConstraint(tkl.sym.M, tkl.sym.THREADS_PER_WAVE_M),
        tkw.WaveConstraint(tkl.sym.N, tkl.sym.THREADS_PER_WAVE_N),
        tkw.WorkgroupConstraint(tkl.sym.M, tkl.sym.THREADS_PER_WORKGROUP_M, 0),
        tkw.WorkgroupConstraint(tkl.sym.N, tkl.sym.THREADS_PER_WORKGROUP_N, 1),
    ]

    @tkw.wave(constraints)
    def kernel_def(
        inp: tkl.Memory[tkl.sym.M, tkl.sym.N, tkl.sym.ADDRESS_SPACE, tkl.i8],
        out: tkl.Memory[
            tkl.sym.M, tkl.sym.N, tkl.global_symbols.GLOBAL_ADDRESS_SPACE, tkl.i8
        ],
    ):
        reg = tkw.read(inp, elements_per_thread=tkl.sym.THREADS_PER_WAVE_N)
        tkw.write(reg, out, elements_per_thread=tkl.sym.THREADS_PER_WAVE_N)

    inp = torch.randint(high=32, size=(1024, 4096), dtype=torch.int8, device="cuda")
    out = torch.randint(high=32, size=(1024, 4096), dtype=torch.int8, device="cuda")

    hyper_params = {
        tkl.sym.M: 1024,
        tkl.sym.N: 4096,
        tkl.sym.THREADS_PER_WAVE_M: 64,
        tkl.sym.THREADS_PER_WAVE_N: 32,
        tkl.sym.THREADS_PER_WORKGROUP_M: 64,
        tkl.sym.THREADS_PER_WORKGROUP_N: 64,
        tkl.sym.ADDRESS_SPACE: tkl.global_symbols.SHARED_ADDRESS_SPACE,
    }

    default_params = tkw.utils.general_utils.get_default_scheduling_params()
    hyper_params.update(default_params)
    options = compiler.WaveCompileOptions(subs=hyper_params)
    options = tkw.utils.run_utils.set_default_run_config(options)

    kernel = compiler.wave_compile(options, kernel_def)
    kernel(inp, out)
    torch.testing.assert_close(inp, out)
    print("ok")


if __name__ == "__main__":
    main()
