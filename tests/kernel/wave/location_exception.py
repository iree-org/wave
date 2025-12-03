import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel._support.location import StackTraceInfo
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.water import is_water_available
import pytest

from iree.compiler.ir import Context

M = tkl.sym.M
K = tkl.sym.K


@pytest.mark.skipif(
    not is_water_available(), reason="Water MLIR package not installed."
)
def test_error_messages():
    @tkw.wave(
        [
            tkw.HardwareConstraint(threads_per_wave=64, vector_shapes={M: 1, K: 1}),
            tkw.WorkgroupConstraint(M, 1, 0),
            tkw.WorkgroupConstraint(K, 1, 1),
        ]
    )
    def empty(a: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16]):
        tkw.read(a)

    location = StackTraceInfo.capture_current_location()
    with Context():
        location_str = str(location.to_mlir())

    override_ir = (
        """
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "foo"}> ({
    %0 = "arith.constant"() <{value = false}> : () -> i1
    "cf.assert"(%0) <{msg = "assertion message test"}> : (i1) -> () """
        + location_str
        + """
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    )

    options = WaveCompileOptions(
        subs={M: 4, K: 4},
        compile_to_mlir=True,
        use_water_leak_check=override_ir,
    )

    with pytest.raises(Exception) as exception:
        wave_compile(options, empty)

    assert "assertion known to be false" in repr(exception.value)
    assert "location_exception.py" in repr(exception.value)
