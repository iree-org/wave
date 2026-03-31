# RUN: python %s | FileCheck %s

import sympy

from iree.compiler.ir import (
    Context,
    Location,
    InsertionPoint,
    IndexType,
    FunctionType,
    Module,
)
from iree.compiler.dialects import func as func_d

from wave_lang.kernel._support.indexing import IndexingContext
from wave_lang.kernel.compiler.wave_codegen.emitter import gen_sympy_index


def _gen_ir(expr, sym_list):
    """Emit *expr* inside a throwaway function and return the MLIR string."""
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            idx = IndexType.get()
            ftype = FunctionType.get([idx] * len(sym_list), [idx])
            func = func_d.FuncOp("test_fn", ftype)
            entry = func.add_entry_block()
            with InsertionPoint(entry):
                with IndexingContext() as idxc:
                    idxc.finalize()
                    dynamics = {s: entry.arguments[i] for i, s in enumerate(sym_list)}
                    result = gen_sympy_index(dynamics, expr)
                    func_d.return_([result])
        return str(module)


x, y, z = sympy.symbols("x y z")

# CHECK-LABEL: same_denom_two_fractions
# x/5 + 2*y/5 should keep denominator 5, not cross-multiply to 25.
# Single symbol denominator (no multiplication in floordiv operand).
# CHECK: floordiv s{{[0-9]+}})>
# CHECK-NOT: arith.constant 25
print("same_denom_two_fractions")
expr = sympy.Rational(1, 5) * x + sympy.Rational(2, 5) * y
print(_gen_ir(expr, [x, y]))
print("-----")

# CHECK-LABEL: same_denom_three_fractions
# x/3 + y/3 + 2*z/3 should keep denominator 3, not blow up to 9 or 27.
# CHECK: floordiv s{{[0-9]+}})>
# CHECK-NOT: arith.constant 9
# CHECK-NOT: arith.constant 27
print("same_denom_three_fractions")
expr = sympy.Rational(1, 3) * x + sympy.Rational(1, 3) * y + sympy.Rational(2, 3) * z
print(_gen_ir(expr, [x, y, z]))
print("-----")

# CHECK-LABEL: different_denom
# x/5 + y/7 must cross-multiply: denominator is a product of two symbols.
# CHECK: floordiv (s{{[0-9]+}} * s{{[0-9]+}})
print("different_denom")
expr = sympy.Rational(1, 5) * x + sympy.Rational(1, 7) * y
print(_gen_ir(expr, [x, y]))
print("-----")

# CHECK-LABEL: mixed_rational_and_integer
# x/4 + y should become (x + 4*y) / 4.
# CHECK: floordiv s{{[0-9]+}})>
# CHECK-NOT: arith.constant 16
print("mixed_rational_and_integer")
expr = sympy.Rational(1, 4) * x + y
print(_gen_ir(expr, [x, y]))
print("-----")
