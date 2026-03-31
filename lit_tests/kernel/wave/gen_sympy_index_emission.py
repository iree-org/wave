# RUN: python %s | FileCheck %s

import sympy

from wave_lang.kernel.compiler.wave_codegen.emitter import _group_same_denom_fractions

x, y, z = sympy.symbols("x y z")

# CHECK-LABEL: same_denom_two_fractions
# x/5 + 2*y/5 should group into (x + 2*y) / 5.
# CHECK: (x + 2*y)/5
print("same_denom_two_fractions")
expr = sympy.Rational(1, 5) * x + sympy.Rational(2, 5) * y
print(_group_same_denom_fractions(expr))
print("-----")

# CHECK-LABEL: same_denom_three_fractions
# x/3 + y/3 + 2*z/3 should group into (x + y + 2*z) / 3.
# CHECK: (x + y + 2*z)/3
print("same_denom_three_fractions")
expr = sympy.Rational(1, 3) * x + sympy.Rational(1, 3) * y + sympy.Rational(2, 3) * z
print(_group_same_denom_fractions(expr))
print("-----")

# CHECK-LABEL: different_denom
# x/5 + y/7 has different denominators — should be unchanged.
# CHECK: x/5 + y/7
print("different_denom")
expr = sympy.Rational(1, 5) * x + sympy.Rational(1, 7) * y
print(_group_same_denom_fractions(expr))
print("-----")

# CHECK-LABEL: mixed_rational_and_integer
# x/4 + y: only one term has denom 4, no grouping needed — should be unchanged.
# CHECK: x/4 + y
print("mixed_rational_and_integer")
expr = sympy.Rational(1, 4) * x + y
print(_group_same_denom_fractions(expr))
print("-----")

# CHECK-LABEL: no_fractions
# x + 2*y: no fractions at all — should be unchanged.
# CHECK: x + 2*y
print("no_fractions")
expr = x + 2 * y
print(_group_same_denom_fractions(expr))
print("-----")
