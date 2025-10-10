# REQUIRES: water_python
# RUN: env PYTHONPATH=%py_pkg_root:%PYTHONPATH %python %s | FileCheck %s

from water_mlir import ir
from water_mlir.sympy_to_affine_converter import (
    convert_sympy_to_affine_map,
    AffineConversionError,
)

import sympy


def test_in_context(f):
    with ir.Context():
        f()
    return f


@test_in_context
def test_basic_conversion():
    symbols = ["x", "y", "z"]

    result = convert_sympy_to_affine_map(sympy.sympify("x + 2*y + 3"), symbols)
    # CHECK: ()[s0, s1, s2] -> (s0 + s1 * 2 + 3)
    print(result)


@test_in_context
def test_mul_conversion():
    symbols = ["x", "y"]

    try:
        convert_sympy_to_affine_map(sympy.sympify("x/4"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: x/4" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    try:
        convert_sympy_to_affine_map(sympy.sympify("x/y"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: x/y" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    try:
        convert_sympy_to_affine_map(sympy.sympify("3/x"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: 3/x" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    try:
        convert_sympy_to_affine_map(sympy.sympify("y*(2/3)"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: 2*y/3" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    try:
        convert_sympy_to_affine_map(sympy.sympify("1/y"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: 1/y" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    try:
        convert_sympy_to_affine_map(sympy.sympify("(x+5)/(y+x)"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: (x + 5)/(x + y)" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."


@test_in_context
def test_rational_conversion():
    symbols = []

    try:
        convert_sympy_to_affine_map(sympy.sympify("1/2"), symbols)
    except AffineConversionError as e:
        assert "Expression results in invalid fraction: 1/2" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."


@test_in_context
def test_floor_ceil_conversion():
    symbols = ["x", "y", "z"]

    # CHECK: ()[s0, s1, s2] -> ((s0 * 3) floordiv 2)
    print(convert_sympy_to_affine_map(sympy.sympify("floor(3*x/2)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (s1 ceildiv 5)
    print(convert_sympy_to_affine_map(sympy.sympify("ceiling(y/5)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (5)
    print(convert_sympy_to_affine_map(sympy.sympify("floor(5)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (7)
    print(convert_sympy_to_affine_map(sympy.sympify("ceiling(7)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (s2)
    print(convert_sympy_to_affine_map(sympy.sympify("floor(z)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (s1 floordiv s2)
    print(convert_sympy_to_affine_map(sympy.sympify("floor(y/z)"), symbols))
    # CHECK: ()[s0, s1, s2] -> (s0 ceildiv s1)
    print(convert_sympy_to_affine_map(sympy.sympify("ceiling(x/y)"), symbols))
    # CHECK: ()[s0, s1, s2] -> ((s0 + 3) floordiv (s1 + s2))
    print(convert_sympy_to_affine_map(sympy.sympify("floor((3+x)/(y+z))"), symbols))
    # CHECK: ()[s0, s1, s2] -> ((s0 + 2) ceildiv s1)
    print(convert_sympy_to_affine_map(sympy.sympify("ceiling((x+2)/y)"), symbols))
    # CHECK: ()[s0, s1, s2] -> ((s0 ceildiv s1) * (3 floordiv s0) + (s0 * 4 + s2) ceildiv 4 + s0 floordiv 2)
    print(
        convert_sympy_to_affine_map(
            sympy.sympify("floor(x/2) + ceiling(x/y) * floor(3/x) + ceiling(x + z/4)"),
            symbols,
        )
    )


@test_in_context
def test_mod_conversion():
    symbols = ["x", "y"]

    # CHECK: ()[s0, s1] -> (s0 mod s1)
    print(convert_sympy_to_affine_map(sympy.sympify("Mod(x, y)"), symbols))
    # CHECK: ()[s0, s1] -> ((s1 floordiv 5) mod 2)
    print(convert_sympy_to_affine_map(sympy.sympify("Mod(floor(y/5), 2)"), symbols))
    # CHECK: ()[s0, s1] -> (3 mod (s0 + 4))
    print(convert_sympy_to_affine_map(sympy.sympify("Mod(3, x+4)"), symbols))


@test_in_context
def test_pow_conversion():
    symbols = ["x", "y"]

    # CHECK: ()[s0, s1] -> ((s0 * s0) * s0)
    print(convert_sympy_to_affine_map(sympy.sympify("x^3"), symbols))

    try:
        convert_sympy_to_affine_map(sympy.sympify("x^y"), symbols)
    except AffineConversionError as e:
        assert "Only integer exponents are supported. Got: y" in str(e)
    else:
        assert False, "Expected to fail with AffineConversionError."

    # CHECK: ()[s0, s1] -> ((s0 * s0) * s0 + s1 * s1)
    print(convert_sympy_to_affine_map(sympy.sympify("x^3 + y^2"), symbols))
    # CHECK: ()[s0, s1] -> (((s0 * s0) * 8 + ((s1 * s1) * s1) * 9) floordiv (((s1 * s1) * s1) * (s0 * s0)))
    print(
        convert_sympy_to_affine_map(sympy.sympify("floor((3/x)^2 + 8*y^(-3))"), symbols)
    )
