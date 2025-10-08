# Copyright 2025 The Wave Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
from water_mlir import ir


from typing import List
from dataclasses import dataclass


class AffineConversionError(Exception):
    """Exception raised for errors in affine expression conversion."""

    pass


@dataclass
class AffineFraction:
    """Represents a fractional affine expression: numerator / denominator"""

    numerator: ir.AffineExpr
    denominator: ir.AffineExpr

    @classmethod
    def from_integer(cls, value: int) -> "AffineFraction":
        """Create a fraction from an integer"""
        return cls.from_affine_expr(ir.AffineExpr.get_constant(value))

    @classmethod
    def from_affine_expr(cls, expr: ir.AffineExpr) -> "AffineFraction":
        """Create a fraction from an AffineExpr"""
        return cls(expr, ir.AffineExpr.get_constant(1))

    @staticmethod
    def is_affine_expr_constant_one(expr: ir.AffineExpr) -> bool:
        return isinstance(expr, ir.AffineConstantExpr) and expr.value == 1

    def __add__(self, other: "AffineFraction") -> "AffineFraction":
        """Add two fractions: a/b + c/d = (a*d + c*b)/(b*d)"""
        if not isinstance(other, AffineFraction):
            return NotImplemented

        # Both denominators must be constants for addition
        a, b = self.numerator, self.denominator
        c, d = other.numerator, other.denominator

        # Compute common denominator: b*d
        common_denom = b * d

        # Compute numerators: a*d + c*b
        if b == d:
            # Same denominator: simple addition
            return AffineFraction(a + c, b)

        # Different denominators: a*d + c*b
        term1 = a * d if not AffineFraction.is_affine_expr_constant_one(d) else a
        term2 = c * b if not AffineFraction.is_affine_expr_constant_one(b) else c
        new_numerator = term1 + term2
        return AffineFraction(term1 + term2, common_denom)

    def __mul__(self, other: "AffineFraction") -> "AffineFraction":
        """Multiply two fractions: (a/b) * (c/d) = (a*c)/(b*d)"""
        if not isinstance(other, AffineFraction):
            return NotImplemented

        numerator = self.numerator * other.numerator
        if isinstance(self.denominator, ir.AffineConstantExpr) and isinstance(
            other.denominator, ir.AffineConstantExpr
        ):
            denominator = ir.AffineExpr.get_constant(
                self.denominator.value * other.denominator.value
            )
        else:
            denominator = self.denominator * other.denominator

        return AffineFraction(numerator, denominator)


def _convert_expr_to_fraction(
    sympy_expr: sympy.core.expr.Expr, symbols: List[sympy.core.symbol.Symbol]
) -> AffineFraction:
    """Recursively convert sympy expression to AffineFraction"""
    if sympy_expr.is_Integer:
        return AffineFraction.from_integer(sympy_expr.p)

    elif sympy_expr.is_Symbol:
        if sympy_expr in symbols:
            symbol_idx = symbols.index(sympy_expr)
            return AffineFraction.from_affine_expr(ir.AffineExpr.get_symbol(symbol_idx))
        else:
            available_symbols = [str(s) for s in symbols]
            raise AffineConversionError(
                f"Unknown symbol '{sympy_expr}'. Available symbols: {available_symbols}"
            )

    elif sympy_expr.is_Rational:
        return AffineFraction(
            ir.AffineExpr.get_constant(sympy_expr.p),
            ir.AffineExpr.get_constant(sympy_expr.q),
        )

    elif sympy_expr.is_Add:
        # Convert all terms to fractions and find common denominator
        fractions = []
        result = AffineFraction.from_integer(0)
        for term in sympy_expr.args:
            frac = _convert_expr_to_fraction(term, symbols)
            result += frac
        return result

    elif sympy_expr.is_Mul:
        # Multiply fractions: (a/b) * (c/d) = (a*c)/(b*d)
        result = AffineFraction.from_integer(1)

        for factor in sympy_expr.args:
            frac = _convert_expr_to_fraction(factor, symbols)
            result *= frac

        return result

    elif sympy_expr.is_Pow:
        # Division by symbol is also expressed as Pow: x^(-1) = 1/x
        if not sympy_expr.exp.is_Integer:
            raise AffineConversionError(
                f"Only integer exponents are supported. Got: {sympy_expr.exp}"
            )

        base_frac = _convert_expr_to_fraction(sympy_expr.base, symbols)
        result = base_frac

        if sympy_expr.exp == 0:
            return AffineFraction.from_integer(1)

        for _ in range(1, abs(sympy_expr.exp)):
            result *= base_frac

        return (
            AffineFraction(
                result.denominator,
                result.numerator,
            )
            if sympy_expr.exp < 0
            else AffineFraction(
                result.numerator,
                result.denominator,
            )
        )

    elif hasattr(sympy_expr, "func"):
        func_name = sympy_expr.func.__name__

        if func_name in ("floor", "ceiling") and len(sympy_expr.args) == 1:
            arg_frac = _convert_expr_to_fraction(sympy_expr.args[0], symbols)

            # Apply floor or ceiling to the fraction
            if func_name == "floor":
                result_expr = ir.AffineExpr.get_floor_div(
                    arg_frac.numerator, arg_frac.denominator
                )
            else:  # ceiling
                result_expr = ir.AffineExpr.get_ceil_div(
                    arg_frac.numerator, arg_frac.denominator
                )

            return AffineFraction.from_affine_expr(result_expr)

        elif func_name == "Mod" and len(sympy_expr.args) == 2:
            x_frac = _convert_expr_to_fraction(sympy_expr.args[0], symbols)
            y_frac = _convert_expr_to_fraction(sympy_expr.args[1], symbols)

            result_expr = x_frac.numerator % y_frac.numerator
            return AffineFraction.from_affine_expr(result_expr)

        else:
            raise AffineConversionError(
                f"Unsupported function '{func_name}': {sympy_expr}"
            )

    else:
        raise AffineConversionError(
            f"Unsupported expression type: {type(sympy_expr).__name__}: {sympy_expr}"
        )


def convert_sympy_to_affine_map(
    expr: sympy.core.expr.Expr, symbols: List[sympy.core.symbol.Symbol]
) -> ir.AffineMap:
    """
    Convert a sympy expression to an MLIR AffineMap using fractional algebra.

    Args:
        expr: sympy expression to convert
        symbols: list of symbols that the expression might contain

    Returns:
        AffineMap representing the expression

    Raises:
        AffineConversionError: if conversion fails for any reason
    """

    fraction = _convert_expr_to_fraction(expr, symbols)

    # Final validation: we must have valid fraction (i.e. denominator = 1) result for AffineMap
    if not AffineFraction.is_affine_expr_constant_one(fraction.denominator):
        raise AffineConversionError(
            f"Expression results in invalid fraction: {expr}\n"
            f"  Numerator: {fraction.numerator}\n"
            f"  Denominator: {fraction.denominator}"
        )

    return ir.AffineMap.get(0, len(symbols), [fraction.numerator])
