import logging
import unittest

import sympy

from wave_lang.kernel.compiler.wave_codegen.emitter import _group_same_denom_fractions


class GroupSameDenomFractionsTest(unittest.TestCase):
    """Tests for _group_same_denom_fractions."""

    def setUp(self):
        self.x, self.y, self.z = sympy.symbols("x y z")

    def test_same_denominator_two_fractions(self):
        """x/5 + 2y/5 should group into (x + 2y) / 5."""
        expr = sympy.Rational(1, 5) * self.x + sympy.Rational(2, 5) * self.y
        result = _group_same_denom_fractions(expr)
        n, d = result.as_numer_denom()
        self.assertEqual(d, 5)
        self.assertEqual(sympy.expand(n), self.x + 2 * self.y)

    def test_same_denominator_three_fractions(self):
        """x/3 + y/3 + 2z/3 should group into (x + y + 2z) / 3."""
        expr = (
            sympy.Rational(1, 3) * self.x
            + sympy.Rational(1, 3) * self.y
            + sympy.Rational(2, 3) * self.z
        )
        result = _group_same_denom_fractions(expr)
        n, d = result.as_numer_denom()
        self.assertEqual(d, 3)
        self.assertEqual(sympy.expand(n), self.x + self.y + 2 * self.z)

    def test_different_denominators_unchanged(self):
        """x/5 + y/7 has different denominators — should be unchanged."""
        expr = sympy.Rational(1, 5) * self.x + sympy.Rational(1, 7) * self.y
        result = _group_same_denom_fractions(expr)
        self.assertEqual(result, expr)

    def test_mixed_rational_and_integer_unchanged(self):
        """x/4 + y: only one term has denom 4 — should be unchanged."""
        expr = sympy.Rational(1, 4) * self.x + self.y
        result = _group_same_denom_fractions(expr)
        self.assertEqual(result, expr)

    def test_no_fractions_unchanged(self):
        """x + 2y: no fractions — should be unchanged."""
        expr = self.x + 2 * self.y
        result = _group_same_denom_fractions(expr)
        self.assertEqual(result, expr)

    def test_non_add_passthrough(self):
        """Non-Add expressions should be returned as-is."""
        expr = self.x * self.y
        result = _group_same_denom_fractions(expr)
        self.assertEqual(result, expr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
