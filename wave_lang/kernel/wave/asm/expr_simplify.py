# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Algebraic simplification for SymPy expressions before assembly emission.

This module provides aggressive algebraic simplification to reduce the number
of instructions emitted by the ASM backend. Key optimizations:

1. **Factor/combine terms**: Recognize `a*x + b*x` → `(a+b)*x`
2. **Floor/Mod cancellation**: Simplify `floor(x/n)*n + mod(x,n)` → `x`
3. **Shift fusion**: Recognize `x*2^n` → shift, combine multiple shifts
4. **Bound-aware simplification**: Use known ranges to eliminate redundant ops
5. **Term coalescing**: Reduce n-ary additions to fewer terms

Usage:
    from expr_simplify import simplify_for_emission
    
    simplified = simplify_for_emission(sympy_expr, bounds={tid_x: (0, 63)})
"""

import os
import sympy
from typing import Dict, Optional, Tuple, Set, List
from functools import lru_cache

# Debug flag for simplification logging
DEBUG_SIMPLIFY = os.environ.get("WAVE_EXPR_SIMPLIFY_LOG", "0") == "1"


# ============================================================================
# Known symbol bounds for the ASM backend
# ============================================================================

def get_default_bounds() -> Dict[sympy.Symbol, Tuple[int, int]]:
    """
    Get default bounds for common symbols in wave kernels.
    
    These bounds are based on hardware constraints:
    - tid_x, tid_y, tid_z: Thread ID within workgroup (max 1024 per dim)
    - wgid_x, wgid_y, wgid_z: Workgroup ID (can be large)
    - lane_id: 0-63 for CDNA
    """
    tid_x = sympy.Symbol("tid_x", nonnegative=True, integer=True)
    tid_y = sympy.Symbol("tid_y", nonnegative=True, integer=True)
    tid_z = sympy.Symbol("tid_z", nonnegative=True, integer=True)
    
    # For multi-wave kernels, tid_x/tid_y are typically 0-255 (4 waves * 64 threads)
    # but in practice often 0-63 for single-wave or 0-127 for 2-wave
    return {
        tid_x: (0, 1023),
        tid_y: (0, 1023),
        tid_z: (0, 1023),
    }


# ============================================================================
# Pattern-based simplification rules
# ============================================================================

def simplify_floor_mod_identity(expr: sympy.Expr) -> sympy.Expr:
    """
    Simplify floor/mod identity: floor(x/n)*n + mod(x,n) = x
    
    This is a common pattern in address computation that can be eliminated.
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    # Look for floor(x/n)*n + Mod(x, n) patterns
    terms = list(expr.args)
    
    for i, term1 in enumerate(terms):
        # Look for floor(x/n)*n
        if isinstance(term1, sympy.Mul):
            for factor in term1.args:
                if hasattr(factor, 'func') and factor.func == sympy.floor:
                    floor_arg = factor.args[0]
                    if isinstance(floor_arg, sympy.Mul):
                        # floor(x * (1/n)) * n pattern
                        for j, term2 in enumerate(terms):
                            if i != j and isinstance(term2, sympy.Mod):
                                # Check if they match
                                x1 = _extract_floor_dividend(factor)
                                n1 = _extract_floor_divisor(factor)
                                x2, n2 = term2.args
                                
                                if x1 is not None and n1 is not None:
                                    if x1 == x2 and n1 == n2:
                                        # Check if term1 is floor(x/n)*n
                                        remaining_factors = [f for f in term1.args if f != factor]
                                        if len(remaining_factors) == 1 and remaining_factors[0] == n1:
                                            # Found the pattern! Replace with x
                                            other_terms = [t for k, t in enumerate(terms) if k != i and k != j]
                                            return x1 + sum(other_terms) if other_terms else x1
    
    return expr


def _extract_floor_dividend(floor_expr: sympy.Expr) -> Optional[sympy.Expr]:
    """Extract x from floor(x/n)."""
    if not (hasattr(floor_expr, 'func') and floor_expr.func == sympy.floor):
        return None
    
    arg = floor_expr.args[0]
    
    # Handle floor(x/n) where division is explicit
    if isinstance(arg, sympy.Mul):
        # Look for x * (1/n) pattern
        for factor in arg.args:
            if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                # Found 1/n, the other factors are x
                other_factors = [f for f in arg.args if f != factor]
                if len(other_factors) == 1:
                    return other_factors[0]
                return sympy.Mul(*other_factors)
    
    # Handle floor(x * Rational(1, n))
    if isinstance(arg, sympy.Mul):
        for factor in arg.args:
            if isinstance(factor, sympy.Rational) and factor.p == 1:
                other_factors = [f for f in arg.args if f != factor]
                if len(other_factors) == 1:
                    return other_factors[0]
                return sympy.Mul(*other_factors)
    
    return None


def _extract_floor_divisor(floor_expr: sympy.Expr) -> Optional[sympy.Integer]:
    """Extract n from floor(x/n)."""
    if not (hasattr(floor_expr, 'func') and floor_expr.func == sympy.floor):
        return None
    
    arg = floor_expr.args[0]
    
    # Handle floor(x/n) where division is explicit
    if isinstance(arg, sympy.Mul):
        for factor in arg.args:
            if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                return factor.args[0]
            if isinstance(factor, sympy.Rational) and factor.p == 1:
                return sympy.Integer(factor.q)
    
    return None


def simplify_redundant_floor(expr: sympy.Expr, bounds: Dict[sympy.Symbol, Tuple[int, int]]) -> sympy.Expr:
    """
    Eliminate redundant floor operations using known bounds.
    
    If x is in [0, n-1], then floor(x/n) = 0.
    """
    if not (hasattr(expr, 'func') and expr.func == sympy.floor):
        return expr
    
    arg = expr.args[0]
    
    # Try to determine if floor is always 0
    if isinstance(arg, sympy.Mul):
        # floor(x * (1/n)) where x < n → 0
        divisor = _extract_floor_divisor(expr)
        dividend = _extract_floor_dividend(expr)
        
        if divisor is not None and dividend is not None:
            max_val = _get_max_value(dividend, bounds)
            if max_val is not None and max_val < int(divisor):
                return sympy.Integer(0)
    
    return expr


def _get_max_value(expr: sympy.Expr, bounds: Dict[sympy.Symbol, Tuple[int, int]]) -> Optional[int]:
    """Compute the maximum possible value of an expression given bounds."""
    if isinstance(expr, sympy.Integer):
        return int(expr)
    
    if isinstance(expr, sympy.Symbol):
        if expr in bounds:
            return bounds[expr][1]
        # Check by name
        for sym, (lo, hi) in bounds.items():
            if str(sym) == str(expr):
                return hi
        return None
    
    if isinstance(expr, sympy.Mod):
        # Mod(x, n) has max value n-1
        _, n = expr.args
        if isinstance(n, sympy.Integer):
            return int(n) - 1
        return None
    
    if isinstance(expr, sympy.Add):
        # Sum of maxes
        total = 0
        for arg in expr.args:
            max_arg = _get_max_value(arg, bounds)
            if max_arg is None:
                return None
            total += max_arg
        return total
    
    if isinstance(expr, sympy.Mul):
        # Product of maxes (for positive values)
        product = 1
        for arg in expr.args:
            max_arg = _get_max_value(arg, bounds)
            if max_arg is None:
                return None
            product *= max_arg
        return product
    
    return None


def factor_common_terms(expr: sympy.Expr) -> sympy.Expr:
    """
    Factor out common terms from additions.
    
    a*x + b*x → (a+b)*x
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    # Use SymPy's factor_terms which does this well
    try:
        factored = sympy.factor_terms(expr)
        return factored
    except:
        return expr


def combine_shifts(expr: sympy.Expr) -> sympy.Expr:
    """
    Combine consecutive shifts and recognize shift patterns.
    
    (x << a) << b → x << (a+b)
    x * 2^n → x << n
    """
    if isinstance(expr, sympy.Mul):
        # Look for x * power_of_2
        shift_amount = 0
        other_factors = []
        
        for factor in expr.args:
            if isinstance(factor, sympy.Integer):
                val = int(factor)
                if val > 0 and (val & (val - 1)) == 0:
                    # Power of 2
                    shift_amount += (val.bit_length() - 1)
                else:
                    other_factors.append(factor)
            elif isinstance(factor, sympy.Pow):
                base, exp = factor.args
                if base == 2 and isinstance(exp, sympy.Integer):
                    shift_amount += int(exp)
                else:
                    other_factors.append(factor)
            else:
                other_factors.append(factor)
        
        if shift_amount > 0 and other_factors:
            base = sympy.Mul(*other_factors) if len(other_factors) > 1 else other_factors[0]
            # Return as multiplication by power of 2 (emission will convert to shift)
            return base * (2 ** shift_amount)
    
    return expr


def simplify_nested_floor_mod(expr: sympy.Expr) -> sympy.Expr:
    """
    Simplify nested floor/mod patterns.
    
    floor(Mod(x, a*b) / a) → Mod(floor(x/a), b)
    Mod(Mod(x, a*b), a) → Mod(x, a)
    """
    # Handle floor(Mod(x, n*m) / n)
    if hasattr(expr, 'func') and expr.func == sympy.floor:
        arg = expr.args[0]
        if isinstance(arg, sympy.Mul):
            mod_term = None
            divisor = None
            for factor in arg.args:
                if isinstance(factor, sympy.Mod):
                    mod_term = factor
                elif isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                    divisor = factor.args[0]
                elif isinstance(factor, sympy.Rational) and factor.p == 1:
                    divisor = sympy.Integer(factor.q)
            
            if mod_term is not None and divisor is not None:
                x, modulus = mod_term.args
                if isinstance(modulus, sympy.Mul) and divisor in modulus.args:
                    # floor(Mod(x, n*m) / n) = Mod(floor(x/n), m)
                    other_factors = [f for f in modulus.args if f != divisor]
                    if len(other_factors) == 1:
                        m = other_factors[0]
                        return sympy.Mod(sympy.floor(x / divisor), m)
    
    # Handle Mod(Mod(x, a*b), a)
    if isinstance(expr, sympy.Mod):
        inner, outer_mod = expr.args
        if isinstance(inner, sympy.Mod):
            x, inner_mod = inner.args
            if isinstance(inner_mod, sympy.Mul) and outer_mod in inner_mod.args:
                # Mod(Mod(x, a*b), a) = Mod(x, a)
                return sympy.Mod(x, outer_mod)
    
    return expr


def simplify_linear_floor_to_mod(expr: sympy.Expr) -> sympy.Expr:
    """
    Simplify linear + floor patterns to modulo.
    
    Identity: a*x + b*floor(x/n) = a*Mod(x, n) when b = -a*n
    
    This eliminates a floor operation by recognizing that:
        a*x - a*n*floor(x/n) = a*(x - n*floor(x/n)) = a*Mod(x, n)
    
    Example:
        128*tid_x - 2048*floor(tid_x/16)
        Here: a=128, n=16, b=-2048, and -a*n = -128*16 = -2048 = b ✓
        Simplifies to: 128*Mod(tid_x, 16)
    
    This saves instructions: instead of LSHL + LSHR + LSHL + SUB, we get AND + LSHL.
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    terms = list(expr.args)
    
    # Collect linear terms: {symbol: coefficient}
    linear_terms = {}
    # Collect floor terms: {(symbol, divisor): coefficient}
    floor_terms = {}
    # Other terms that don't match these patterns
    other_terms = []
    
    for term in terms:
        linear_info = _extract_linear_term(term)
        if linear_info is not None:
            symbol, coeff = linear_info
            key = symbol
            if key in linear_terms:
                linear_terms[key] = linear_terms[key] + coeff
            else:
                linear_terms[key] = coeff
            continue
        
        floor_info = _extract_floor_term(term)
        if floor_info is not None:
            coeff, base, divisor = floor_info
            # Only handle simple symbol bases, not complex expressions
            if isinstance(base, sympy.Symbol):
                key = (base, divisor)
                if key in floor_terms:
                    floor_terms[key] = floor_terms[key] + coeff
                else:
                    floor_terms[key] = coeff
                continue
        
        other_terms.append(term)
    
    # Look for matching pairs: a*x and -a*n*floor(x/n)
    changed = False
    new_terms = list(other_terms)
    
    # Symbols that have been consumed by simplification
    consumed_linear = set()
    consumed_floor = set()
    
    for symbol, a in linear_terms.items():
        if symbol in consumed_linear:
            continue
        
        # Look for matching floor(symbol/n) term
        for (floor_sym, divisor), b in floor_terms.items():
            if floor_sym != symbol:
                continue
            if (floor_sym, divisor) in consumed_floor:
                continue
            
            # Check if b = -a*n
            n = divisor
            if not isinstance(n, (int, sympy.Integer)):
                continue
            n_val = int(n)
            
            expected_b = -a * n_val
            if b == expected_b:
                # Pattern matches! Replace with a*Mod(x, n)
                new_terms.append(a * sympy.Mod(symbol, n_val))
                consumed_linear.add(symbol)
                consumed_floor.add((floor_sym, divisor))
                changed = True
                break
    
    # Add back unconsumed linear terms
    for symbol, coeff in linear_terms.items():
        if symbol not in consumed_linear:
            if coeff == 1:
                new_terms.append(symbol)
            else:
                new_terms.append(coeff * symbol)
    
    # Add back unconsumed floor terms
    for (symbol, divisor), coeff in floor_terms.items():
        if (symbol, divisor) not in consumed_floor:
            floor_expr = sympy.floor(symbol / divisor)
            if coeff == 1:
                new_terms.append(floor_expr)
            else:
                new_terms.append(coeff * floor_expr)
    
    if not changed:
        return expr
    
    if not new_terms:
        return sympy.Integer(0)
    if len(new_terms) == 1:
        return new_terms[0]
    return sympy.Add(*new_terms)


def _extract_linear_term(term: sympy.Expr):
    """
    Extract linear term info: (symbol, coefficient) or None.
    
    Matches patterns like:
        x → (x, 1)
        a*x → (x, a)
        -x → (x, -1)
    
    Only matches simple symbol terms, not complex expressions.
    """
    # Direct symbol
    if isinstance(term, sympy.Symbol):
        return (term, sympy.Integer(1))
    
    # Coefficient * symbol
    if isinstance(term, sympy.Mul):
        coeff = sympy.Integer(1)
        symbol = None
        
        for factor in term.args:
            if isinstance(factor, sympy.Symbol):
                if symbol is not None:
                    return None  # Multiple symbols - not a simple linear term
                symbol = factor
            elif isinstance(factor, (sympy.Integer, sympy.Rational)):
                coeff = coeff * factor
            else:
                return None  # Other factor type - not a simple linear term
        
        if symbol is not None:
            return (symbol, coeff)
    
    return None


# ============================================================================
# Main simplification pipeline
# ============================================================================

def simplify_for_emission(
    expr: sympy.Expr,
    bounds: Optional[Dict[sympy.Symbol, Tuple[int, int]]] = None,
    aggressive: bool = True
) -> sympy.Expr:
    """
    Simplify a SymPy expression for optimal assembly emission.
    
    This applies a pipeline of simplifications focused on REDUCING INSTRUCTIONS,
    not mathematical simplicity. We avoid transformations that increase instruction count.
    
    Transformations applied:
    1. Expand to expose additive structure (don't factor!)
    2. Floor/Mod identity simplification
    3. Bound-aware redundant floor elimination  
    4. Nested floor/mod simplification
    5. Combine redundant terms
    
    Args:
        expr: The SymPy expression to simplify
        bounds: Optional dict mapping symbols to (min, max) bounds
        aggressive: If True, use more simplification passes
        
    Returns:
        Simplified expression
    """
    if bounds is None:
        bounds = get_default_bounds()
    
    original = expr
    
    # Step 1: Only apply targeted pattern-based rules
    # We AVOID SymPy's simplify/factor/cancel which can make expressions worse for codegen
    
    # Step 2: Custom pattern-based rules
    expr = _apply_custom_rules(expr, bounds)
    
    # Step 4: Recursive simplification on subexpressions
    expr = _simplify_subexpressions(expr, bounds)
    
    # Step 5: Combine like terms (but don't factor!)
    if isinstance(expr, sympy.Add):
        expr = _combine_like_terms(expr)
    
    if DEBUG_SIMPLIFY and expr != original:
        print(f"[SIMPLIFY] {original} → {expr}")
    
    return expr


def _combine_like_terms(expr: sympy.Add) -> sympy.Expr:
    """
    Combine like terms in an addition without factoring.
    
    3*x + 5*x → 8*x (good - reduces one ADD)
    8*x + 8*y → 8*x + 8*y (keep as is - factoring would add MUL)
    """
    # Group terms by their non-constant factors
    term_groups: Dict[tuple, List[sympy.Expr]] = {}
    constants = []
    
    for term in expr.args:
        if isinstance(term, sympy.Integer):
            constants.append(term)
            continue
        
        # Extract coefficient and base
        if isinstance(term, sympy.Mul):
            coeff = sympy.Integer(1)
            base_factors = []
            for factor in term.args:
                if isinstance(factor, (sympy.Integer, sympy.Rational)):
                    coeff = coeff * factor
                else:
                    base_factors.append(factor)
            if base_factors:
                base = tuple(sorted(str(f) for f in base_factors))
                key = base
            else:
                constants.append(coeff)
                continue
        else:
            coeff = sympy.Integer(1)
            key = (str(term),)
        
        if key not in term_groups:
            term_groups[key] = []
        term_groups[key].append((coeff, term))
    
    # Rebuild expression, combining coefficients for like terms
    new_terms = []
    
    for key, terms in term_groups.items():
        if len(terms) == 1:
            new_terms.append(terms[0][1])
        else:
            # Combine coefficients
            total_coeff = sum(t[0] for t in terms)
            if total_coeff == 0:
                continue
            # Get the base (non-coefficient part)
            base_term = terms[0][1]
            if isinstance(base_term, sympy.Mul):
                base_factors = [f for f in base_term.args if not isinstance(f, (sympy.Integer, sympy.Rational))]
                if base_factors:
                    base = sympy.Mul(*base_factors) if len(base_factors) > 1 else base_factors[0]
                    new_terms.append(total_coeff * base)
            else:
                new_terms.append(total_coeff * base_term)
    
    # Add constants
    const_sum = sum(constants)
    if const_sum != 0:
        new_terms.append(const_sum)
    
    if not new_terms:
        return sympy.Integer(0)
    if len(new_terms) == 1:
        return new_terms[0]
    return sympy.Add(*new_terms)


def _apply_custom_rules(expr: sympy.Expr, bounds: Dict[sympy.Symbol, Tuple[int, int]]) -> sympy.Expr:
    """Apply custom simplification rules."""
    # Apply each rule - NOTE: we intentionally do NOT call factor_common_terms
    # as factoring makes code generation worse (adds multiplies)
    expr = simplify_floor_mod_identity(expr)
    expr = simplify_redundant_floor(expr, bounds)
    expr = combine_shifts(expr)
    expr = simplify_nested_floor_mod(expr)
    expr = simplify_floor_difference(expr)
    expr = simplify_linear_floor_to_mod(expr)  # a*x - a*n*floor(x/n) → a*Mod(x, n)
    # factor_common_terms removed - it hurts code generation
    
    return expr


def _simplify_subexpressions(expr: sympy.Expr, bounds: Dict[sympy.Symbol, Tuple[int, int]]) -> sympy.Expr:
    """Recursively simplify subexpressions."""
    if isinstance(expr, (sympy.Symbol, sympy.Integer, sympy.Rational)):
        return expr
    
    if not hasattr(expr, 'args') or not expr.args:
        return expr
    
    # Simplify arguments
    new_args = []
    changed = False
    
    for arg in expr.args:
        new_arg = _apply_custom_rules(arg, bounds)
        if new_arg != arg:
            changed = True
        new_args.append(new_arg)
    
    if changed:
        try:
            return expr.func(*new_args)
        except:
            return expr
    
    return expr


def canonicalize_for_emission(expr: sympy.Expr) -> sympy.Expr:
    """
    Canonicalize expression for consistent emission.
    
    This ensures expressions are in a predictable form for pattern matching
    during code generation:
    - Additions sorted by term complexity
    - Multiplications with constants first
    - Nested operations flattened where possible
    """
    # Expand to expose all terms
    expr = sympy.expand(expr)
    
    # For additions, SymPy already maintains a canonical order
    # For multiplications, ensure constants come first
    if isinstance(expr, sympy.Mul):
        constants = []
        non_constants = []
        for arg in expr.args:
            if isinstance(arg, (sympy.Integer, sympy.Rational)):
                constants.append(arg)
            else:
                non_constants.append(arg)
        if constants and non_constants:
            const_product = sympy.Mul(*constants) if len(constants) > 1 else constants[0]
            expr = const_product * sympy.Mul(*non_constants)
    
    return expr


# ============================================================================
# Specialized simplifiers for common address patterns
# ============================================================================

def simplify_address_expr(
    expr: sympy.Expr,
    tid_x: sympy.Symbol,
    tid_y: sympy.Symbol,
    wgid_x: sympy.Symbol,
    wgid_y: sympy.Symbol,
    threads_per_wave: int = 64,
    waves_per_wg_x: int = 1,
    waves_per_wg_y: int = 1,
) -> sympy.Expr:
    """
    Simplify address expressions with wave/thread structure knowledge.
    
    This uses the fact that in a multi-wave workgroup:
    - tid_x = wave_id_x * wave_size_x + lane_x
    - tid_y = wave_id_y * wave_size_y + lane_y
    
    Many floor/mod patterns come from extracting these components.
    """
    # Build bounds based on wave structure
    max_tid_x = waves_per_wg_x * threads_per_wave - 1
    max_tid_y = waves_per_wg_y * threads_per_wave - 1
    
    bounds = {
        tid_x: (0, max_tid_x),
        tid_y: (0, max_tid_y),
        wgid_x: (0, 65535),  # Practical upper bound
        wgid_y: (0, 65535),
    }
    
    return simplify_for_emission(expr, bounds, aggressive=True)


# ============================================================================
# Statistics and debugging
# ============================================================================

class SimplifyStats:
    """Track simplification statistics."""
    
    def __init__(self):
        self.total_exprs = 0
        self.simplified_exprs = 0
        self.terms_before = 0
        self.terms_after = 0
    
    def record(self, before: sympy.Expr, after: sympy.Expr):
        self.total_exprs += 1
        if before != after:
            self.simplified_exprs += 1
        self.terms_before += _count_terms(before)
        self.terms_after += _count_terms(after)
    
    def summary(self) -> str:
        if self.total_exprs == 0:
            return "No expressions simplified"
        
        pct = 100 * self.simplified_exprs / self.total_exprs
        term_reduction = self.terms_before - self.terms_after
        return (
            f"Simplified {self.simplified_exprs}/{self.total_exprs} exprs ({pct:.1f}%), "
            f"reduced {term_reduction} terms ({self.terms_before} → {self.terms_after})"
        )


def _count_terms(expr: sympy.Expr) -> int:
    """Count the number of terms in an expression."""
    if isinstance(expr, sympy.Add):
        return len(expr.args)
    elif isinstance(expr, sympy.Mul):
        return sum(_count_terms(arg) for arg in expr.args)
    elif hasattr(expr, 'args') and expr.args:
        return sum(_count_terms(arg) for arg in expr.args)
    else:
        return 1



def simplify_floor_difference(expr: sympy.Expr) -> sympy.Expr:
    """
    Simplify differences of floor operations with related divisors.
    
    Key identity: When d = k*c (d is a multiple of c):
        floor(x/c) = (d/c)*floor(x/d) + floor(Mod(x,d)/c)
    
    So: a*floor(x/c) + b*floor(x/d) where d = k*c
      = a*((d/c)*floor(x/d) + floor(Mod(x,d)/c)) + b*floor(x/d)
      = (a*(d/c) + b)*floor(x/d) + a*floor(Mod(x,d)/c)
    
    This reduces the number of floor operations by combining related ones.
    
    Example:
        512*floor(x/64) - 512*floor(x/16)
        → -1536*floor(x/64) - 512*floor(Mod(x,64)/16)
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    # Collect floor terms: {(base_expr, divisor): coefficient}
    floor_terms = {}
    other_terms = []
    
    for term in expr.args:
        floor_info = _extract_floor_term(term)
        if floor_info is not None:
            coeff, base, divisor = floor_info
            key = (base, divisor)
            if key in floor_terms:
                floor_terms[key] = floor_terms[key] + coeff
            else:
                floor_terms[key] = coeff
        else:
            other_terms.append(term)
    
    if len(floor_terms) < 2:
        return expr  # No opportunity for combination
    
    # Look for pairs where one divisor divides the other
    keys = list(floor_terms.keys())
    changed = False
    
    for i, (base1, div1) in enumerate(keys):
        for j, (base2, div2) in enumerate(keys):
            if i >= j:
                continue
            if base1 != base2:
                continue
            
            # Check if one divides the other
            if not (isinstance(div1, sympy.Integer) and isinstance(div2, sympy.Integer)):
                continue
            
            d1, d2 = int(div1), int(div2)
            
            # Make div_large > div_small
            if d1 > d2:
                div_large, div_small = div1, div2
                key_large, key_small = (base1, div1), (base2, div2)
            else:
                div_large, div_small = div2, div1
                key_large, key_small = (base2, div2), (base1, div1)
            
            d_large, d_small = int(div_large), int(div_small)
            
            if d_large % d_small != 0:
                continue
            
            # Apply the identity!
            # floor(x/small) = (large/small)*floor(x/large) + floor(Mod(x,large)/small)
            k = d_large // d_small
            coeff_small = floor_terms[key_small]
            coeff_large = floor_terms[key_large]
            
            # New coefficient for floor(x/large): coeff_small * k + coeff_large
            new_coeff_large = coeff_small * k + coeff_large
            
            # New term: coeff_small * floor(Mod(x, large) / small)
            mod_floor_term = coeff_small * sympy.floor(sympy.Mod(base1, div_large) / div_small)
            
            # Update floor_terms
            floor_terms[key_large] = new_coeff_large
            del floor_terms[key_small]
            other_terms.append(mod_floor_term)
            
            changed = True
            break
        
        if changed:
            break
    
    if not changed:
        return expr
    
    # Rebuild expression
    result_terms = list(other_terms)
    for (base, divisor), coeff in floor_terms.items():
        if coeff == 0:
            continue
        floor_expr = sympy.floor(base / divisor)
        if coeff == 1:
            result_terms.append(floor_expr)
        else:
            result_terms.append(coeff * floor_expr)
    
    if not result_terms:
        return sympy.Integer(0)
    
    result = sympy.Add(*result_terms) if len(result_terms) > 1 else result_terms[0]
    
    # Recursively apply to handle multiple pairs
    return simplify_floor_difference(result)


def _extract_floor_term(term: sympy.Expr):
    """
    Extract floor term info: (coefficient, base_expr, divisor) or None.
    
    Matches patterns like:
        floor(x/n) → (1, x, n)
        a * floor(x/n) → (a, x, n)
        -floor(x/n) → (-1, x, n)
    """
    # Direct floor
    if hasattr(term, 'func') and term.func == sympy.floor:
        base, divisor = _extract_floor_components(term)
        if base is not None:
            return (sympy.Integer(1), base, divisor)
        return None
    
    # Coefficient * floor
    if isinstance(term, sympy.Mul):
        coeff = sympy.Integer(1)
        floor_expr = None
        
        for factor in term.args:
            if hasattr(factor, 'func') and factor.func == sympy.floor:
                floor_expr = factor
            elif isinstance(factor, (sympy.Integer, sympy.Rational)):
                coeff = coeff * factor
            else:
                # Other factor - not a simple floor term
                return None
        
        if floor_expr is not None:
            base, divisor = _extract_floor_components(floor_expr)
            if base is not None:
                return (coeff, base, divisor)
    
    return None


def _extract_floor_components(floor_expr: sympy.Expr):
    """
    Extract (base, divisor) from floor(base/divisor).
    
    Returns (base, divisor) or (None, None) if not a simple floor division.
    """
    if not (hasattr(floor_expr, 'func') and floor_expr.func == sympy.floor):
        return None, None
    
    arg = floor_expr.args[0]
    
    # Handle floor(x/n) where division creates x * (1/n)
    if isinstance(arg, sympy.Mul):
        divisor = None
        base_factors = []
        
        for factor in arg.args:
            if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                # Found 1/n
                if divisor is not None:
                    return None, None  # Multiple divisors
                divisor = factor.args[0]
            elif isinstance(factor, sympy.Rational) and not isinstance(factor, sympy.Integer):
                if divisor is not None:
                    return None, None
                divisor = sympy.Integer(factor.q)
                if factor.p != 1:
                    base_factors.append(sympy.Integer(factor.p))
            else:
                base_factors.append(factor)
        
        if divisor is not None and base_factors:
            base = sympy.Mul(*base_factors) if len(base_factors) > 1 else base_factors[0]
            return base, divisor
    
    return None, None
