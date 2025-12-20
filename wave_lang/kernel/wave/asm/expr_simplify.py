# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Algebraic simplification using declarative rewrite rules.

Rules are specified as (pattern, replacement) pairs using SymPy Wild symbols.
A single generic matcher handles all patterns.

REWRITE RULES
=============

┌──────────────────────────────────────────────────────────────────────────┐
│  Pattern                          →  Replacement                         │
├──────────────────────────────────────────────────────────────────────────┤
│  floor(x/n)*n + Mod(x, n)         →  x                                   │
│  a*x - a*n*floor(x/n)             →  a*Mod(x, n)                         │
│  floor(x/n)  when max(x) < n      →  0                                   │
│  Mod(Mod(x, a*b), a)              →  Mod(x, a)                           │
│  x * 2^a * 2^b                    →  x * 2^(a+b)                         │
└──────────────────────────────────────────────────────────────────────────┘

Usage:
    from expr_simplify import simplify_for_emission
    simplified = simplify_for_emission(expr, bounds={tid_x: (0, 63)})
"""

import os
import sympy
from sympy import Wild, floor, Mod, Integer
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Callable, Any

DEBUG_SIMPLIFY = os.environ.get("WAVE_EXPR_SIMPLIFY_LOG", "0") == "1"


# ============================================================================
# Wild Symbols for Pattern Matching
# ============================================================================

# Basic wildcards
_x = Wild('x')
_y = Wild('y')
_n = Wild('n', properties=[lambda k: isinstance(k, Integer) and k > 0])
_m = Wild('m', properties=[lambda k: isinstance(k, Integer) and k > 0])
_a = Wild('a')  # coefficient
_b = Wild('b')  # coefficient


# ============================================================================
# Rewrite Rules - Declarative Specification
# ============================================================================

@dataclass
class RewriteRule:
    """A rewrite rule: pattern → replacement."""
    name: str
    pattern: sympy.Expr
    replacement: sympy.Expr
    # Optional condition function: (match_dict, bounds) -> bool
    condition: Optional[Callable[[Dict, Dict], bool]] = None


# All rules in one place - easy to see and modify
REWRITE_RULES: List[RewriteRule] = [
    # Rule 1: Division algorithm identity
    # floor(x/n)*n + Mod(x, n) = x
    RewriteRule(
        name="floor_mod_identity",
        pattern=floor(_x / _n) * _n + Mod(_x, _n),
        replacement=_x,
    ),
    
    # Rule 2: Linear floor to mod
    # a*x - a*n*floor(x/n) = a*Mod(x, n)
    RewriteRule(
        name="linear_floor_to_mod",
        pattern=_a * _x + (-_a * _n) * floor(_x / _n),
        replacement=_a * Mod(_x, _n),
    ),
    
    # Rule 3: Nested mod simplification
    # Mod(Mod(x, a*b), a) = Mod(x, a)
    RewriteRule(
        name="nested_mod",
        pattern=Mod(Mod(_x, _n * _m), _n),
        replacement=Mod(_x, _n),
    ),
]


# ============================================================================
# Generic Pattern Matcher and Transformer
# ============================================================================

def try_match(expr: sympy.Expr, pattern: sympy.Expr) -> Optional[Dict]:
    """
    Try to match expr against pattern.
    
    Returns: Dictionary of {Wild: matched_expr} if successful, None otherwise.
    
    Uses SymPy's built-in match() for simple cases, with fallback handling
    for commutative operations (Add, Mul).
    """
    # Try direct match first
    match = expr.match(pattern)
    if match is not None:
        return match
    
    # For Add/Mul, try matching subset of terms (handles a+b+c matching a+b)
    if isinstance(expr, sympy.Add) and isinstance(pattern, sympy.Add):
        return _match_commutative_subset(expr, pattern, sympy.Add)
    
    if isinstance(expr, sympy.Mul) and isinstance(pattern, sympy.Mul):
        return _match_commutative_subset(expr, pattern, sympy.Mul)
    
    return None


def _match_commutative_subset(expr, pattern, op_class) -> Optional[Dict]:
    """
    Match pattern terms as a subset of expr terms.
    
    For example, match (a + b) in (a + b + c).
    """
    expr_terms = set(expr.args)
    pattern_terms = list(pattern.args)
    
    # Try to find a consistent assignment of wildcards
    # that matches all pattern terms to expr terms
    return _try_match_terms(list(expr_terms), pattern_terms, {})


def _try_match_terms(expr_terms: List, pattern_terms: List, bindings: Dict) -> Optional[Dict]:
    """Recursively try to match pattern terms to expr terms."""
    if not pattern_terms:
        return bindings  # All patterns matched
    
    pattern_term = pattern_terms[0]
    remaining_patterns = pattern_terms[1:]
    
    for i, expr_term in enumerate(expr_terms):
        # Try matching this expr_term to pattern_term
        match = expr_term.match(pattern_term)
        if match is not None:
            # Check consistency with existing bindings
            if _bindings_consistent(bindings, match):
                new_bindings = {**bindings, **match}
                remaining_exprs = expr_terms[:i] + expr_terms[i+1:]
                result = _try_match_terms(remaining_exprs, remaining_patterns, new_bindings)
                if result is not None:
                    return result
    
    return None


def _bindings_consistent(existing: Dict, new: Dict) -> bool:
    """Check if new bindings are consistent with existing ones."""
    for key, value in new.items():
        if key in existing and existing[key] != value:
            return False
    return True


def apply_rule(expr: sympy.Expr, rule: RewriteRule, bounds: Dict) -> Tuple[sympy.Expr, bool]:
    """
    Try to apply a rule to an expression.
    
    Returns: (result_expr, was_applied)
    """
    match = try_match(expr, rule.pattern)
    
    if match is None:
        return expr, False
    
    # Check condition if present
    if rule.condition is not None:
        if not rule.condition(match, bounds):
            return expr, False
    
    # Apply replacement by substituting matched values
    try:
        result = rule.replacement.xreplace(match)
        
        # If there were extra terms in expr (for Add/Mul), add them back
        if isinstance(expr, sympy.Add):
            matched_terms = set()
            for pattern_term in rule.pattern.args if isinstance(rule.pattern, sympy.Add) else [rule.pattern]:
                for expr_term in expr.args:
                    if expr_term.match(pattern_term.xreplace(match)) is not None:
                        matched_terms.add(expr_term)
                        break
            remaining = [t for t in expr.args if t not in matched_terms]
            if remaining:
                result = result + sympy.Add(*remaining)
        
        return result, True
    except Exception:
        return expr, False


def apply_all_rules(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """Apply all rewrite rules until no more changes."""
    changed = True
    while changed:
        changed = False
        for rule in REWRITE_RULES:
            expr, applied = apply_rule(expr, rule, bounds)
            if applied:
                changed = True
                if DEBUG_SIMPLIFY:
                    print(f"[{rule.name}] Applied")
                break  # Restart from first rule after any change
    return expr


# ============================================================================
# Special Rules (require custom logic beyond pattern matching)
# ============================================================================

def simplify_redundant_floor(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """
    Rule: floor(x/n) → 0 when max(x) < n
    
    This requires bounds checking, not just pattern matching.
    """
    if not (hasattr(expr, 'func') and expr.func == sympy.floor):
        return expr
    
    arg = expr.args[0]
    
    # Extract x and n from floor(x/n) = floor(x * (1/n))
    if isinstance(arg, sympy.Mul):
        x_parts = []
        n = None
        
        for factor in arg.args:
            if isinstance(factor, sympy.Pow) and factor.args[1] == -1:
                n = factor.args[0]
            elif isinstance(factor, sympy.Rational) and not isinstance(factor, sympy.Integer):
                n = sympy.Integer(factor.q)
                if factor.p != 1:
                    x_parts.append(sympy.Integer(factor.p))
            else:
                x_parts.append(factor)
        
        if n is not None and x_parts:
            x = sympy.Mul(*x_parts) if len(x_parts) > 1 else x_parts[0]
            max_x = _get_max_value(x, bounds)
            if max_x is not None and max_x < int(n):
                return sympy.Integer(0)
    
    return expr


def simplify_combine_shifts(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """
    Rule: x * 2^a * 2^b → x * 2^(a+b)
    
    Combines multiple power-of-2 factors.
    """
    if not isinstance(expr, sympy.Mul):
        return expr
    
    shift_amount = 0
    other_factors = []
    
    for factor in expr.args:
        if isinstance(factor, sympy.Integer):
            val = int(factor)
            if val > 0 and (val & (val - 1)) == 0:  # Power of 2
                shift_amount += (val.bit_length() - 1)
            else:
                other_factors.append(factor)
        elif isinstance(factor, sympy.Pow) and factor.args[0] == 2:
            exp = factor.args[1]
            if isinstance(exp, sympy.Integer):
                shift_amount += int(exp)
            else:
                other_factors.append(factor)
        else:
            other_factors.append(factor)
    
    if shift_amount > 0 and other_factors:
        base = sympy.Mul(*other_factors) if len(other_factors) > 1 else other_factors[0]
        return base * (2 ** shift_amount)
    
    return expr


def simplify_floor_difference(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """
    Rule: Combine floor(x/c) and floor(x/d) when d = k*c
    
    Uses identity: floor(x/c) = k*floor(x/d) + floor(Mod(x,d)/c) when d=k*c
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    # Collect floor terms: {(base, divisor): coefficient}
    floor_terms: Dict[Tuple, sympy.Expr] = {}
    other_terms: List[sympy.Expr] = []
    
    for term in expr.args:
        floor_info = _extract_floor_term(term)
        if floor_info is not None:
            coeff, base, divisor = floor_info
            key = (base, divisor)
            floor_terms[key] = floor_terms.get(key, sympy.Integer(0)) + coeff
        else:
            other_terms.append(term)
    
    if len(floor_terms) < 2:
        return expr
    
    # Find pairs where one divisor divides the other
    keys = list(floor_terms.keys())
    changed = False
    
    for i, (base1, div1) in enumerate(keys):
        for j, (base2, div2) in enumerate(keys):
            if i >= j or base1 != base2:
                continue
            if not (isinstance(div1, sympy.Integer) and isinstance(div2, sympy.Integer)):
                continue
            
            d1, d2 = int(div1), int(div2)
            if d1 > d2:
                div_large, div_small = div1, div2
                key_large, key_small = (base1, div1), (base2, div2)
            else:
                div_large, div_small = div2, div1
                key_large, key_small = (base2, div2), (base1, div1)
            
            d_large, d_small = int(div_large), int(div_small)
            if d_large % d_small != 0:
                continue
            
            # Apply the identity
            k = d_large // d_small
            coeff_small = floor_terms[key_small]
            coeff_large = floor_terms[key_large]
            
            floor_terms[key_large] = coeff_small * k + coeff_large
            del floor_terms[key_small]
            other_terms.append(coeff_small * sympy.floor(sympy.Mod(base1, div_large) / div_small))
            changed = True
            break
        if changed:
            break
    
    if not changed:
        return expr
    
    # Rebuild
    result_terms = list(other_terms)
    for (base, divisor), coeff in floor_terms.items():
        if coeff != 0:
            floor_expr = sympy.floor(base / divisor)
            result_terms.append(coeff * floor_expr if coeff != 1 else floor_expr)
    
    if not result_terms:
        return sympy.Integer(0)
    
    result = result_terms[0] if len(result_terms) == 1 else sympy.Add(*result_terms)
    return simplify_floor_difference(result, bounds)  # Recurse


def simplify_linear_floor_to_mod(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """
    Rule: a*x - a*n*floor(x/n) → a*Mod(x,n)
    
    Pattern matching version that handles the specific structure.
    """
    if not isinstance(expr, sympy.Add):
        return expr
    
    # Collect linear and floor terms
    linear: Dict[sympy.Symbol, sympy.Expr] = {}  # {symbol: coeff}
    floors: Dict[Tuple, sympy.Expr] = {}  # {(symbol, divisor): coeff}
    other: List[sympy.Expr] = []
    
    for term in expr.args:
        # Try linear: a*x
        if isinstance(term, sympy.Symbol):
            linear[term] = linear.get(term, 0) + 1
        elif isinstance(term, sympy.Mul):
            sym = None
            coeff = sympy.Integer(1)
            is_linear = True
            
            for f in term.args:
                if isinstance(f, sympy.Symbol):
                    if sym is None:
                        sym = f
                    else:
                        is_linear = False
                        break
                elif isinstance(f, (sympy.Integer, sympy.Rational)):
                    coeff *= f
                else:
                    is_linear = False
                    break
            
            if is_linear and sym is not None:
                linear[sym] = linear.get(sym, 0) + coeff
                continue
        
        # Try floor: c*floor(x/n)
        floor_info = _extract_floor_term(term)
        if floor_info and isinstance(floor_info[1], sympy.Symbol):
            coeff, sym, div = floor_info
            floors[(sym, div)] = floors.get((sym, div), 0) + coeff
            continue
        
        other.append(term)
    
    # Find matches: a*x paired with -a*n*floor(x/n)
    changed = False
    used_linear = set()
    used_floors = set()
    new_terms = list(other)
    
    for sym, a in linear.items():
        if sym in used_linear:
            continue
        for (fsym, n), b in floors.items():
            if fsym != sym or (fsym, n) in used_floors:
                continue
            if isinstance(n, sympy.Integer) and b == -a * int(n):
                # Match! a*x - a*n*floor(x/n) → a*Mod(x,n)
                new_terms.append(a * sympy.Mod(sym, n))
                used_linear.add(sym)
                used_floors.add((fsym, n))
                changed = True
                break
    
    if not changed:
        return expr
    
    # Add unused terms back
    for sym, coeff in linear.items():
        if sym not in used_linear:
            new_terms.append(coeff * sym if coeff != 1 else sym)
    for (sym, n), coeff in floors.items():
        if (sym, n) not in used_floors:
            new_terms.append(coeff * sympy.floor(sym / n) if coeff != 1 else sympy.floor(sym / n))
    
    if not new_terms:
        return sympy.Integer(0)
    return new_terms[0] if len(new_terms) == 1 else sympy.Add(*new_terms)


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_floor_term(term) -> Optional[Tuple[sympy.Expr, sympy.Expr, sympy.Integer]]:
    """Extract (coefficient, base, divisor) from c*floor(x/n)."""
    if hasattr(term, 'func') and term.func == sympy.floor:
        parts = _extract_floor_parts(term)
        if parts:
            return (sympy.Integer(1), parts[0], parts[1])
        return None
    
    if isinstance(term, sympy.Mul):
        coeff = sympy.Integer(1)
        floor_expr = None
        for f in term.args:
            if hasattr(f, 'func') and f.func == sympy.floor:
                floor_expr = f
            elif isinstance(f, (sympy.Integer, sympy.Rational)):
                coeff *= f
            else:
                return None
        if floor_expr:
            parts = _extract_floor_parts(floor_expr)
            if parts:
                return (coeff, parts[0], parts[1])
    return None


def _extract_floor_parts(floor_expr) -> Optional[Tuple[sympy.Expr, sympy.Integer]]:
    """Extract (base, divisor) from floor(base/divisor)."""
    if not (hasattr(floor_expr, 'func') and floor_expr.func == sympy.floor):
        return None
    
    arg = floor_expr.args[0]
    if isinstance(arg, sympy.Mul):
        divisor = None
        base_parts = []
        for f in arg.args:
            if isinstance(f, sympy.Pow) and f.args[1] == -1:
                divisor = f.args[0]
            elif isinstance(f, sympy.Rational) and not isinstance(f, sympy.Integer):
                divisor = sympy.Integer(f.q)
                if f.p != 1:
                    base_parts.append(sympy.Integer(f.p))
            else:
                base_parts.append(f)
        if divisor and base_parts:
            base = sympy.Mul(*base_parts) if len(base_parts) > 1 else base_parts[0]
            return (base, divisor)
    return None


def _get_max_value(expr: sympy.Expr, bounds: Dict) -> Optional[int]:
    """Get maximum value of expression given bounds."""
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if isinstance(expr, sympy.Symbol):
        for sym, (lo, hi) in bounds.items():
            if str(sym) == str(expr):
                return hi
        return None
    if isinstance(expr, sympy.Mod):
        _, n = expr.args
        return int(n) - 1 if isinstance(n, sympy.Integer) else None
    if isinstance(expr, sympy.Add):
        total = 0
        for arg in expr.args:
            m = _get_max_value(arg, bounds)
            if m is None:
                return None
            total += m
        return total
    if isinstance(expr, sympy.Mul):
        prod = 1
        for arg in expr.args:
            m = _get_max_value(arg, bounds)
            if m is None:
                return None
            prod *= m
        return prod
    return None


def get_default_bounds() -> Dict[sympy.Symbol, Tuple[int, int]]:
    """Default bounds for wave kernel symbols."""
    tid_x = sympy.Symbol("tid_x", nonnegative=True, integer=True)
    tid_y = sympy.Symbol("tid_y", nonnegative=True, integer=True)
    tid_z = sympy.Symbol("tid_z", nonnegative=True, integer=True)
    return {tid_x: (0, 1023), tid_y: (0, 1023), tid_z: (0, 1023)}


# ============================================================================
# Main Entry Point
# ============================================================================

def simplify_for_emission(
    expr: sympy.Expr,
    bounds: Optional[Dict[sympy.Symbol, Tuple[int, int]]] = None,
) -> sympy.Expr:
    """
    Simplify expression for optimal assembly emission.
    
    Applies declarative rewrite rules plus special-case simplifications.
    """
    if bounds is None:
        bounds = get_default_bounds()
    
    original = expr
    
    # Apply declarative rules
    expr = apply_all_rules(expr, bounds)
    
    # Apply special rules that need custom logic
    expr = _apply_special_rules(expr, bounds)
    
    # Recursively simplify subexpressions
    expr = _simplify_subexprs(expr, bounds)
    
    # Combine like terms
    if isinstance(expr, sympy.Add):
        expr = _combine_like_terms(expr)
    
    if DEBUG_SIMPLIFY and expr != original:
        print(f"[SIMPLIFY] {original} → {expr}")
    
    return expr


def _apply_special_rules(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """Apply rules that require custom logic beyond pattern matching."""
    expr = simplify_redundant_floor(expr, bounds)
    expr = simplify_combine_shifts(expr, bounds)
    expr = simplify_floor_difference(expr, bounds)
    expr = simplify_linear_floor_to_mod(expr, bounds)
    return expr


def _simplify_subexprs(expr: sympy.Expr, bounds: Dict) -> sympy.Expr:
    """Recursively simplify subexpressions."""
    if isinstance(expr, (sympy.Symbol, sympy.Integer, sympy.Rational)):
        return expr
    if not hasattr(expr, 'args') or not expr.args:
        return expr
    
    new_args = []
    changed = False
    for arg in expr.args:
        new_arg = apply_all_rules(arg, bounds)
        new_arg = _apply_special_rules(new_arg, bounds)
        if new_arg != arg:
            changed = True
        new_args.append(new_arg)
    
    if changed:
        try:
            return expr.func(*new_args)
        except:
            pass
    return expr


def _combine_like_terms(expr: sympy.Add) -> sympy.Expr:
    """Combine like terms: 3*x + 5*x → 8*x."""
    groups: Dict[tuple, List] = {}
    constants = []
    
    for term in expr.args:
        if isinstance(term, sympy.Integer):
            constants.append(term)
            continue
        
        if isinstance(term, sympy.Mul):
            coeff = sympy.Integer(1)
            base = []
            for f in term.args:
                if isinstance(f, (sympy.Integer, sympy.Rational)):
                    coeff *= f
                else:
                    base.append(f)
            if base:
                key = tuple(sorted(str(f) for f in base))
                if key not in groups:
                    groups[key] = []
                groups[key].append((coeff, base))
                continue
        
        key = (str(term),)
        if key not in groups:
            groups[key] = []
        groups[key].append((sympy.Integer(1), [term]))
    
    result = []
    for key, terms in groups.items():
        total = sum(t[0] for t in terms)
        if total == 0:
            continue
        base = terms[0][1]
        base_expr = sympy.Mul(*base) if len(base) > 1 else base[0]
        result.append(total * base_expr if total != 1 else base_expr)
    
    if sum(constants) != 0:
        result.append(sum(constants))
    
    if not result:
        return sympy.Integer(0)
    return result[0] if len(result) == 1 else sympy.Add(*result)


# ============================================================================
# Statistics
# ============================================================================

@dataclass  
class SimplifyStats:
    """Track simplification statistics."""
    total: int = 0
    simplified: int = 0
    
    def record(self, before, after):
        self.total += 1
        if before != after:
            self.simplified += 1
    
    def summary(self) -> str:
        if self.total == 0:
            return "No expressions"
        return f"Simplified {self.simplified}/{self.total} ({100*self.simplified/self.total:.0f}%)"
