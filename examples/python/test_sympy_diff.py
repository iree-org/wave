#!/usr/bin/env python3
"""
Test suite for mem_simplify — the domain-specific memory access simplifier.

Verifies that mem_simplify handles all the algebraic identities needed for
tensor memory indexing without relying on sympy.simplify().

Usage inside docker:
    source /workspace/wave/.venv/bin/activate
    python test_sympy_diff.py
"""

import sympy
from sympy import Symbol, Mod, floor, simplify, expand, Integer

from wave_lang.kernel.wave.utils.mapping_utils import (
    mem_simplify,
    linearize_dims,
    _eval_concrete_floor_mod,
    _expand_mod,
)

print(f"SymPy version: {sympy.__version__}")
print("=" * 70)

K = Symbol("_K_div_256", integer=True, positive=True)
idx0 = Symbol("$index0", integer=True, nonnegative=True)
idx1 = Symbol("$index1", integer=True, nonnegative=True)

PASS = 0
FAIL = 0


def check(name, got, expected):
    global PASS, FAIL
    ok = (
        sympy.expand(got - expected) == 0
        if not isinstance(got, bool)
        else got == expected
    )
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        print(f"         got:      {got}")
        print(f"         expected: {expected}")
        print()


# ── 1. The fundamental identity: a*floor(b/a) + Mod(b,a) == b ──────────
print("\n── 1. mem_simplify on floor/Mod identity ──")

for b_val in [32, 64, 96, 128, 256]:
    expr = K * floor(b_val / K) + Mod(b_val, K)
    check(f"K*floor({b_val}/K)+Mod({b_val},K)", mem_simplify(expr), b_val)

for b_val in [32, 64, 96, 128]:
    expr = 8 * K * floor(b_val / K) + 8 * Mod(b_val, K)
    check(f"8*K*floor({b_val}/K)+8*Mod({b_val},K)", mem_simplify(expr), 8 * b_val)

print(f"  Section 1: {PASS} passed")
p1 = PASS

# ── 2. linearize_dims: the 2D-to-flat round-trip ───────────────────────
print("\n── 2. linearize_dims ──")

Ks = 8 * K
for flat_val in [0, 256, 512, 768, 1024]:
    dim0 = floor(flat_val / Ks)
    dim1 = Mod(flat_val, Ks)
    result = linearize_dims([dim0, dim1], [Ks, Integer(1)])
    check(f"linearize_dims(flat={flat_val})", result, flat_val)

E = 64 * Mod(idx0, 4) + 4 * Mod(idx1, 16) + 256 * floor(idx0 / 8)
dim0_sym = floor(E / Ks)
dim1_sym = Mod(E, Ks)
result_sym = linearize_dims([dim0_sym, dim1_sym], [Ks, Integer(1)])
check("linearize_dims(symbolic flat)", mem_simplify(result_sym - E), 0)

print(f"  Section 2: {PASS - p1} passed")
p2 = PASS

# ── 3. _eval_concrete_floor_mod ────────────────────────────────────────
print("\n── 3. Concrete floor/Mod evaluation ──")

check("floor(32/8)", _eval_concrete_floor_mod(floor(Integer(32) / 8)), Integer(4))
check("floor(7/3)", _eval_concrete_floor_mod(floor(Integer(7) / 3)), Integer(2))
check("Mod(256, 64)", _eval_concrete_floor_mod(Mod(256, 64)), Integer(0))
check("Mod(10, 3)", _eval_concrete_floor_mod(Mod(10, 3)), Integer(1))
check("Mod(0, K)", _eval_concrete_floor_mod(Mod(0, K)), Integer(0))
check("floor(sym) unchanged", _eval_concrete_floor_mod(floor(K / 8)), floor(K / 8))

print(f"  Section 3: {PASS - p2} passed")
p3 = PASS

# ── 4. mem_simplify preserves already-simple expressions ───────────────
print("\n── 4. Passthrough cases ──")

check("integer", mem_simplify(Integer(42)), Integer(42))
check("symbol", mem_simplify(K), K)
check("linear expr", mem_simplify(3 * K + 7), 3 * K + 7)
check("floor(sym/n) unchanged", mem_simplify(floor(K / 8)), floor(K / 8))

print(f"  Section 4: {PASS - p3} passed")
p4 = PASS

# ── 5. _expand_mod ─────────────────────────────────────────────────────
print("\n── 5. _expand_mod ──")

check("expand_mod(Mod(32,K))", _expand_mod(Mod(32, K)), 32 - K * floor(32 / K))
expr5 = 8 * K * floor(32 / K) + 8 * Mod(32, K)
check("expand_mod + expand cancels", expand(_expand_mod(expr5)), Integer(256))

print(f"  Section 5: {PASS - p4} passed")
p5 = PASS

# ── 6. The full scale-mapping simulation ───────────────────────────────
print("\n── 6. Scale mapping addr simulation ──")

b_scale_flat = (
    256 * K * floor(idx1 / 32)
    + 64 * Mod(idx0, 4)
    + 4 * Mod(idx1, 16)
    + 256 * floor(idx0 / 8)
    + 2 * floor(Mod(idx0, 8) / 4)
    + floor(Mod(idx1, 32) / 16)
)

for iv_val, expected_addr in [(0, 0), (1, 256), (2, 512), (3, 768), (4, 1024)]:
    dim0 = floor(b_scale_flat / Ks)
    dim1 = Mod(b_scale_flat, Ks)
    addr_expr = linearize_dims([dim0, dim1], [Ks, Integer(1)])
    addr_concrete = mem_simplify(addr_expr.subs({idx0: 8 * iv_val, idx1: 0}))
    check(f"scale_addr(iv={iv_val})", addr_concrete, expected_addr)

print(f"  Section 6: {PASS - p5} passed")
p6 = PASS

# ── 7. Mod auto-factoring ─────────────────────────────────────────────
print("\n── 7. Mod factoring (SymPy built-in) ──")

for b_val in [32, 64, 128]:
    m1 = Mod(8 * b_val, 8 * K)
    m2 = 8 * Mod(b_val, K)
    check(f"Mod({8*b_val}, 8K) == 8*Mod({b_val},K)", simplify(m1 - m2), 0)

print(f"  Section 7: {PASS - p6} passed")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Results:  {PASS} PASS,  {FAIL} FAIL")
if FAIL:
    print("*** SOME TESTS FAILED ***")
else:
    print("All tests passed.")
print("=" * 70)

exit(1 if FAIL else 0)
