# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Generate test/corpus_expected.txt from test/corpus.txt using SymPy.

Reads corpus.txt (format: "simplify time: X.XXXXs: <expression>" per non-blank
line), strips the prefix, parses with sympy.sympify, applies simplify with
assumptions from corpus_assumptions.txt, and writes simplified expressions
to corpus_expected.txt. Uses Mod(p, q, evaluate=False) to work around SymPy
issue #28744 (Mod squaring bug with integer=True symbols).

SymPy's parser cannot handle $ in identifiers, so $VAR is replaced with
dollar_VAR in the input string before parsing; symbols keep their real
names for correct output.
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

import sympy

# All 20 variables from DESIGN.md
VARIABLES = (
    "$T0",
    "$T1",
    "$T2",
    "$WG0",
    "$WG1",
    "$ARGK",
    "$GPR_NUM",
    "$MMA_ACC",
    "$MMA_LHS_SCALE",
    "$MMA_RHS_SCALE",
    "$MMA_SCALE_FP4",
    "$index0",
    "$index1",
    "_M_div_32",
    "_N_div_32",
    "_K_div_256",
    "_aligned",
    "M",
    "N",
    "K",
)

# Variables with nonnegative assumption (>= 0)
NONNEGATIVE = frozenset(VARIABLES)

# Variables with positive assumption (>= 1)
POSITIVE = frozenset(("M", "N", "K"))

# Map corpus name -> Python-safe name for parsing ($VAR -> dollar_VAR)
_NAME_TO_PY: dict[str, str] = {}
for v in VARIABLES:
    if v.startswith("$"):
        _NAME_TO_PY[v] = "dollar_" + v[1:]
    else:
        _NAME_TO_PY[v] = v


def _make_locals() -> dict[str, sympy.Symbol]:
    """Build symbol dict for sympify: integer=True, nonneg/positive where applicable."""
    locals_: dict[str, sympy.Symbol] = {}
    for name in VARIABLES:
        kwargs: dict[str, bool] = {"integer": True}
        if name in POSITIVE:
            kwargs["positive"] = True
        elif name in NONNEGATIVE:
            kwargs["nonnegative"] = True
        sym = sympy.Symbol(name, **kwargs)
        locals_[_NAME_TO_PY[name]] = sym
    return locals_


def _substitute_names(s: str) -> str:
    """Replace $VAR with dollar_VAR so SymPy parser can handle it."""
    # Replace longest names first to avoid $T0 matching inside $T01 etc.
    for name in sorted(VARIABLES, key=len, reverse=True):
        if name.startswith("$"):
            s = s.replace(name, _NAME_TO_PY[name])
    return s


def _fix_mods(expr: sympy.Basic) -> sympy.Basic:
    """Rebuild Mod nodes with evaluate=False to avoid SymPy #28744."""
    if getattr(expr, "is_Mod", lambda: False)():
        return sympy.Mod(_fix_mods(expr.args[0]), _fix_mods(expr.args[1]), evaluate=False)
    if expr.is_Atom:
        return expr
    return expr.func(*(_fix_mods(a) for a in expr.args))


def _extract_expression(line: str) -> str | None:
    """Extract raw expression from corpus line. Returns None for blank lines."""
    line = line.strip()
    if not line:
        return None
    # Format: "simplify time: X.XXXXs: <expression>"
    idx = line.find(": ")
    if idx == -1:
        return line
    idx = line.find(": ", idx + 1)
    if idx == -1:
        return line
    return line[idx + 2 :].strip()


def _load_assumptions(path: Path, locals_: dict[str, sympy.Symbol]) -> list[sympy.Basic]:
    """Parse assumption file into SymPy expressions.

    Build programmatically to avoid $ parse issues.
    """
    assumptions: list[sympy.Basic] = []
    op_map = {
        ">=": sympy.Ge,
        "<=": sympy.Le,
        ">": sympy.Gt,
        "<": sympy.Lt,
        "==": sympy.Eq,
        "!=": sympy.Ne,
    }
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\S+)\s*(>=|<=|>|<|==|!=)\s*(-?\d+)\s*$", line)
        if not m:
            sys.stderr.write(f"Warning: could not parse assumption '{line}'\n")
            continue
        var_name, op, val_str = m.groups()
        py_name = _NAME_TO_PY.get(var_name, var_name)
        if py_name not in locals_:
            sys.stderr.write(f"Warning: unknown variable '{var_name}' in assumption\n")
            continue
        assumptions.append(op_map[op](locals_[py_name], int(val_str)))
    return assumptions


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    corpus_path = root / "test" / "corpus.txt"
    assumptions_path = root / "test" / "corpus_assumptions.txt"
    output_path = root / "test" / "corpus_expected.txt"

    if not corpus_path.exists():
        sys.stderr.write(f"Error: {corpus_path} not found\n")
        return 1
    if not assumptions_path.exists():
        sys.stderr.write(f"Error: {assumptions_path} not found\n")
        return 1

    locals_ = _make_locals()
    assumptions = _load_assumptions(assumptions_path, locals_)
    assumption_and = sympy.And(*assumptions) if assumptions else sympy.true

    lines = corpus_path.read_text().splitlines()
    expressions: list[str] = []
    for i, line in enumerate(lines):
        raw = _extract_expression(line)
        if raw is None:
            continue
        try:
            parseable = _substitute_names(raw)
            expr = sympy.sympify(parseable, locals=locals_)
        except Exception as e:
            sys.stderr.write(f"Error parsing line {i + 1}: {e}\n")
            expressions.append(raw)  # Copy verbatim on parse error
            continue

        try:
            simplified = sympy.simplify(expr)
            if assumptions:
                with contextlib.suppress(ValueError, TypeError):
                    simplified = sympy.refine(simplified, assumption_and)
            simplified = _fix_mods(simplified)
            expressions.append(str(simplified))
        except Exception as e:
            sys.stderr.write(f"Error simplifying line {i + 1}: {e}\n")
            expressions.append(raw)  # Copy verbatim on simplify error

        if (len(expressions)) % 100 == 0:
            print(f"Processed {len(expressions)} expressions...", file=sys.stderr)

    output_path.write_text("\n".join(expressions) + "\n")
    print(f"Wrote {len(expressions)} expressions to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
