#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify that the amalgamated shared library exports only public API symbols.

Compiles ixsimpl_amalg.c into a temporary .so, extracts exported text
symbols via nm, and compares against function declarations parsed from
include/ixsimpl.h.  Exits 0 if clean, 1 if leaked or missing symbols
are found.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AMALG = REPO / "ixsimpl_amalg.c"
HEADER = REPO / "include" / "ixsimpl.h"

FUNC_DECL_RE = re.compile(
    r"^(?:ixs_\w+)\s*\(",
    re.MULTILINE,
)


def parse_public_api(header: Path) -> set[str]:
    """Extract function names declared in the public header."""
    text = header.read_text()
    names: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("typedef") or line.startswith("#") or line.startswith("/*"):
            continue
        m = re.search(r"\b(ixs_\w+)\s*\(", line)
        if m:
            name = m.group(1)
            if name not in ("ixs_ctx", "ixs_node", "ixs_visit_fn"):
                names.add(name)
    return names


def build_so(amalg: Path) -> Path:
    """Compile the amalgamation into a temp shared library."""
    tmp = Path(tempfile.mktemp(suffix=".so"))
    cmd = [
        "gcc",
        "-std=c99",
        "-pedantic",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O0",
        "-shared",
        "-fPIC",
        "-o",
        str(tmp),
        str(amalg),
        f"-I{REPO / 'include'}",
        f"-I{REPO / 'src'}",
    ]
    subprocess.check_call(cmd, stderr=subprocess.PIPE)
    return tmp


def get_exported_functions(so_path: Path) -> set[str]:
    """Get exported text symbols from a shared library."""
    out = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(so_path)],
        text=True,
    )
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "T":
            names.add(parts[2])
    return names


def main() -> int:
    if not AMALG.exists():
        print(f"error: {AMALG} not found; run scripts/amalgamate.py first", file=sys.stderr)
        return 1

    public_api = parse_public_api(HEADER)
    so_path = build_so(AMALG)
    try:
        exported = get_exported_functions(so_path)
    finally:
        so_path.unlink(missing_ok=True)

    leaked = exported - public_api
    missing = public_api - exported

    ok = True
    if leaked:
        print(f"LEAKED ({len(leaked)} symbols not in public API):")
        for name in sorted(leaked):
            print(f"  {name}")
        ok = False
    if missing:
        print(f"MISSING ({len(missing)} public API symbols not exported):")
        for name in sorted(missing):
            print(f"  {name}")
        ok = False

    if ok:
        print(f"OK: {len(exported)} exported symbols match public API.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
