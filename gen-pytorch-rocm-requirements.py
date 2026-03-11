#!/usr/bin/env python3
"""Detect the installed ROCm version and emit a pytorch-rocm-requirements.txt
that points pip at a wheel source carrying matching ROCm builds of PyTorch.

Two wheel sources are tried, in order:

  1. PyTorch Foundation index  --index-url https://download.pytorch.org/whl/rocmX.Y
  2. AMD repo (flat listing)   --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-X.Y/

Source (1) is preferred because it is a proper PIP package index; source (2)
is a flat directory of wheels that AMD publishes for every ROCm release.

The script probes each URL with an HTTP HEAD request and picks the first that
responds.  Use --offline to skip probing and always emit the AMD --find-links
URL (it covers every ROCm release).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PYTORCH_INDEX = "https://download.pytorch.org/whl/rocm{ver}"
AMD_FIND_LINKS = "https://repo.radeon.com/rocm/manylinux/rocm-rel-{ver}/"

TORCH_SPEC = "torch>=2.6"

# ---------------------------------------------------------------------------
# ROCm version detection
# ---------------------------------------------------------------------------


def _read_version_file(rocm_root: Path) -> str | None:
    p = rocm_root / ".info" / "version"
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _detect_rocm_version() -> str:
    """Return the installed ROCm version string (e.g. '7.2.0').

    Search order:
      1. $ROCM_PATH/.info/version  (or $ROCM_HOME)
      2. /opt/rocm/.info/version
      3. ``hipconfig --version`` output
    """
    for env in ("ROCM_PATH", "ROCM_HOME"):
        val = os.environ.get(env)
        if val:
            ver = _read_version_file(Path(val))
            if ver:
                return ver

    ver = _read_version_file(Path("/opt/rocm"))
    if ver:
        return ver

    try:
        out = subprocess.check_output(
            ["hipconfig", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return out
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    raise SystemExit(
        "error: cannot detect ROCm version.\n"
        "Set ROCM_PATH or ensure /opt/rocm/.info/version exists."
    )


# ---------------------------------------------------------------------------
# Version string candidates
# ---------------------------------------------------------------------------

_VER_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")


def _version_candidates(raw: str) -> list[str]:
    """Return version strings to try, most-specific first.

    '7.2.0' -> ['7.2']        (patch 0 is always omitted)
    '6.2.4' -> ['6.2.4', '6.2']
    '7.0.2' -> ['7.0.2', '7.0']
    """
    m = _VER_RE.match(raw)
    if not m:
        raise SystemExit(f"error: cannot parse ROCm version {raw!r}")
    major, minor, patch = m.group(1), m.group(2), m.group(3) or "0"
    short = f"{major}.{minor}"
    if patch == "0":
        return [short]
    full = f"{major}.{minor}.{patch}"
    return [full, short]


# ---------------------------------------------------------------------------
# URL probing
# ---------------------------------------------------------------------------


def _url_exists(url: str, timeout: float = 10) -> bool:
    try:
        req = Request(url, method="HEAD")
        resp = urlopen(req, timeout=timeout)
        return resp.status < 400
    except (URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file (default: stdout)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Skip URL probing; always emit AMD --find-links URL",
    )
    ap.add_argument(
        "--rocm-version",
        default=None,
        help="Override ROCm version instead of auto-detecting",
    )
    args = ap.parse_args()

    raw_ver = args.rocm_version or _detect_rocm_version()
    candidates = _version_candidates(raw_ver)
    print(
        f"Detected ROCm {raw_ver}, candidate version tags: {candidates}",
        file=sys.stderr,
    )

    # Try each candidate against each source.
    sources: list[tuple[str, str]] = []
    for ver in candidates:
        sources.append(("--index-url", PYTORCH_INDEX.format(ver=ver)))
    for ver in candidates:
        sources.append(("--find-links", AMD_FIND_LINKS.format(ver=ver)))

    chosen_flag = chosen_url = None

    if args.offline:
        # In offline mode, pick the first AMD --find-links URL without probing.
        for flag, url in sources:
            if flag == "--find-links":
                chosen_flag, chosen_url = flag, url
                break
    else:
        for flag, url in sources:
            probe = url.rstrip("/") + "/"
            print(f"  probing {probe} ...", end=" ", file=sys.stderr, flush=True)
            if _url_exists(probe):
                print("ok", file=sys.stderr)
                chosen_flag, chosen_url = flag, url
                break
            print("not found", file=sys.stderr)

    if chosen_url is None:
        raise SystemExit(f"error: no PyTorch wheel source found for ROCm {raw_ver}")

    lines = [f"{chosen_flag} {chosen_url}\n", f"{TORCH_SPEC}\n"]

    if args.output == "-":
        sys.stdout.writelines(lines)
    else:
        Path(args.output).write_text("".join(lines))
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
