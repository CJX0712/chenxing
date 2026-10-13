#!/usr/bin/env python
"""Create an isolated venv and install Chenxing (晨星) editable + dev extras.

Idempotent: if .venv already exists it is reused. Requires network for the
first `pip install`. After this, `python -m chenxing.cli selfcheck` and
`pytest` both work.

Usage:
  python scripts/install.py          # create .venv and install
  python scripts/install.py --no-dev # skip pytest extra
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dev", action="store_true")
    args = ap.parse_args()

    if not VENV.exists():
        print(f"[install] creating venv at {VENV}")
        venv.create(VENV, with_pip=True)
    else:
        print(f"[install] reusing existing venv at {VENV}")

    if sys.platform.startswith("win"):
        pip = VENV / "Scripts" / "pip.exe"
        py = VENV / "Scripts" / "python.exe"
    else:
        pip = VENV / "bin" / "pip"
        py = VENV / "bin" / "python"

    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True)
    extra = "" if args.no_dev else "[dev]"
    subprocess.run([str(pip), "install", "-e", f".{extra}"], cwd=str(REPO), check=True)

    print("[install] done.")
    print(f"  run checks : {py} -m chenxing.cli selfcheck")
    print(f"  tests      : {py} -m pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
