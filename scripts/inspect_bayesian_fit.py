#!/usr/bin/env python3
"""Backward-compatible wrapper for inspecting Bayesian fit diagnostics.

This delegates to the module ``inspect`` CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure local package import works when run from repository checkout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pyreplab_harness import outcome_model as om


def main() -> int:
    return om.main(["inspect", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
