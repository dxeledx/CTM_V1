"""
Utilities to import from the vendored CTM source tree.

We keep the original CTM implementation under `continuous-thought-machines-main/`
and import selected building blocks (e.g., SuperLinear) without modifying CTM code.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_ctm_on_path() -> Path:
    """
    Add `continuous-thought-machines-main/` to sys.path if needed.
    Returns the resolved path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    ctm_root = repo_root / "continuous-thought-machines-main"
    if not ctm_root.exists():
        raise FileNotFoundError(f"CTM source not found at {ctm_root}")
    ctm_root_str = str(ctm_root)
    if ctm_root_str not in sys.path:
        sys.path.insert(0, ctm_root_str)
    return ctm_root

