"""
Lightweight config helpers (YAML/JSON -> dataclasses).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text()) or {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def dataclass_from_dict(default_obj: T, overrides: dict[str, Any]) -> T:
    """
    Merge overrides into a dataclass (recursively) using dict representation.
    """
    if not is_dataclass(default_obj):
        raise TypeError("default_obj must be a dataclass instance")
    base = asdict(default_obj)
    merged = deep_update(base, overrides)
    return type(default_obj)(**merged)  # type: ignore[arg-type]

