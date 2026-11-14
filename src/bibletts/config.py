"""Loading the YAML config files in configs/."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """Read a YAML config and apply nested overrides.

    Overrides use dotted keys, for example {"paths.local_root": "/tmp/data"}.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for dotted, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write a config next to outputs, so results can be traced to settings."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)
