"""YAML config loader with validation and defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a validated dict.

    Applies defaults for optional fields:
      - seed: 42
      - n_seeds: 1
      - output_dir: "results"
      - save_checkpoints: True
      - save_plots: True
      - verbose: True
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg)}")

    # Apply defaults
    cfg.setdefault("seed", 42)
    cfg.setdefault("n_seeds", 1)
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("save_checkpoints", True)
    cfg.setdefault("save_plots", True)
    cfg.setdefault("verbose", True)

    # Ensure experiment name exists
    if "name" not in cfg:
        cfg["name"] = path.stem

    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Save a config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (base is modified)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
