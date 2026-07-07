"""Configuration loading for the index.

The entire method — taxonomy, weights, signs, deduction parameters, display
scale — lives in a YAML file (``config/v14.yaml`` ships the frozen v1.4
constants). Users experiment by passing overrides:

    score_matches(..., config="my_variant.yaml")
    score_matches(..., config={"taxes": {"dead_rubber_k": 0.2}})   # deep-merged
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "v14.yaml"

ConfigLike = Union[None, str, Path, Mapping]


def _deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge,
    everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config: ConfigLike = None) -> dict:
    """Resolve a config argument to a validated dict.

    ``None`` -> the shipped v1.4 defaults; a path -> that YAML file (merged over
    the defaults, so a partial file works); a mapping -> deep-merged overrides."""
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if config is None:
        pass
    elif isinstance(config, (str, Path)):
        with open(config) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    elif isinstance(config, Mapping):
        cfg = _deep_merge(cfg, config)
    else:
        raise TypeError(f"config must be None, a path, or a mapping — got {type(config)}")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail loudly on the mistakes people actually make when editing the YAML."""
    taxonomy, weights = cfg["taxonomy"], cfg["weights"]
    missing = set(taxonomy) ^ set(weights)
    if missing:
        raise ValueError(f"taxonomy and weights must cover the same families; mismatch: {missing}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"family weights must sum to 1 (got {total:.6f}) — "
                         "re-normalize after editing")
    all_measures = {m for feats in taxonomy.values() for m in feats}
    for m in cfg.get("negative_signs", []):
        if m not in all_measures:
            raise ValueError(f"negative_signs lists unknown measure {m!r}")
    bucketed = {fam for fams in cfg["buckets"].values() for fam in fams}
    if bucketed != set(taxonomy):
        raise ValueError("buckets must partition exactly the taxonomy families; "
                         f"difference: {bucketed ^ set(taxonomy)}")
    for key in ("dead_rubber_k", "aliveness_delta"):
        v = cfg["taxes"][key]
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"taxes.{key} must be in [0, 1] (got {v})")
