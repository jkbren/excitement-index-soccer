"""Configuration loading for the index.

The entire method — taxonomy, weights, signs, deduction parameters, display
scale — lives in a YAML file (``config/default.yaml`` ships the frozen
constants). Users experiment by passing overrides:

    score_matches(..., config="my_variant.yaml")
    score_matches(..., config={"taxes": {"dead_rubber_k": 0.2}})   # deep-merged
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

ConfigLike = Union[None, str, Path, Mapping]


def _deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Args:
        base: The base config dict; never mutated (a deep copy is returned).
        override: Mapping whose keys are layered on top of ``base``.

    Returns:
        A new dict where nested dict values are merged key-by-key and every other
        value (scalars, lists) replaces the base value outright.

    Lists replace rather than concatenate so a partial override file can restate a
    whole family or anchor list without inheriting stale entries.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config: ConfigLike = None) -> dict:
    """Resolve a config argument to a validated config dict.

    Args:
        config: The config source. ``None`` uses the shipped defaults; a ``str`` or
            ``Path`` is read as a YAML file and merged over the defaults (so a
            partial file works); a ``Mapping`` is deep-merged over the defaults.

    Returns:
        The fully merged config dict, after :func:`validate_config` has passed.

    The shipped defaults are always loaded first so every returned config is complete;
    overrides only need to name the keys they change. A bad type raises ``TypeError``.
    """
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
    """Fail loudly on the mistakes people actually make when editing the YAML.

    Args:
        cfg: A merged config dict to check in place. Raises ``ValueError`` on the
            first structural problem found; returns ``None`` when the config passes.

    Checks, in order:
        - ``taxonomy`` and ``weights`` cover exactly the same set of families,
        - family weights sum to 1 (within 1e-6, since they are the mixing weights of
          the family means),
        - every ``negative_signs`` entry names a measure that exists in the taxonomy,
        - ``buckets`` partition exactly the taxonomy families (so bucket contributions
          sum to the raw score),
        - each tax coefficient lies in [0, 1].

    Note: the ``scale`` block is not validated here. Malformed scale settings (e.g.
    unequal-length or non-increasing ``anchor_percentiles`` / ``anchor_display_values``,
    percentiles outside [0, 1], a first display value <= 0, or a last value >= 10)
    surface only as NaN/inf values far downstream in
    :func:`excitement_index.scale.fit_scale_map`.
    """
    taxonomy, weights = cfg["taxonomy"], cfg["weights"]
    # Symmetric difference is empty only when the two family sets match exactly.
    missing = set(taxonomy) ^ set(weights)
    if missing:
        raise ValueError(f"taxonomy and weights must cover the same families; mismatch: {missing}")
    total = sum(weights.values())
    # Weights mix the family means, so they must be a partition of unity (1e-6 slack).
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"family weights must sum to 1 (got {total:.6f}) — "
                         "re-normalize after editing")
    all_measures = {m for feats in taxonomy.values() for m in feats}
    for m in cfg.get("negative_signs", []):
        if m not in all_measures:
            raise ValueError(f"negative_signs lists unknown measure {m!r}")
    bucketed = {fam for fams in cfg["buckets"].values() for fam in fams}
    # Buckets must be an exact partition so bucket_* columns sum to raw.
    if bucketed != set(taxonomy):
        raise ValueError("buckets must partition exactly the taxonomy families; "
                         f"difference: {bucketed ^ set(taxonomy)}")
    for key in ("dead_rubber_k", "aliveness_delta"):
        v = cfg["taxes"][key]
        # Both taxes are fractional multipliers; outside [0, 1] they would invert or
        # over-subtract.
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"taxes.{key} must be in [0, 1] (got {v})")
