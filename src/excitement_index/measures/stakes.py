"""Stakes measures — what the match is worth before a ball is kicked.

All three compute on any StatsBomb feed (tier ``core``): they read only the
fixture-sheet stage string (``ctx.stage``, lower-cased) and the team names.
"""
from __future__ import annotations

from .registry import measure

#: WC-2026 host nations (both name spellings that appear in fixture sheets).
HOSTS = {"United States", "USA", "Mexico", "Canada"}

#: Stage substring -> ordinal, checked in order (first hit wins).
_STAGE_ORDINAL = [
    ("group", 0), ("round of 32", 1), ("r32", 1), ("round of 16", 2), ("r16", 2),
    ("quarter", 3), ("qf", 3), ("semi", 4), ("sf", 4),
    ("third", 5), ("3rd", 5), ("final", 5),
]


def stage_ordinal(stage: str) -> float:
    """Stage string -> ordinal via ordered substring match (0 if no key hits)."""
    s = str(stage).lower()
    for key, val in _STAGE_ORDINAL:
        if key in s:
            return float(val)
    return 0.0


@measure("knockout", tier="core")
def knockout(ctx) -> float:
    """1 if the match is any knockout stage, else 0.

    Faithful-port note: "not knockout" is the exact stage string ``"group"``
    (as in the private implementation), so open-data's ``"group stage"``
    label scores 1.0 — the reference behaviour the frozen index was built on.
    """
    return 0.0 if str(ctx.stage).lower() == "group" else 1.0


@measure("elimination_stakes", tier="core")
def elimination_stakes(ctx) -> float:
    """Stage ordinal: group 0, round of 32 = 1, round of 16 = 2, quarterfinal
    3, semifinal 4, third-place/final 5."""
    return stage_ordinal(ctx.stage)


@measure("host_nation", tier="core")
def host_nation(ctx) -> float:
    """1 if the United States, Mexico, or Canada is playing, else 0. Enters
    the composite at fixed scale (not z-scored)."""
    if ctx.row is not None:
        home, away = ctx.row.get("home"), ctx.row.get("away")
    else:
        home, away = ctx.home, ctx.away
    return 1.0 if (home in HOSTS or away in HOSTS) else 0.0
