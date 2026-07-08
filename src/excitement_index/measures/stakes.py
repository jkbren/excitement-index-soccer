"""Stakes measures — what the match is worth before a ball is kicked.

All three measures compute on any StatsBomb feed (tier ``core``): they read only
the fixture-sheet stage string (``ctx.stage``, lower-cased) and the team names.

Team-name caveat: names reach a measure by two paths that can spell the same
team differently — ``ctx.home`` / ``ctx.away`` (from the event feed) and
``ctx.row["home"]`` / ``ctx.row["away"]`` (from the fixture sheet). ``host_nation``
prefers the row when present and only falls back to the event-feed names, so
``HOSTS`` deliberately carries every spelling that either source uses (e.g. both
``"United States"`` and ``"USA"``).
"""
from __future__ import annotations

from .registry import measure

#: WC-2026 host nations (both name spellings that appear in fixture sheets).
HOSTS = {"United States", "USA", "Mexico", "Canada"}

#: Stage substring -> ordinal, checked in order (first hit wins).
#: DO NOT REORDER: the keys are substrings and several overlap. "quarter"/"qf"/
#: "semi"/"sf"/"third"/"3rd" MUST precede "final", because the strings
#: "quarter-final", "semi-final" and "third-place final" all contain "final" —
#: if "final" were checked first it would swallow them and mislabel the ordinal.
_STAGE_ORDINAL = [
    ("group", 0), ("round of 32", 1), ("r32", 1), ("round of 16", 2), ("r16", 2),
    ("quarter", 3), ("qf", 3), ("semi", 4), ("sf", 4),
    ("third", 5), ("3rd", 5), ("final", 5),
]


def stage_ordinal(stage: str) -> float:
    """Map a stage string to its ordinal by ordered substring match.

    Args:
        stage: Fixture-sheet stage label; case-insensitive.

    Returns:
        The ordinal of the first matching key in ``_STAGE_ORDINAL`` (group 0 up
        to final 5), or 0.0 if no key is a substring of ``stage``. Because the
        match is by substring in list order, any stage containing "final"
        (quarter-, semi-, third-place) resolves to its own earlier key first.
    """
    s = str(stage).lower()
    for key, val in _STAGE_ORDINAL:
        if key in s:
            return float(val)
    return 0.0


@measure("knockout", tier="core")
def knockout(ctx) -> float:
    """Flag whether the match is a knockout-stage game.

    Args:
        ctx: Match context; reads ``ctx.stage``.

    Returns:
        1.0 for any knockout stage, 0.0 only for the group stage.

    The group test is an *exact* string equality against ``"group"``, unlike the
    substring matching in :func:`stage_ordinal`. This is a faithful-port quirk:
    the private reference implementation compared exactly, so open-data's
    ``"group stage"`` label is not equal to ``"group"`` and therefore scores 1.0.
    That is the reference behavior the frozen index was calibrated on and must
    not change. (This is why ``knockout`` and ``elimination_stakes`` can disagree
    on a "group stage" fixture — the latter matches "group" as a substring.)
    """
    return 0.0 if str(ctx.stage).lower() == "group" else 1.0


@measure("elimination_stakes", tier="core")
def elimination_stakes(ctx) -> float:
    """Stage ordinal — how deep in the tournament the match sits.

    Args:
        ctx: Match context; reads ``ctx.stage``.

    Returns:
        The stage ordinal from :func:`stage_ordinal`: group 0, round of 32 = 1,
        round of 16 = 2, quarterfinal 3, semifinal 4, third-place/final 5.

    Uses substring matching (via :func:`stage_ordinal`), so unlike
    :func:`knockout` a ``"group stage"`` label resolves to 0 here.
    """
    return stage_ordinal(ctx.stage)


@measure("host_nation", tier="core")
def host_nation(ctx) -> float:
    """Flag whether a WC-2026 host nation is playing.

    Args:
        ctx: Match context; reads the fixture-sheet ``ctx.row["home"]`` /
            ``ctx.row["away"]`` when a row exists, otherwise the event-feed
            ``ctx.home`` / ``ctx.away``.

    Returns:
        1.0 if the United States, Mexico, or Canada is either team, else 0.0.
        Enters the composite at fixed scale (not z-scored).

    The two name sources can spell a team differently, which is why ``HOSTS``
    lists every spelling either path produces (see the module docstring).
    """
    if ctx.row is not None:
        # Prefer the curated fixture-sheet names; fall back to the event feed.
        home, away = ctx.row.get("home"), ctx.row.get("away")
    else:
        home, away = ctx.home, ctx.away
    return 1.0 if (home in HOSTS or away in HOSTS) else 0.0
