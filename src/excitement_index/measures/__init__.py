"""The measure catalog.

Importing this package runs every ``@measure`` registration. One module per
sub-family; see ``registry.py`` for how to add your own measure."""
from . import (  # noqa: F401
    aliveness,
    backforth,
    brilliance,
    chances,
    controversy,
    flow,
    keeping,
    prematch,
    resolution,
    stakes,
    timing,
    upset,
)

# (import order is alphabetical; registration order does not affect scores)
from .registry import MatchContext, compute_all, measure, registered_measures  # noqa: F401
