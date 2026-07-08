"""The measure catalog.

Importing this package imports every family module, and each module's
``@measure`` decorators run on import — so ``registered_measures()`` is only
complete once this package has been imported. There is one module per measure
sub-family (aliveness, backforth, controversy, prematch, stakes, ...); see
``registry.py`` for how a measure plugs into the pipeline and how to add one.
"""
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
