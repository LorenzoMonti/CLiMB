"""
CLiMB -- CLustering In Multiphase Boundaries.

A two-phase clustering algorithm for datasets with both known and exploratory
components: Phase 1 anchors constrained clusters to literature seeds (KBound),
Phase 2 looks for structure in whatever Phase 1 rejected.

See ``CLiMB.explain`` for the interpretability layer, which reconstructs both
phases' decisions in closed form rather than fitting a surrogate to them.
"""

__version__ = "0.4.0"
