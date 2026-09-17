"""
Interpretability layer for CLiMB.

CLiMB's decisions are not a black box, and this layer does not treat them as
one. Phase 1 applies two explicit gates plus seed forcing; DBSCAN applies a
radius rule. Those are rules that can be replayed, so the explanation is a
reconstruction of the model in closed form -- not a surrogate fitted to imitate
it, as SHAP or LIME would build. A reconstruction either reproduces the model's
own output or it is broken, so it is checked, every time, and it raises rather
than quietly returning a table that does not describe the model.

Not everything worth reporting is a reconstruction, and the package keeps the
two apart rather than letting one borrow the other's credibility:

**Exact reconstructions** -- replayed from the fitted model and verified:
    ``KBound.decision_path``       gates and labels; fidelity vs ``labels_``
    ``KBound.feature_attribution`` additive split of the metric's quadratic form
    ``DBSCANExploratory.explain``  roles; fidelity vs ``core_sample_indices_``
    ``OPTICSExploratory.explain``  core distances; fidelity vs ``core_distances_``
    ``CLiMB.explain``              both phases joined, with both checks

**Reported model state** -- exact, but nothing independent to check it against:
    ``HDBSCANExploratory.explain`` membership by stability; ``fidelity_`` is None

**Descriptive statistics** -- computed after the fact, by our rules, unverified:
    ``CLiMB.explain.descriptive``  contrastive signatures, effect sizes

The plot helpers render whatever they are given and cannot tell the categories
apart, which is why their labels are the caller's to supply.
"""
from .descriptive import cluster_signatures, plot_cluster_signatures
from .plots import (
    plot_feature_attribution,
    plot_gate_accounting,
    plot_heatmap,
    plot_margin_map,
    plot_roles,
)

__all__ = [
    "cluster_signatures",
    "plot_cluster_signatures",
    "plot_feature_attribution",
    "plot_gate_accounting",
    "plot_heatmap",
    "plot_margin_map",
    "plot_roles",
]
