"""
Descriptive statistics about a clustering -- **not** reconstructions of it.

Everything in this module is computed *after* the fact, from the labels and the
data, by rules that are ours rather than the model's. Nothing here replays a
decision, nothing is verified against the fitted model, and no function here
sets ``fidelity_``. A contrastive signature says "these clusters differ on this
feature"; it does not say "this is why the point was assigned", and reading it
as though it did is the mistake this module is laid out to prevent.

The exact reconstructions live elsewhere and are checked against the model:
``KBound.decision_path``, ``KBound.feature_attribution``,
``<Algorithm>.explain`` and ``CLiMB.explain``.
"""
import warnings

import numpy as np
import pandas as pd

from ..utils.util import cohens_d


def cluster_signatures(X, labels, reference=None, feature_names=None,
                       include_noise=False):
    """
    Per-cluster contrastive signature: how each feature separates a cluster
    from a reference population.

    **Descriptive, post-hoc.** This measures a property of the result, not the
    rule that produced it. Two clusters can differ sharply on a feature the
    model never consulted, and a feature the model leaned on heavily can show a
    small effect here. Do not read these numbers as attributions -- for the
    exact per-feature split of what placed a point where it is, use
    ``KBound.feature_attribution``, which decomposes the model's own metric.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix the labels refer to, row-aligned with ``labels``.
    labels : array-like of shape (n_samples,)
        Cluster labels, ``-1`` for noise.
    reference : array-like of shape (n_reference, n_features), optional
        Population to contrast each cluster against. Defaults to the noise
        points of ``X``, i.e. the surrounding field. Passing the full matrix
        instead gives an absolute contrast rather than a local one; the two
        answer different questions and are worth reporting side by side.
    feature_names : sequence of str, optional
        Defaults to ``feature_0 ... feature_{d-1}``.
    include_noise : bool, default=False
        Whether to also produce a row for the noise label.

    Returns
    -------
    pandas.DataFrame
        Clusters as rows, features as columns, effect sizes as values. Empty if
        there is nothing to contrast.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional; got shape {X.shape}.")
    if len(X) != len(labels):
        raise ValueError(
            f"X has {len(X)} rows but labels has {len(labels)} entries."
        )

    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(X.shape[1])]
    feature_names = list(feature_names)
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries for "
            f"{X.shape[1]} columns of X."
        )

    default_reference = reference is None
    if default_reference:
        reference = X[labels == -1]
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != X.shape[1]:
        raise ValueError(
            f"reference must have {X.shape[1]} columns; got shape {reference.shape}."
        )

    # An effect size needs at least two values a side. Saying so beats handing
    # back a table of NaN and leaving the caller to work out why.
    if len(reference) < 2:
        warnings.warn(
            f"reference has {len(reference)} row(s), so every effect size will be "
            "NaN: at least 2 are needed."
            + (" The default reference is the noise points of X, and this "
               "clustering left almost none; pass an explicit reference "
               "population, such as the full matrix, for an absolute contrast."
               if default_reference else ""),
            RuntimeWarning,
            stacklevel=2,
        )

    clusters = sorted(c for c in np.unique(labels) if include_noise or c != -1)
    rows = {}
    for cluster in clusters:
        members = X[labels == cluster]
        rows[int(cluster)] = [
            cohens_d(members[:, j], reference[:, j])
            for j in range(X.shape[1])
        ]

    table = pd.DataFrame.from_dict(rows, orient="index", columns=feature_names)
    table.index.name = "cluster"
    return table


def plot_cluster_signatures(signatures, cluster_names=None, ax=None, save_path=None,
                            title="What distinguishes each cluster (descriptive)"):
    """
    Heatmap of ``cluster_signatures``.

    The colourbar names the statistic and the title says "descriptive" on
    purpose: the same heatmap shape is used for the exact attribution, and a
    reader who cannot tell them apart will assume the stronger guarantee.

    Parameters
    ----------
    signatures : pandas.DataFrame
        As returned by ``cluster_signatures``.
    cluster_names : dict, optional
        ``{cluster index: display name}``.
    """
    from .plots import plot_heatmap

    labels = [(cluster_names or {}).get(int(c), f"cluster {int(c)}")
              for c in signatures.index]
    return plot_heatmap(
        signatures.to_numpy(),
        list(signatures.columns),
        labels,
        colourbar_label="effect size (median-based, pooled SD)\ndescriptive statistic, not an attribution",
        title=title,
        ax=ax,
        save_path=save_path,
    )
