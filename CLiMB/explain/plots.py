"""
Plot helpers for the explanation tables.

Every function is domain-neutral: axis and colourbar text is passed in, never
built from assumptions about what the features mean. Nothing is written to disk
or shown unless the caller asks, and each function returns its Axes so the
caller can keep composing.

These render whatever table they are handed. They do not know, and cannot check,
whether a column is an exact reconstruction or a descriptive statistic -- the
labels the caller supplies are what tells the reader. See ``CLiMB.explain`` for
that distinction.
"""
import numpy as np
import matplotlib.pyplot as plt


GATE_COLOURS = {
    "assigned": "#2ca02c",
    "seed_forced": "#1f77b4",
    "rejected_distance": "#d62728",
    "rejected_density": "#ff7f0e",
}
GATE_ORDER = ("assigned", "seed_forced", "rejected_distance", "rejected_density")


def _finish(ax, save_path, dpi=300):
    if save_path:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return ax


def plot_gate_accounting(explanation, column="gate", ax=None, save_path=None,
                         title="Which mechanism decided each point",
                         ylabel="Number of points"):
    """
    Bar chart of how many points each decision mechanism accounted for.

    Parameters
    ----------
    explanation : pandas.DataFrame
        Any table with a categorical decision column, e.g. the ``gate`` column
        of ``KBound.decision_path`` or ``phase1_gate`` of ``CLiMB.explain``.
    column : str, default='gate'
        The categorical column to count.
    ax : matplotlib Axes, optional
    save_path : str, optional
    title, ylabel : str
        Wording is the caller's, so it can name the domain's objects.
    """
    counts = explanation[column].value_counts()
    ordered = [gate for gate in GATE_ORDER if gate in counts.index]
    ordered += [gate for gate in counts.index if gate not in GATE_ORDER]
    counts = counts.reindex(ordered)
    total = int(counts.sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(counts.index, counts.values,
                  color=[GATE_COLOURS.get(gate, "gray") for gate in counts.index])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value}\n({value / total * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", labelrotation=15)
    ax.figure.tight_layout()
    return _finish(ax, save_path)


def plot_margin_map(explanation, x, y, value, xlabel=None, ylabel=None,
                    colourbar_label=None, noise_mask=None, cmap="viridis",
                    symmetric=False, ax=None, save_path=None,
                    title="Assignment robustness"):
    """
    Scatter of two features coloured by how much slack a point has.

    Serves both margin questions: Phase 1's distance to the decision boundary
    and Phase 2's ``eps_margin``. Pass ``symmetric=True`` with a diverging
    colormap when zero is the meaningful midpoint, as it is for ``eps_margin``
    where the sign flips core status.

    Parameters
    ----------
    explanation : pandas.DataFrame
    x, y : str
        Feature columns to place the points.
    value : str
        Column to colour by.
    xlabel, ylabel, colourbar_label : str, optional
        Axis and colourbar text, including units. Defaults to the column names:
        the package has no idea what the features measure, so anything more
        specific has to come from the caller.
    noise_mask : array-like of bool, optional
        Points to draw greyed out behind the rest, e.g. unclustered points.
    symmetric : bool, default=False
        Centre the colour scale on zero, clipped at the 98th percentile of
        \\|value\\| so a single extreme point cannot flatten the scale.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    data = explanation
    if noise_mask is not None:
        noise_mask = np.asarray(noise_mask, dtype=bool)
        noise = explanation[noise_mask]
        data = explanation[~noise_mask]
        if len(noise):
            ax.scatter(noise[x], noise[y], c="lightgray", s=8, alpha=0.4, label="noise")

    limits = {}
    if symmetric:
        extent = np.nanpercentile(np.abs(data[value]), 98)
        extent = float(extent) if np.isfinite(extent) and extent > 0 else 1.0
        limits = {"vmin": -extent, "vmax": extent}

    points = ax.scatter(data[x], data[y], c=data[value], cmap=cmap,
                        s=18, ec="black", lw=0.15, **limits)
    bar = ax.figure.colorbar(points, ax=ax)
    bar.set_label(colourbar_label or value, fontsize=11)

    ax.set_xlabel(xlabel or x, fontsize=12)
    ax.set_ylabel(ylabel or y, fontsize=12)
    ax.set_title(title, fontsize=13)
    if noise_mask is not None and noise_mask.any():
        ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.figure.tight_layout()
    return _finish(ax, save_path)


def plot_roles(explanation, x, y, role_column="role", cluster_column="cluster",
               xlabel=None, ylabel=None, cluster_names=None, ax=None,
               save_path=None, title="Density skeleton (large markers = core)"):
    """
    Scatter of the density roles, one colour per cluster.

    Core points are drawn larger with an edge, border points smaller and plain,
    so the density skeleton of each cluster is visible. Only meaningful for an
    algorithm that has a core/border split -- DBSCAN. OPTICS and HDBSCAN do not
    produce one, by design, and their tables have no ``role`` column to pass.

    Parameters
    ----------
    cluster_names : dict, optional
        ``{cluster index: display name}``. Unnamed clusters fall back to
        ``cluster <i>``; naming them is the caller's business.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    noise = explanation[explanation[cluster_column] == -1]
    clustered = explanation[explanation[cluster_column] != -1]
    if len(noise):
        ax.scatter(noise[x], noise[y], c="lightgray", s=6, alpha=0.4, label="noise")

    clusters = sorted(clustered[cluster_column].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))
    for colour, cluster in zip(palette, clusters):
        members = clustered[clustered[cluster_column] == cluster]
        is_core = members[role_column] == "core"
        name = (cluster_names or {}).get(int(cluster), f"cluster {int(cluster)}")
        ax.scatter(members.loc[~is_core, x], members.loc[~is_core, y],
                   color=colour, s=14, marker="o", alpha=0.7)
        ax.scatter(members.loc[is_core, x], members.loc[is_core, y],
                   color=colour, s=45, marker="o", ec="black", lw=0.4,
                   label=f"{name} (n={len(members)})")

    ax.set_xlabel(xlabel or x, fontsize=12)
    ax.set_ylabel(ylabel or y, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.figure.tight_layout()
    return _finish(ax, save_path)


def plot_heatmap(matrix, column_labels, row_labels, colourbar_label,
                 title=None, cmap="RdBu_r", vmin=None, vmax=None, symmetric=True,
                 ax=None, save_path=None, annotate=True):
    """
    Annotated heatmap of a clusters-by-features matrix.

    A rendering primitive with no opinion about what it is showing: it serves
    both the exact per-feature attribution and the descriptive contrastive
    signatures. ``colourbar_label`` is required precisely because the figure
    must say which of the two it is -- the plot cannot tell.

    Parameters
    ----------
    matrix : array-like of shape (n_rows, n_columns)
    column_labels, row_labels : sequence of str
    colourbar_label : str
        What the numbers are. Required.
    symmetric : bool, default=True
        Centre the scale on zero using the largest magnitude present, unless
        ``vmin``/``vmax`` are given.
    """
    matrix = np.asarray(matrix, dtype=float)
    if vmin is None and vmax is None and symmetric:
        # An all-NaN matrix is a real outcome (an effect size needs enough data
        # on both sides), so fall back to a unit scale instead of warning from
        # nanmax and handing imshow a NaN limit.
        finite = matrix[np.isfinite(matrix)]
        extent = float(np.max(np.abs(finite))) if finite.size else 0.0
        extent = extent if extent > 0 else 1.0
        vmin, vmax = -extent, extent

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.6 * len(row_labels))))

    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(column, row, "--" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", color="black", fontsize=9)

    bar = ax.figure.colorbar(image, ax=ax)
    bar.set_label(colourbar_label, fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    ax.figure.tight_layout()
    return _finish(ax, save_path)


def plot_feature_attribution(attribution, feature_names, cluster_column="reference_cluster",
                             kind="disc", cluster_names=None, ax=None, save_path=None,
                             title="Per-cluster signature (why these members, not the runner-up)"):
    """
    Heatmap of the exact per-feature attribution, aggregated per cluster.

    Takes the table from ``KBound.feature_attribution``. The cell is the
    **median** share across a cluster's members, not the mean: a forced seed
    whose geometry disagrees with its supervised label sits at a near-zero
    margin, and dividing by that margin makes its per-row fraction explode. One
    such point would dominate a mean.

    Parameters
    ----------
    kind : {'disc', 'contrib'}, default='disc'
        ``'disc'`` shows the share of the margin favouring the assigned cluster
        over the runner-up; ``'contrib'`` shows the share of the distance to the
        assigned centroid.
    """
    if kind not in ("disc", "contrib"):
        raise ValueError(f"kind must be 'disc' or 'contrib'; got {kind!r}.")

    feature_names = list(feature_names)
    clusters = sorted(attribution[cluster_column].unique())
    matrix = np.array([
        [attribution.loc[attribution[cluster_column] == cluster,
                         f"{kind}_{name}_frac"].median()
         for name in feature_names]
        for cluster in clusters
    ])

    labels = [(cluster_names or {}).get(int(c), f"cluster {int(c)}") for c in clusters]
    colourbar = ("median discriminative share\n(fraction of margin² favouring this cluster)"
                 if kind == "disc" else
                 "median share of squared distance\nto the assigned centroid")
    return plot_heatmap(matrix, feature_names, labels, colourbar,
                        title=title, vmin=-1, vmax=1, ax=ax, save_path=save_path)
