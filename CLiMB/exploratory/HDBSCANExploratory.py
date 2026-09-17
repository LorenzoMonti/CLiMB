import numpy as np
import pandas as pd
import hdbscan
from . import ExploratoryClusteringBase

class HDBSCANExploratory(ExploratoryClusteringBase):
    """
    HDBSCAN for exploratory clustering
    """
    def __init__(self, min_cluster_size=5, min_samples=None):
        """
        Initialize HDBSCAN exploratory clustering

        Parameters:
        -----------
        min_cluster_size : int, default=5
            The minimum size of clusters to be considered.

        min_samples : int, default=None
            The number of samples in a neighborhood for a point to be considered
            as a core point.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

    def fit_predict(self, X):
        """
        Perform HDBSCAN clustering on X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to cluster.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. Noisy samples are given the label -1.
        """
        return self.model.fit_predict(X)

    def get_name(self):
        return f"HDBSCAN"

    def get_parameters(self):
        return f"min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}"

    def explain(self, X, check_fidelity=True):
        """
        Report how strongly each point belongs, and how durable its cluster is.

        HDBSCAN has neither a radius nor a core/border split. It builds a
        hierarchy over every density scale at once, then keeps the clusters that
        survive longest as the scale varies. Membership is therefore a matter of
        degree -- how far a point persists with its cluster before the cluster
        dissolves -- not a gate a point passes or fails.

        **This explanation is not a reconstruction.** Unlike Phase 1, DBSCAN or
        the OPTICS core distances, HDBSCAN's labels come from the stability of a
        condensed tree, and there is no closed form to replay and check. Every
        column below is a quantity the fitted model computed and reports; none
        is independently rebuilt, and ``fidelity_`` is ``None`` to say so rather
        than carry a 1.0 that would mean nothing. A surrogate fitted here to
        manufacture a number would be exactly the kind of thing Phase 1 avoids.

        Columns
        -------
        Reported from the fitted model:
            ``cluster``, ``is_noise``, ``membership_probability``,
            ``outlier_score``, ``cluster_persistence``.

        ``membership_probability``
            How deep inside its cluster the point lasted, on 0-1. Noise is 0.
        ``outlier_score``
            GLOSH: how anomalous the point is relative to the local density it
            sits in. Independent of the label, so a clustered point can still
            score high.
        ``cluster_persistence``
            A per-cluster score copied onto its members: how much of the density
            range the cluster survived. This is the quantity that decided the
            cluster existed at all, so it belongs on the report, but it says
            nothing about the individual point. ``NaN`` for noise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit_predict``.
        check_fidelity : bool, default=True
            Accepted for interface compatibility and ignored: there is no
            reconstruction here to verify.

        Returns
        -------
        pandas.DataFrame
            One row per point.
        """
        X, labels = self._check_explainable(X)
        n = len(X)

        # No closed-form rule to replay, so nothing to verify.
        self.fidelity_ = None

        persistence = np.asarray(
            getattr(self.model, "cluster_persistence_", []), dtype=float
        )
        point_persistence = np.full(n, np.nan)
        clustered = labels != -1
        if persistence.size and clustered.any():
            valid = clustered & (labels < persistence.size)
            point_persistence[valid] = persistence[labels[valid]]

        return pd.DataFrame({
            "point_index": np.arange(n),
            "cluster": labels,
            "is_noise": labels == -1,
            "membership_probability": self._model_array("probabilities_", n),
            "outlier_score": self._model_array("outlier_scores_", n),
            "cluster_persistence": point_persistence,
        })

    def _model_array(self, name, n):
        """Per-point model output, or NaN if this build does not expose it."""
        values = getattr(self.model, name, None)
        if values is None or len(values) != n:
            return np.full(n, np.nan)
        return np.asarray(values, dtype=float)
