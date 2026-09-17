import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from . import ExploratoryClusteringBase

class DBSCANExploratory(ExploratoryClusteringBase):
    """
    DBSCAN for exploratory clustering
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN exploratory clustering

        Parameters:
        -----------
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.

        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered
            as a core point. This includes the point itself.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X):
        """
        Perform DBSCAN clustering on X.

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
        return f"DBSCAN"

    def get_parameters(self):
        return f"eps={self.eps}, min_samples={self.min_samples}"

    def explain(self, X, check_fidelity=True):
        """
        Reconstruct the density roles DBSCAN assigned, in closed form.

        DBSCAN has no centroids: membership follows from density connectivity at
        one fixed radius. A point is CORE when at least ``min_samples`` points
        (itself included) lie within ``eps``; a non-core point that still got a
        label is a BORDER point, reached from someone else's neighbourhood;
        everything else is NOISE. Those rules are explicit, so replaying them is
        an exact reconstruction rather than a surrogate.

        The closed-form quantity worth reporting is the **core distance**: the
        distance to the ``min_samples``-th nearest neighbour, i.e. the smallest
        ``eps`` at which this point would become core. ``eps_margin`` is the
        slack to the chosen radius, so it says how far each point is from losing
        core status -- which is the robustness question the (eps, min_samples)
        choice actually raises.

        Columns
        -------
        Exact reconstructions of DBSCAN's rule:
            ``role``, ``is_core``, ``n_neighbors_eps``, ``core_distance``,
            ``eps_margin``.
        Reported from the fitted model:
            ``cluster``.
        Derived geometry, descriptive rather than part of the rule:
            ``distance_to_nearest_core`` -- how firmly a member attaches to its
            cluster's density skeleton. DBSCAN never computes this; it does not
            enter any decision.

        Fidelity
        --------
        The reconstructed core set is checked against sklearn's own
        ``core_sample_indices_`` and ``fidelity_`` records the agreement. This
        is a genuine cross-check: sklearn derives the core set inside its own
        implementation, so agreeing with it is evidence the reconstruction is
        the same rule and not a lookalike.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit_predict``.
        check_fidelity : bool, default=True
            Raise if the reconstructed core set differs from sklearn's.

        Returns
        -------
        pandas.DataFrame
            One row per point.
        """
        X, labels = self._check_explainable(X)
        n = len(X)
        metric = getattr(self.model, "metric", "euclidean")

        neighbours = NearestNeighbors(metric=metric).fit(X)

        # Neighbours within eps, self included -- exactly the count DBSCAN
        # thresholds against min_samples.
        within_eps = neighbours.radius_neighbors(
            X, radius=self.eps, return_distance=False
        )
        n_neighbors_eps = np.array([len(idx) for idx in within_eps])
        is_core = n_neighbors_eps >= self.min_samples

        # Core distance: distance to the min_samples-th nearest neighbour, with
        # the point itself counting as the first. A point is core exactly when
        # this is <= eps.
        k = int(min(self.min_samples, n))
        k_distances, _ = neighbours.kneighbors(X, n_neighbors=k)
        core_distance = k_distances[:, -1]
        eps_margin = self.eps - core_distance

        role = np.empty(n, dtype=object)
        role[:] = "border"
        role[is_core] = "core"
        role[labels == -1] = "noise"

        distance_to_nearest_core = self._distance_to_nearest_core(
            X, labels, is_core, metric
        )

        self.fidelity_ = self._core_set_fidelity(is_core, n)
        if check_fidelity and self.fidelity_ is not None and self.fidelity_ < 1.0:
            disagree = int(round((1.0 - self.fidelity_) * n))
            raise RuntimeError(
                f"Reconstructed core set matches sklearn on only "
                f"{self.fidelity_:.4%} of points ({disagree} of {n} disagree). "
                "The reconstruction is meant to reproduce DBSCAN's own rule, so "
                "this means they have drifted apart -- check that X is the "
                "matrix fit_predict() was given."
            )

        return pd.DataFrame({
            "point_index": np.arange(n),
            "cluster": labels,
            "role": role,
            "is_core": is_core,
            "n_neighbors_eps": n_neighbors_eps,
            "core_distance": core_distance,
            # >0 means the point stays core if eps shrinks by this much.
            "eps_margin": eps_margin,
            "distance_to_nearest_core": distance_to_nearest_core,
        })

    def _distance_to_nearest_core(self, X, labels, is_core, metric):
        """
        Distance from each clustered point to the nearest core point of its own
        cluster -- for a core point, the nearest core *other than itself*.

        Descriptive only: DBSCAN does not compute or use this.
        """
        distance = np.full(len(X), np.nan)

        for cluster in np.unique(labels[labels != -1]):
            members = np.where(labels == cluster)[0]
            cores = members[is_core[members]]
            if len(cores) == 0:
                continue

            k = int(min(2, len(cores)))
            model = NearestNeighbors(n_neighbors=k, metric=metric).fit(X[cores])
            found, _ = model.kneighbors(X[members], n_neighbors=k)

            # Column 0 is the nearest core, which for a core member is itself at
            # distance 0; that member needs column 1 instead. Border members
            # keep column 0, since the nearest core is the answer for them.
            member_is_core = is_core[members]
            nearest = found[:, 0].copy()
            if k > 1:
                nearest[member_is_core] = found[member_is_core, 1]
            else:
                # The cluster's only core is the member itself.
                nearest[member_is_core] = np.nan
            distance[members] = nearest

        return distance

    def _core_set_fidelity(self, is_core, n):
        """
        Agreement between the reconstructed core set and sklearn's own, or None
        when the fitted model does not expose one.
        """
        indices = getattr(self.model, "core_sample_indices_", None)
        if indices is None:
            return None
        sklearn_core = np.zeros(n, dtype=bool)
        sklearn_core[indices] = True
        return float(np.mean(sklearn_core == is_core))
