import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from . import ExploratoryClusteringBase

class OPTICSExploratory(ExploratoryClusteringBase):
    """
    OPTICS for exploratory clustering
    """
    def __init__(self, min_samples=5):
        """
        Initialize OPTICS exploratory clustering

        Parameters:
        -----------

        min_samples : int, default=None
            The number of samples in a neighborhood for a point to be considered
            as a core point.
        """
        self.min_samples = min_samples
        self.model = OPTICS(
            min_samples=min_samples
        )

    def fit_predict(self, X):
        """
        Perform OPTICS clustering on X.

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
        return f"OPTICS"

    def get_parameters(self):
        return f"min_samples={self.min_samples}"

    def explain(self, X, check_fidelity=True):
        """
        Explain the reachability ordering OPTICS built.

        OPTICS deliberately refuses to fix a radius. Instead of labelling points
        core or border at one ``eps``, it walks the data in an order that always
        steps to the cheapest reachable point next, and records what that step
        cost. Clusters are then read off the resulting reachability profile as
        valleys between peaks. So there is no core/border split to report here --
        importing DBSCAN's vocabulary would describe a decision this algorithm
        never makes.

        What the profile says, per point:

        ``core_distance``
            The radius at which the point would become core, i.e. the distance
            to its ``min_samples``-th nearest neighbour. This is the only
            genuinely closed-form quantity, and the one that is verified.
        ``reachability``
            What it cost to reach the point from the already-ordered set: the
            larger of the predecessor's core distance and the distance between
            them. Low means the point sits inside a dense region; a spike marks
            a cluster boundary. Infinite for a point that was never reached,
            which is where the walk restarted.
        ``order_position``
            Where the point falls in the reachability plot. Reading the table
            sorted by this column is reading the plot itself.
        ``predecessor``
            The point it was reached from, ``-1`` if none.

        Columns
        -------
        Exact reconstruction of OPTICS's rule:
            ``core_distance``.
        Reported from the fitted model:
            ``cluster``, ``is_noise``, ``reachability``, ``order_position``,
            ``predecessor``.

        Fidelity
        --------
        ``core_distance`` is recomputed from the data and checked against
        sklearn's ``core_distances_``; ``fidelity_`` records the agreement.
        Note this is a narrower guarantee than DBSCAN's: it confirms the core
        distances, not the cluster extraction, which the xi method derives from
        the shape of the reachability profile rather than from a rule that can
        be replayed in closed form.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit_predict``.
        check_fidelity : bool, default=True
            Raise if the recomputed core distances disagree with sklearn's.

        Returns
        -------
        pandas.DataFrame
            One row per point.
        """
        X, labels = self._check_explainable(X)
        n = len(X)

        core_distance = self._reconstruct_core_distances(X)
        reference = np.asarray(self.model.core_distances_)

        # inf == inf must count as agreement, which np.isclose does not give.
        both_infinite = np.isinf(core_distance) & np.isinf(reference)
        agree = both_infinite | np.isclose(core_distance, reference, equal_nan=True)
        self.fidelity_ = float(np.mean(agree))

        if check_fidelity and self.fidelity_ < 1.0:
            raise RuntimeError(
                f"Recomputed core distances match sklearn on only "
                f"{self.fidelity_:.4%} of points "
                f"({int(np.size(agree) - np.count_nonzero(agree))} of {n} disagree). "
                "Check that X is the matrix fit_predict() was given."
            )

        ordering = np.asarray(self.model.ordering_)
        order_position = np.empty(n, dtype=int)
        order_position[ordering] = np.arange(n)

        return pd.DataFrame({
            "point_index": np.arange(n),
            "cluster": labels,
            "is_noise": labels == -1,
            "core_distance": core_distance,
            "reachability": np.asarray(self.model.reachability_),
            "order_position": order_position,
            "predecessor": np.asarray(self.model.predecessor_),
        })

    def _reconstruct_core_distances(self, X):
        """
        Distance to the ``min_samples``-th nearest neighbour, the point itself
        counting as the first, with sklearn's rule that anything beyond
        ``max_eps`` is reported as infinite.
        """
        n = len(X)
        min_samples = self.min_samples

        # sklearn reads min_samples <= 1 as a fraction of the sample count.
        if min_samples is not None and min_samples <= 1:
            k = max(2, int(round(min_samples * n)))
        else:
            k = int(min_samples)
        k = int(min(k, n))

        metric = getattr(self.model, "metric", "minkowski")
        distances, _ = NearestNeighbors(n_neighbors=k, metric=metric).fit(X).kneighbors(X)
        core_distance = distances[:, -1]

        max_eps = getattr(self.model, "max_eps", np.inf)
        return np.where(core_distance > max_eps, np.inf, core_distance)
