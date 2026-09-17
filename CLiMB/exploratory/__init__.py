from abc import ABC, abstractmethod

import numpy as np


class ExploratoryClusteringBase(ABC):
    """
    Base abstract class for exploratory clustering algorithms
    """

    @abstractmethod
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to cluster.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        Returns the name of the clustering algorithm
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Returns the parameters of the clustering algorithm
        """
        pass

    @abstractmethod
    def explain(self, X, check_fidelity=True):
        """
        Per-point account of how this algorithm reached its clustering.

        Deliberately **not** generalised across implementations. Phase 2's
        algorithms do not share a decision structure: DBSCAN assigns roles from
        a fixed radius, OPTICS orders points by reachability without committing
        to one radius at all, and HDBSCAN decides membership by cluster
        stability. Forcing them into common columns would invent a
        core/border distinction for algorithms that have none, so each
        implementation returns its own columns and documents its own meaning.

        What *is* common is the honesty contract. Implementations state which
        columns are exact reconstructions of the model's own rule and which are
        quantities the fitted model reports. Where a closed-form reconstruction
        exists, it is verified against the fitted model and ``fidelity_`` is
        set; where none exists, that is documented rather than approximated.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit_predict``.
        check_fidelity : bool, default=True
            Where the implementation can verify its reconstruction against the
            fitted model, raise on any disagreement instead of returning a
            table that quietly does not describe the model.

        Returns
        -------
        pandas.DataFrame
            One row per point of ``X``. The ``cluster`` column carries the
            algorithm's **own** numbering, starting at 0, because that is what
            the algorithm produced and this object knows nothing of CLiMB.

            When Phase 2 runs inside CLiMB those numbers are offset so they do
            not collide with Phase 1's, so ``cluster`` here is *not* comparable
            with ``CLiMB.get_labels()``. The two overlap numerically, so a
            ``{label: name}`` map built for the whole clustering will match
            these silently rather than raise. ``CLiMB.explain`` keeps both, as
            ``phase2_label`` and ``phase2_algorithm_cluster``.
        """
        pass

    def _check_explainable(self, X):
        """
        Shared guard: the model must be fitted, and ``X`` must be the matrix it
        was fitted on. Results are stored by row position, so another matrix
        would produce a table that looks valid and describes nothing.
        """
        labels = getattr(self.model, "labels_", None)
        if labels is None:
            raise RuntimeError(
                f"{self.get_name()}.explain() needs a fitted model; "
                "call fit_predict(X) first."
            )

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional; got shape {X.shape}.")
        if len(X) != len(labels):
            raise ValueError(
                f"{self.get_name()}.explain() expects the matrix passed to "
                f"fit_predict(): got {len(X)} rows, fitted on {len(labels)}."
            )
        return X, np.asarray(labels)
