import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from ..utils.util import hungarian_match

class KBound:
    """
    Constrained k-means anchored to literature seed points.

    The attributes below are set by ``fit`` and are public, stable API.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Raw cluster index per point, ``-1`` for points rejected by a gate.
        These are the labels CLiMB's Phase 1 exposes as ``constrained_labels``.
    centroids_ : ndarray of shape (n_clusters, n_features)
        Final centroid estimate. Use it to plot, or to place new points.
    decision_centroids_ : ndarray of shape (n_clusters, n_features)
        The centroids ``labels_`` was actually computed against. Identical to
        ``centroids_`` whenever the fit converged; they differ by one update
        when the loop exits on ``max_iter``, because the centroid refresh at
        the end of the loop body is skipped by the convergence ``break`` but
        not by exhaustion. Reconstructing decisions requires this one --
        ``centroids_`` would describe a decision the algorithm never took.
    point_densities_, local_density_ : ndarray of shape (n_samples,)
        Normalised Gaussian-kernel local density, the input to the density
        gate. ``local_density_`` is a read-only alias.
    seed_indices_ : dict of {int: list of int}
        ``{cluster index: row indices of X pinned to it}``, populated only for
        dictionary seeds. These are the points ``_post_process_seeds`` forces
        into their cluster after the gates have run, so their label is copied
        rather than derived.
    unassigned_mask_ : ndarray of shape (n_samples,)
        True where a gate rejected the point (before seed forcing).
    n_iter_ : int
        Iterations actually run.
    converged_ : bool
        Whether the loop exited on the convergence test rather than ``max_iter``.
    fidelity_ : float
        Set by ``decision_path``: fraction of non-seed points whose label the
        closed-form reconstruction reproduces. Expected to be exactly 1.0.
    """

    def __init__(
        self,
        n_clusters,
        seeds=None,
        max_iter=300,
        density_threshold=0.5,
        distance_threshold=2.0,
        radial_threshold=1.0,
        convergence_tolerance=0.1,
        distance_metric='euclidean',
        metric_params=None
    ):
        """
        Initialize 3D KBound (Constrained K-Means)

        Parameters:
        - n_clusters: Number of target clusters
        - seeds: Dictionary of {centroid_point: [seed_points]} where centroid_point is the initial centroid location and seed_points is a list of points to be associated with this centroid.
                 Only the dictionary form pins its seed points to a cluster (see seed_indices_); the other forms
                 just place the initial centroids.
                 Alternatively, a sequence of initial centroid points (list, tuple or ndarray), or None for
                 random initialization. Any other type raises TypeError rather than falling back to random.
        - max_iter: Maximum iterations for convergence
        - density_threshold: Minimum local density required for cluster assignment
        - distance_threshold: Maximum distance from centroid for point retention
        - radial_threshold: Maximum radial centroid's distance
        - convergence_tolerance: defines the minimum movement required for centroids before the algorithm stops
        - distance_metric: Distance metric to use ('euclidean', 'mahalanobis', 'custom'). Default: 'euclidean'
        - metric_params: Dictionary of parameters for the chosen distance metric.
                        For 'mahalanobis', it should contain 'VI' (inverse covariance matrix).
                        For 'custom', it should contain 'func' (the custom distance function).
        """
        self.n_clusters = n_clusters
        self.seeds = seeds
        self.max_iter = max_iter
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.radial_threshold = radial_threshold
        self.convergence_tolerance = convergence_tolerance
        self.mapped_labels_ = list()
        self.distance_metric = distance_metric
        self.metric_params = metric_params
        self.seed_indices_ = {} # Store indices of seed points for each cluster


    def _compute_local_density(self, X, sigma=None):
        """
        Calculate 3D local point density using Gaussian kernel
        (Using Euclidean distance for density calculation as it's about local proximity)

        Returns:
        - Normalized local density for each point
        """
        distances = squareform(pdist(X, metric='euclidean'))

        if sigma is None:
            sigma = np.mean(distances)

        density = np.sum(
            np.exp(-0.5 * (distances / sigma) ** 2),
            axis=1
        )
        return density / np.max(density)


    def _initialize_centroids(self, X):
        """
        Intelligent centroid initialization strategy, now handles dictionary seeds
        """
        if self.seeds is None:
            return X[np.random.choice(len(X), self.n_clusters, replace=False)]

        if isinstance(self.seeds, dict):
            initial_centroid_locations = list(self.seeds.keys())
            num_initial_centroids = len(initial_centroid_locations)
            centroids = np.array(initial_centroid_locations)

            if num_initial_centroids >= self.n_clusters:
                distances = self._cdist_custom(centroids, centroids)
                np.fill_diagonal(distances, np.inf)

                selected_centroid_indices = []
                while len(selected_centroid_indices) < self.n_clusters:
                    if not selected_centroid_indices:
                        selected_centroid_indices.append(0)
                    else:
                        candidates = [
                            i for i in range(num_initial_centroids)
                            if i not in selected_centroid_indices
                        ]
                        max_min_distance = -1
                        best_candidate = None

                        for candidate in candidates:
                            min_dist = min(
                                self._cdist_custom(
                                    [centroids[candidate]],
                                    [centroids[idx] for idx in selected_centroid_indices]
                                ).min(),
                                0
                            )
                            if min_dist > max_min_distance:
                                max_min_distance = min_dist
                                best_candidate = candidate
                        selected_centroid_indices.append(best_candidate)
                return centroids[selected_centroid_indices]

            elif num_initial_centroids < self.n_clusters:
                remaining_centroids_needed = self.n_clusters - num_initial_centroids

                distances_from_initial_centroids = self._cdist_custom(X, centroids)
                furthest_point_indices = np.argsort(
                    distances_from_initial_centroids.min(axis=1)
                )[-remaining_centroids_needed:]

                additional_centroids = X[furthest_point_indices]
                return np.vstack([centroids, additional_centroids])

        elif isinstance(self.seeds, (list, tuple, np.ndarray)):
            # Any sequence of centroid points. numpy arrays belong here: scaling
            # seeds produces one (scaler.transform always returns an ndarray),
            # so this is the shape callers arrive with most often.
            seeds = np.asarray(self.seeds)
            if len(seeds) == self.n_clusters:
                return seeds
            elif len(seeds) > self.n_clusters:
                distances = self._cdist_custom(seeds, seeds)
                np.fill_diagonal(distances, np.inf)

                selected_seed_indices = []
                while len(selected_seed_indices) < self.n_clusters:
                    if not selected_seed_indices:
                        selected_seed_indices.append(0)
                    else:
                        candidates = [
                            i for i in range(len(seeds))
                            if i not in selected_seed_indices
                        ]
                        max_min_distance = -1
                        best_candidate = None

                        for candidate in candidates:
                            min_dist = min(
                                self._cdist_custom(
                                    [seeds[candidate]],
                                    [seeds[idx] for idx in selected_seed_indices]
                                ).min(),
                                0
                            )
                            if min_dist > max_min_distance:
                                max_min_distance = min_dist
                                best_candidate = candidate

                        selected_seed_indices.append(best_candidate)
                return seeds[selected_seed_indices]

            elif len(seeds) < self.n_clusters:
                initial_centroids = seeds.copy()
                remaining_centroids = self.n_clusters - len(seeds)

                distances_from_seeds = self._cdist_custom(X, initial_centroids)
                furthest_point_indices = np.argsort(
                    distances_from_seeds.min(axis=1)
                )[-remaining_centroids:]

                additional_centroids = X[furthest_point_indices]
                return np.vstack([initial_centroids, additional_centroids])
        else:
            # Previously this fell back to random initialisation, which meant an
            # unrecognised container silently discarded the seeds and left the
            # caller believing the clustering was anchored when it was not.
            raise TypeError(
                f"seeds must be a dict of {{centroid: [seed points]}}, a sequence "
                f"of centroid points (list, tuple or ndarray), or None for random "
                f"initialisation; got {type(self.seeds).__name__}."
            )


    def _cdist_custom(self, XA, XB):
        """
        Wrapper for cdist with custom distance metric handling.
        """

        XA = np.asarray(XA)
        XB = np.asarray(XB)

        if self.distance_metric == 'euclidean':
            return cdist(XA, XB, metric='euclidean')
        elif self.distance_metric == 'mahalanobis':
            if self.metric_params and 'VI' in self.metric_params:
                VI = self.metric_params['VI']
                return cdist(XA, XB, metric='mahalanobis', VI=VI)
            else:
                raise ValueError("For mahalanobis distance, metric_params must contain 'VI' (inverse covariance matrix).")
        
        elif self.distance_metric == 'custom':
            if self.metric_params and 'func' in self.metric_params:
                custom_dist_func = self.metric_params['func']
                distances = np.zeros((XA.shape[0], XB.shape[0]))
                for i in range(XA.shape[0]):
                    for j in range(XB.shape[0]):
                        distances[i, j] = custom_dist_func(XA[i], XB[j])
                return distances
            else:
                raise ValueError("For custom distance, metric_params must contain 'func' (custom distance function).")
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")


    def fit(self, X, known_labels=None, is_slight_movement=False):
        """
        Perform density-constrained clustering with radial threshold constraints and custom distance metrics and seed points.
        """
        point_densities = self._compute_local_density(X)
        centroids = self._initialize_centroids(X)
        known_centroids = centroids.copy()
        initial_centroids = centroids.copy()

        if known_labels is None:
            known_labels = np.arange(self.n_clusters)

        # Calculate global covariance matrix for Mahalanobis if needed (outside loop)
        if self.distance_metric == 'mahalanobis' and self.metric_params is None:
            covariance_matrix = np.cov(X.T)
            try:
                inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            except np.linalg.LinAlgError:
                inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
            self.metric_params = {'VI': inv_covariance_matrix}


        # Prepare seed point indices if seeds are provided as dictionary
        seed_points_list = []
        if isinstance(self.seeds, dict):
            centroid_keys = list(self.seeds.keys())
            for cluster_idx, centroid_point in enumerate(centroid_keys):
                seed_points = self.seeds[centroid_point]
                seed_indices_for_cluster = []
                for seed_point in seed_points:
                    seed_index = np.where((X == seed_point).all(axis=1))[0] # Find index of seed point in X
                    if len(seed_index) > 0:
                        seed_indices_for_cluster.append(seed_index[0])
                        seed_points_list.append(X[seed_index[0]]) # Collect seed points for visualization
                    else:
                        print(f"Warning: Seed point {seed_point} not found in dataset X.")
                self.seed_indices_[cluster_idx] = seed_indices_for_cluster # Store indices per cluster
            seed_points_array = np.array(seed_points_list) if seed_points_list else None
        else:
            seed_points_array = np.array(self.seeds) if self.seeds is not None else None


        iteration = -1
        converged = False
        decision_centroids = centroids.copy()

        for iteration in range(self.max_iter):
            prev_centroids = centroids.copy()

            # Centroids that actually decide this round's labels. `centroids` is
            # only updated at the very end of the loop body, which the
            # convergence `break` skips -- but a max_iter exit does not. Keeping
            # the deciding set separately is what lets decision_path() rebuild
            # labels_ exactly however the loop ended.
            decision_centroids = centroids.copy()

            # Compute distances to centroids using custom distance function
            distances = self._cdist_custom(X, centroids)

            preliminary_labels = np.argmin(distances, axis=1)

            filtered_labels = preliminary_labels.copy()
            unassigned_mask = np.zeros(len(X), dtype=bool)

            for i in range(len(X)):
                if point_densities[i] > 1 - self.density_threshold:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1
                elif distances[i, filtered_labels[i]] > self.distance_threshold and filtered_labels[i] != -1:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1


            new_centroids_raw_list = []
            for k in range(self.n_clusters):
                cluster_points = X[filtered_labels == k]
                if np.any(filtered_labels == k):
                    new_centroids_raw_list.append(cluster_points.mean(axis=0))
                else:
                    new_centroids_raw_list.append(centroids[k]) # Keep old centroid if no points in cluster
            new_centroids_raw = np.array(new_centroids_raw_list)
            new_centroids = new_centroids_raw.copy()

            for k in range(self.n_clusters):
                displacement = new_centroids_raw[k] - initial_centroids[k]
                distance_from_initial = np.linalg.norm(displacement)

                if is_slight_movement:
                    if distance_from_initial > self.radial_threshold:
                        new_centroids[k] = initial_centroids[k] + (displacement / distance_from_initial) * self.radial_threshold
                else:
                    if distance_from_initial > self.radial_threshold:
                        new_centroids[k] = prev_centroids[k]

            centroid_displacements = np.linalg.norm(new_centroids - centroids, axis=1)

            if np.all(centroid_displacements < self.convergence_tolerance):
                converged = True
                break

            centroids = new_centroids.copy()

        if known_labels is not None:
            cluster_mapping, mapped_labels = hungarian_match(known_centroids, centroids, known_labels, filtered_labels)
            self.mapped_labels_ = mapped_labels
            self.cluster_mapping_ = cluster_mapping
        else:
            self.mapped_labels_ = filtered_labels
            self.cluster_mapping_ = {i: i for i in range(self.n_clusters)}

        self.labels_ = filtered_labels
        self.original_centroids_ = known_centroids
        self.centroids_ = centroids
        self.decision_centroids_ = decision_centroids
        self.n_iter_ = iteration + 1
        self.converged_ = converged
        self.point_densities_ = point_densities
        self.unassigned_mask_ = unassigned_mask
        self.seed_points_array_ = seed_points_array if isinstance(self.seeds, dict) else seed_points_array # Store seed points for visualization

        self._post_process_seeds()

        return self


    def _post_process_seeds(self):
        """
        Post-process cluster labels to ensure seed points are assigned to their intended clusters.
        This function is called at the end of the fit method.
        """
        if isinstance(self.seeds, dict):
            for cluster_idx in self.seed_indices_:
                for seed_index in self.seed_indices_[cluster_idx]:
                    self.labels_[seed_index] = cluster_idx # Forcefully assign seed points to their cluster in final labels


    # ------------------------------------------------------------------
    # Interpretability: closed-form reconstruction of the Phase-1 decisions
    # ------------------------------------------------------------------

    @property
    def local_density_(self):
        """
        Normalised local density per point, the quantity the density gate
        compares against ``1 - density_threshold``.

        Read-only alias of ``point_densities_``, kept so explainers do not have
        to re-derive the density (and silently drift from it if the kernel
        here ever changes).
        """
        if not hasattr(self, "point_densities_"):
            raise AttributeError(
                "local_density_ is only available after fit(); call fit(X) first."
            )
        return self.point_densities_

    def _seed_labels(self, n_samples):
        """
        Per-point cluster index for pinned seeds, ``-1`` where the point is not
        a seed. Derived from ``seed_indices_``, so it stays in step with what
        ``_post_process_seeds`` actually forced.
        """
        seed_cluster = np.full(n_samples, -1, dtype=int)
        for cluster_idx, indices in self.seed_indices_.items():
            for row in indices:
                seed_cluster[row] = cluster_idx
        return seed_cluster

    def decision_path(self, X, check_fidelity=True):
        """
        Reconstruct, in closed form, why each point received its label.

        Phase 1 uses no latent representation: a point goes to the nearest
        seed-anchored centroid unless one of two explicit gates rejects it, and
        literature seeds are pinned regardless. This method replays those rules
        against the state stored by ``fit``, so it is an exact reconstruction of
        the model rather than a surrogate such as SHAP or LIME.

        The gates are evaluated in the order ``fit`` applies them:

        1. density  -- ``local_density > 1 - density_threshold`` -> noise
        2. distance -- ``d(point, nearest centroid) > distance_threshold`` -> noise
        3. otherwise, assigned to the nearest centroid
        4. seed forcing overrides 1-3 for dictionary seed points

        Because seed labels are copied rather than derived, they are excluded
        from the fidelity check, which would otherwise be vacuous on them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit``. Densities and seed indices are
            stored by row position, so another matrix yields a meaningless
            table; ``check_fidelity`` is what turns that into a loud failure.
        check_fidelity : bool, default=True
            Raise if the reconstruction does not reproduce ``labels_`` on every
            non-seed point. Leave it on: it is the guarantee this method exists
            to provide.

        Returns
        -------
        pandas.DataFrame
            One row per point, with the gate outcomes, the reconstructed label
            and the margins that separate each point from flipping.

        Raises
        ------
        RuntimeError
            If the model is not fitted, or the reconstruction disagrees with
            ``labels_`` while ``check_fidelity`` is on.
        ValueError
            If ``X`` has a different number of rows than the fitted data.
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError(
                "decision_path() needs a fitted model; call fit(X) first."
            )

        X = np.asarray(X)
        n = len(self.labels_)
        if len(X) != n:
            raise ValueError(
                f"decision_path() expects the matrix passed to fit(): "
                f"got {len(X)} rows, fitted on {n}."
            )

        arange = np.arange(n)

        # Reconstruct against the centroids that decided, and through the same
        # metric wrapper fit() used -- so a change of distance_metric cannot
        # quietly desynchronise the explanation from the model.
        distances = self._cdist_custom(X, self.decision_centroids_)
        nearest = np.argmin(distances, axis=1)
        d_nearest = distances[arange, nearest]

        density = self.point_densities_
        density_limit = 1 - self.density_threshold
        density_passed = density <= density_limit
        distance_passed = d_nearest <= self.distance_threshold

        seed_cluster = self._seed_labels(n)
        is_seed = seed_cluster != -1

        gate = np.empty(n, dtype=object)
        gate[:] = "assigned"
        gate[~distance_passed] = "rejected_distance"
        gate[~density_passed] = "rejected_density"   # density is tested first

        reconstructed = np.where(density_passed & distance_passed, nearest, -1)

        gate[is_seed] = "seed_forced"
        reconstructed[is_seed] = seed_cluster[is_seed]

        # Runner-up centroid: how much of a lead the winner had.
        if self.decision_centroids_.shape[0] > 1:
            masked = distances.copy()
            masked[arange, nearest] = np.inf
            competitor = np.argmin(masked, axis=1)
            d_competitor = distances[arange, competitor]
        else:
            competitor = np.full(n, -1, dtype=int)
            d_competitor = np.full(n, np.nan)

        non_seed = ~is_seed
        if non_seed.any():
            agree = reconstructed[non_seed] == self.labels_[non_seed]
            self.fidelity_ = float(np.mean(agree))
            n_mismatch = int(np.size(agree) - np.count_nonzero(agree))
        else:
            self.fidelity_ = 1.0
            n_mismatch = 0

        if check_fidelity and n_mismatch:
            raise RuntimeError(
                f"decision_path() reproduced only {self.fidelity_:.4%} of labels_ "
                f"({n_mismatch} of {int(non_seed.sum())} non-seed points disagree). "
                "The reconstruction is meant to be exact, so this means the model "
                "state and the gates have drifted apart -- check that X is the "
                "matrix fit() was given, and that fit() stored decision_centroids_."
            )

        return pd.DataFrame({
            "point_index": arange,
            "label": self.labels_,
            "reconstructed_label": reconstructed,
            "gate": gate,
            "is_seed": is_seed,
            "seed_cluster": seed_cluster,
            "local_density": density,
            "density_passed": density_passed,
            # >0 means the point sits below the density ceiling, so it survives.
            "density_margin": density_limit - density,
            "nearest_cluster": nearest,
            "distance_to_nearest": d_nearest,
            "distance_passed": distance_passed,
            # >0 means the point sits inside the distance ball.
            "distance_margin": self.distance_threshold - d_nearest,
            "competitor_cluster": competitor,
            "distance_to_competitor": d_competitor,
            # Lead of the winning centroid over the runner-up.
            "margin": d_competitor - d_nearest,
        })

    def _metric_matrix(self):
        """
        The matrix M for which the metric is the quadratic form
        ``d(x, c)^2 = (x - c)^T M (x - c)``, or None when the configured metric
        has no such form.
        """
        if self.distance_metric == "euclidean":
            return np.eye(len(self.centroids_[0]))
        if self.distance_metric == "mahalanobis":
            if self.metric_params and "VI" in self.metric_params:
                return np.asarray(self.metric_params["VI"])
        return None

    def feature_attribution(self, X, feature_names=None):
        """
        Split each point's distance into an exact per-feature contribution.

        Euclidean and Mahalanobis distances are quadratic forms, so the squared
        distance decomposes additively over the features::

            d(x, c)^2 = (x - c)^T M (x - c) = sum_a (x - c)_a * (M (x - c))_a

        Each term is that feature's share of the distance, and the shares sum to
        the distance exactly. There is no attribution model here and nothing is
        approximated -- this is algebra on the metric the model actually used,
        which is why it belongs beside ``decision_path`` rather than with the
        descriptive statistics.

        Two views are returned. The **absolute** contribution says which feature
        placed the point where it is relative to its own cluster centroid. The
        **discriminative** contribution says which feature favours that cluster
        over the runner-up: it is the difference of the two decompositions, so
        its terms sum to ``d_competitor^2 - d_assigned^2``. The discriminative
        view is usually the one worth reading, since a large absolute term may
        just mean the feature has a wide spread.

        Points are explained against the cluster they were actually placed in,
        which for a forced seed need not be the nearest one. Noise points are
        explained against the nearest cluster they failed to join.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same matrix passed to ``fit``.
        feature_names : sequence of str, optional
            Column names, defaulting to ``feature_0 ... feature_{d-1}``.

        Returns
        -------
        pandas.DataFrame
            One row per point: the reference and competitor clusters, the two
            squared distances, and per feature ``contrib_<name>``,
            ``contrib_<name>_frac``, ``disc_<name>``, ``disc_<name>_frac``, plus
            the dominant feature of each view.

        Raises
        ------
        RuntimeError
            If the model is not fitted, or the configured metric is not a
            quadratic form. A custom metric has no additive decomposition, so
            there is no honest per-feature split to return.
        ValueError
            If ``X`` has a different number of rows than the fitted data, or
            ``feature_names`` has the wrong length.
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError(
                "feature_attribution() needs a fitted model; call fit(X) first."
            )

        X = np.asarray(X)
        n = len(self.labels_)
        if len(X) != n:
            raise ValueError(
                f"feature_attribution() expects the matrix passed to fit(): "
                f"got {len(X)} rows, fitted on {n}."
            )

        matrix = self._metric_matrix()
        if matrix is None:
            raise RuntimeError(
                f"distance_metric={self.distance_metric!r} is not a quadratic "
                "form, so the squared distance does not decompose additively "
                "over the features and there is no exact attribution to give. "
                "Use decision_path() for the gate-level explanation."
            )

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{j}" for j in range(n_features)]
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries for "
                f"{n_features} columns of X."
            )

        arange = np.arange(n)
        distances = self._cdist_custom(X, self.decision_centroids_)
        nearest = np.argmin(distances, axis=1)

        # Explain each point against the cluster it ended up in; a forced seed's
        # cluster may not be its nearest. Noise falls back to the nearest one.
        reference = np.where(self.labels_ != -1, self.labels_, nearest)

        masked = distances.copy()
        masked[arange, reference] = np.inf
        competitor = np.argmin(masked, axis=1)

        delta_reference = X - self.decision_centroids_[reference]
        contribution = delta_reference * (delta_reference @ matrix)
        delta_competitor = X - self.decision_centroids_[competitor]
        contribution_competitor = delta_competitor * (delta_competitor @ matrix)
        discriminative = contribution_competitor - contribution

        d2_reference = contribution.sum(axis=1)
        d2_competitor = contribution_competitor.sum(axis=1)
        margin2 = d2_competitor - d2_reference

        table = pd.DataFrame({
            "point_index": arange,
            "reference_cluster": reference,
            "competitor_cluster": competitor,
            "d2_reference": d2_reference,
            "d2_competitor": d2_competitor,
            "margin2": margin2,
        })

        # Guard the shares against a zero denominator: a point sitting on its
        # centroid, or tied between two clusters, has no share to report.
        safe_d2 = np.where(d2_reference > 0, d2_reference, np.nan)
        safe_margin2 = np.where(margin2 != 0, margin2, np.nan)
        for j, name in enumerate(feature_names):
            table[f"contrib_{name}"] = contribution[:, j]
            table[f"contrib_{name}_frac"] = contribution[:, j] / safe_d2
            table[f"disc_{name}"] = discriminative[:, j]
            table[f"disc_{name}_frac"] = discriminative[:, j] / safe_margin2

        table["dominant_feature"] = [
            feature_names[a] for a in np.argmax(contribution, axis=1)
        ]
        table["dominant_discriminative_feature"] = [
            feature_names[a] for a in np.argmax(discriminative, axis=1)
        ]
        return table

    def visualize_clustering(self, X):
        """
        Create comprehensive 3D visualization of clustering results
        """
        fig = plt.figure(figsize=(20, 6), dpi=100)

        # Clustering Results Subplot
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=self.labels_,
            cmap='viridis',
            alpha=0.7
        )
        ax1.set_title('3D Density-Constrained Clustering')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        fig.colorbar(scatter1, ax=ax1, shrink=0.6)

        # Plot seed points with distinct marker style
        if self.seed_points_array_ is not None: # Use stored seed points array
            ax1.scatter(
                self.seed_points_array_[:, 0],
                self.seed_points_array_[:, 1],
                self.seed_points_array_[:, 2],
                c='black',
                marker='x',
                s=100,
                linewidth=3,
                label='Seed Points'
            )

        # Plot cluster centroids
        ax1.scatter(
            self.centroids_[:, 0],
            self.centroids_[:, 1],
            self.centroids_[:, 2],
            c='black',
            marker='^',
            s=100,
            label='Centroids'
        )
        ax1.legend()


        # Point Density Subplot
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=self.point_densities_,
            cmap='plasma',
            alpha=0.7
        )

        # Plot seed points with distinct marker style
        if self.seed_points_array_ is not None: # Use stored seed points array
            ax2.scatter(
                self.seed_points_array_[:, 0],
                self.seed_points_array_[:, 1],
                self.seed_points_array_[:, 2],
                c='black',
                marker='x',
                s=100,
                linewidth=3,
                label='Seed Points'
            )

        # Plot cluster centroids
        ax2.scatter(
            self.centroids_[:, 0],
            self.centroids_[:, 1],
            self.centroids_[:, 2],
            c='black',
            marker='^',
            s=100,
            label='Centroids'
        )

        ax2.set_title('Point Density Distribution')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        fig.colorbar(scatter2, ax=ax2, shrink=0.6)
        ax2.legend()

        # Unassigned Points Subplot
        ax3 = fig.add_subplot(133, projection='3d')

        # Separate clusters and unassigned points
        assigned_points = X[~self.unassigned_mask_]
        unassigned_points = X[self.unassigned_mask_]

        # Plot assigned points
        scatter3_1 = ax3.scatter(
            assigned_points[:, 0],
            assigned_points[:, 1],
            assigned_points[:, 2],
            c='blue',
            alpha=0.5,
            label='Assigned Points'
        )

        # Plot unassigned points
        scatter3_2 = ax3.scatter(
            unassigned_points[:, 0],
            unassigned_points[:, 1],
            unassigned_points[:, 2],
            c='red',
            alpha=0.7,
            label='Unassigned Points'
        )

        # Plot seed points with distinct marker style
        if self.seed_points_array_ is not None: # Use stored seed points array
            ax3.scatter(
                self.seed_points_array_[:, 0],
                self.seed_points_array_[:, 1],
                self.seed_points_array_[:, 2],
                c='black',
                marker='x',
                s=100,
                linewidth=3,
                label='Seed Points'
            )

        # Plot cluster centroids
        ax3.scatter(
            self.centroids_[:, 0],
            self.centroids_[:, 1],
            self.centroids_[:, 2],
            c='black',
            marker='^',
            s=100,
            label='Centroids'
        )

        ax3.set_title('Assigned vs Unassigned Points')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()

        plt.tight_layout()
        plt.show()

        return fig