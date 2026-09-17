import unittest
import numpy as np
from sklearn.datasets import make_blobs
from CLiMB.core.KBound import KBound
from scipy.spatial.distance import euclidean  # For custom distance test

class TestKBound(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        self.X, _ = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
        self.n_clusters = 3

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        kbound = KBound(n_clusters=self.n_clusters)
        self.assertEqual(kbound.n_clusters, self.n_clusters)
        self.assertIsNone(kbound.seeds)
        self.assertEqual(kbound.distance_metric, 'euclidean')
        self.assertIsNone(kbound.metric_params)

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        seeds_list = [self.X[0], self.X[20], self.X[40]]
        metric_params = {'VI': np.eye(3)} # Example VI for Mahalanobis
        kbound = KBound(
            n_clusters=self.n_clusters,
            seeds=seeds_list,
            max_iter=100,
            density_threshold=0.2,
            distance_threshold=2.5,
            radial_threshold=1.5,
            convergence_tolerance=0.01,
            distance_metric='mahalanobis',
            metric_params=metric_params
        )
        self.assertEqual(kbound.n_clusters, self.n_clusters)
        self.assertEqual(len(kbound.seeds), len(seeds_list))
        self.assertEqual(kbound.max_iter, 100)
        self.assertEqual(kbound.density_threshold, 0.2)
        self.assertEqual(kbound.distance_threshold, 2.5)
        self.assertEqual(kbound.radial_threshold, 1.5)
        self.assertEqual(kbound.convergence_tolerance, 0.01)
        self.assertEqual(kbound.distance_metric, 'mahalanobis')
        self.assertEqual(kbound.metric_params, metric_params)

    def test_fit_no_seeds(self):
        """Test fit method with no seeds."""
        kbound = KBound(n_clusters=self.n_clusters)
        kbound.fit(self.X)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))
        self.assertIsNotNone(kbound.centroids_)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))

    def test_fit_dict_seeds(self):
        """Test fit method with dictionary of seed points."""
        seed_centroids = [self.X[10], self.X[30], self.X[60]]
        seed_dict = {tuple(seed_centroids[0]): [tuple(self.X[1]), tuple(self.X[2])],
                     tuple(seed_centroids[1]): [tuple(self.X[3])],
                     tuple(seed_centroids[2]): []}
        kbound = KBound(n_clusters=self.n_clusters, seeds=seed_dict)
        kbound.fit(self.X)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))
        self.assertEqual(len(np.unique(kbound.labels_)), self.n_clusters)
        self.assertIsNotNone(kbound.centroids_)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))

    def test_fit_mahalanobis_metric_precomputed_vi(self):
        """Test fit method with Mahalanobis distance metric and pre-computed VI."""
        VI = np.linalg.inv(np.cov(self.X.T))
        kbound = KBound(
            n_clusters=self.n_clusters,
            distance_metric='mahalanobis',
            metric_params={'VI': VI}
        )
        kbound.fit(self.X)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))
        self.assertIsNotNone(kbound.centroids_)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))

    def test_fit_mahalanobis_metric_auto_vi(self):
        """Test fit method with Mahalanobis distance metric and automatic VI calculation."""
        kbound = KBound(
            n_clusters=self.n_clusters,
            distance_metric='mahalanobis'
        )
        kbound.fit(self.X)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))
        self.assertIsNotNone(kbound.centroids_)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))

    def test_fit_custom_metric(self):
        """Test fit method with custom distance metric."""
        def custom_dist(u, v):
            return euclidean(u * np.array([2, 1, 1]), v * np.array([2, 1, 1])) # Weighted Euclidean
        kbound = KBound(
            n_clusters=self.n_clusters,
            distance_metric='custom',
            metric_params={'func': custom_dist}
        )
        kbound.fit(self.X)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))
        self.assertIsNotNone(kbound.centroids_)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))

    def test_fit_with_list_seeds_matching_n_clusters(self):
        """A list of exactly n_clusters seeds is the plainest way to call this,
        and it used to crash: the branch assigned the centroids to a local and
        returned None."""
        seeds_list = [self.X[10], self.X[30], self.X[60]]
        kbound = KBound(n_clusters=self.n_clusters, seeds=seeds_list)
        kbound.fit(self.X)
        self.assertEqual(kbound.centroids_.shape, (self.n_clusters, self.X.shape[1]))
        np.testing.assert_array_equal(kbound.original_centroids_, np.array(seeds_list))

    def test_seed_container_does_not_change_the_result(self):
        """
        Seeds used to be dispatched on the container type, so a numpy array --
        what scaler.transform() hands back, and what every example in the repo
        passed -- matched neither the dict nor the list branch and fell through
        to random initialisation. The seeds were discarded in silence and the
        radial constraint anchored to a random point instead of the seed.
        """
        seeds = [self.X[10], self.X[30], self.X[60]]
        kw = dict(n_clusters=self.n_clusters, density_threshold=0.05,
                  distance_threshold=3.0, radial_threshold=0.5,
                  convergence_tolerance=1e-6)

        reference = KBound(seeds=list(seeds), **kw).fit(self.X)
        for name, container in (("list", list(seeds)),
                                ("tuple", tuple(seeds)),
                                ("ndarray", np.array(seeds))):
            with self.subTest(container=name):
                kbound = KBound(seeds=container, **kw).fit(self.X)
                np.testing.assert_array_equal(kbound.original_centroids_, np.array(seeds))
                np.testing.assert_array_equal(kbound.labels_, reference.labels_)

    def test_unsupported_seed_type_raises_instead_of_going_random(self):
        """An unrecognised container must fail loudly, not look like seeding."""
        for bad in ({1, 2, 3}, "seeds", 42):
            with self.subTest(seeds=type(bad).__name__):
                with self.assertRaises(TypeError):
                    KBound(n_clusters=self.n_clusters, seeds=bad).fit(self.X)

    def test_fit_with_known_labels(self):
        """Test fit method with known labels."""
        known_labels = np.array([0] * 30 + [1] * 30 + [2] * 40) # Example known labels
        kbound = KBound(n_clusters=self.n_clusters)
        kbound.fit(self.X, known_labels=known_labels)
        self.assertIsNotNone(kbound.labels_)
        self.assertEqual(len(kbound.labels_), len(self.X))

    def test_post_process_seeds_dict(self):
        """Test _post_process_seeds with dictionary seeds."""
        seed_centroids = [self.X[10], self.X[30], self.X[60]]
        seed_dict = {tuple(seed_centroids[0]): [tuple(self.X[1]), tuple(self.X[2])],
                     tuple(seed_centroids[1]): [tuple(self.X[3])],
                     tuple(seed_centroids[2]): []}
        kbound = KBound(n_clusters=self.n_clusters, seeds=seed_dict)
        kbound.fit(self.X)
        seed_indices_cluster0 = kbound.seed_indices_[0]
        for seed_index in seed_indices_cluster0:
            self.assertEqual(kbound.labels_[seed_index], 0) # Check if seed points are forced to cluster 0
        seed_indices_cluster1 = kbound.seed_indices_[1]
        for seed_index in seed_indices_cluster1:
            self.assertEqual(kbound.labels_[seed_index], 1) # Check if seed points are forced to cluster 1

    def test_invalid_mahalanobis_params(self):
        """Test ValueError when metric_params is missing for Mahalanobis."""
        kbound = KBound(n_clusters=self.n_clusters, distance_metric='mahalanobis', metric_params={"IV": None})
        with self.assertRaisesRegex(ValueError, "For mahalanobis distance, metric_params must contain 'VI'"):
            kbound.fit(self.X) # Fit should raise ValueError if VI is not provided and cannot be auto-calculated (e.g., singular covariance)

    def test_invalid_custom_metric_params(self):
        """Test ValueError when metric_params is missing for custom metric."""
        kbound = KBound(n_clusters=self.n_clusters, distance_metric='custom')
        with self.assertRaisesRegex(ValueError, "For custom distance, metric_params must contain 'func'"):
            kbound.fit(self.X) # Fit should raise ValueError if func is not provided

    def test_unsupported_metric(self):
        """Test ValueError for unsupported distance metric."""
        kbound = KBound(n_clusters=self.n_clusters, distance_metric='unsupported_metric')
        with self.assertRaisesRegex(ValueError, "Unsupported distance metric: unsupported_metric"):
            kbound.fit(self.X) # Fit should raise ValueError for unsupported metric

    def test_fit_with_seed_dict_does_not_raise_attribute_error(self):
        """
        Tests that KBound.fit() does not raise an AttributeError when initialized
        with a dictionary of seed points. This replicates a bug where a list
        was not converted to a NumPy array internally.
        """
        # 1. ARRANGE: Create synthetic data and a seed dictionary
        # (Questa parte è identica a prima)
        X = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 1
            [8, 8], [8, 9], [9, 8], [9, 9]   # Cluster 2
        ])

        initial_centroids = [
            np.array([1.5, 1.5]),
            np.array([8.5, 8.5])
        ]
        seed_points_cluster1 = [np.array([1, 1]), np.array([2, 1])]
        seed_points_cluster2 = [np.array([8, 8])]

        seed_dict = {
            tuple(initial_centroids[0]): [tuple(p) for p in seed_points_cluster1],
            tuple(initial_centroids[1]): [tuple(p) for p in seed_points_cluster2]
        }

        # 2. ACT: Instantiate KBound
        kbound = KBound(
            n_clusters=2,
            seeds=seed_dict
        )

        # 3. ACT & ASSERT: Call fit() and assert that no AttributeError is raised.
        # Il blocco try...except è un modo robusto per catturare l'errore specifico.
        try:
            kbound.fit(X)
        except AttributeError as e:
            # Se si verifica questo errore, il test fallisce con un messaggio chiaro.
            # self.fail() è l'equivalente di pytest.fail()
            self.fail(
                f"KBound.fit() raised an unexpected AttributeError. "
                f"The likely cause is an input to _cdist_custom not being converted to a NumPy array. "
                f"Error: {e}"
            )
        except Exception as e:
            self.fail(f"KBound.fit() raised an unexpected exception: {e}")

        # 4. ASSERT: Use self.assert... methods for final checks.
        self.assertTrue(hasattr(kbound, 'centroids_'), "KBound should have a 'centroids_' attribute after fitting.")
        self.assertIsNotNone(kbound.centroids_, "Centroids should not be None after fitting.")
        # Per confrontare l'uguaglianza, unittest usa self.assertEqual()
        # Nota: per gli array NumPy, è meglio confrontare la tupla delle shape
        self.assertEqual(kbound.centroids_.shape, (2, 2), "The shape of the final centroids should be (n_clusters, n_features).")


if __name__ == '__main__':
    unittest.main()

class TestKBoundDecisionPath(unittest.TestCase):
    """
    Phase-1 decisions are two explicit gates plus seed forcing, so explaining
    them is a closed-form replay of the model, not a surrogate fit. These tests
    promote that to an invariant: the reconstruction must reproduce ``labels_``
    on every non-seed point, exactly, however the fit loop ended.
    """

    def setUp(self):
        from sklearn.preprocessing import StandardScaler
        X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=0)
        self.X = StandardScaler().fit_transform(X)
        self.y = y
        # Seeds deliberately offset from the true blob centres so the centroids
        # have to travel: without movement there is no drift to regress against.
        self.seeds = {
            tuple(self.X[y == c].mean(axis=0) + 0.8):
                [self.X[np.where(y == c)[0][i]].tolist() for i in range(3)]
            for c in range(3)
        }
        # radial_threshold is loose and the tolerance tight, so convergence is
        # reached by actually settling rather than by being clamped in place.
        self.params = dict(
            n_clusters=3, seeds=self.seeds, density_threshold=0.02,
            distance_threshold=1.5, radial_threshold=50.0,
            convergence_tolerance=1e-9,
        )

    def _assert_scenario_is_not_degenerate(self, kbound, df):
        """
        Guard against a test that passes for the wrong reason. If every point
        were rejected, or every point assigned to one cluster, fidelity would be
        trivially 1.0 and the assertions below would prove nothing.
        """
        gates = set(df["gate"])
        self.assertIn("assigned", gates, "no point was assigned: gates are vacuous")
        self.assertGreater((df["gate"] == "assigned").sum(), 20,
                           "too few assignments for the check to mean anything")
        self.assertGreater((df["label"] == -1).sum(), 0,
                           "nothing was rejected: the gates never fired")
        self.assertGreater(len(set(kbound.labels_)) - 1, 1,
                           "fewer than two clusters populated")

    def test_fidelity_is_exact_when_converged(self):
        """Reconstruction reproduces labels_ on every non-seed point."""
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        df = kbound.decision_path(self.X)
        self._assert_scenario_is_not_degenerate(kbound, df)

        self.assertTrue(kbound.converged_)
        self.assertEqual(kbound.fidelity_, 1.0)
        non_seed = ~df["is_seed"].to_numpy()
        np.testing.assert_array_equal(
            df["reconstructed_label"].to_numpy()[non_seed],
            df["label"].to_numpy()[non_seed],
        )

    def test_fidelity_is_exact_when_max_iter_is_exhausted(self):
        """
        Regression test for the centroid off-by-one-iteration.

        The fit loop refreshes the centroids at the end of the body, which the
        convergence ``break`` skips but a ``max_iter`` exit does not. Rebuilding
        against ``centroids_`` then replays a decision the algorithm never took
        (fidelity fell to 0.55 at max_iter=1). ``decision_centroids_`` is what
        keeps it exact; the drift assertion below is what keeps this test honest.
        """
        for max_iter in (1, 2, 3):
            with self.subTest(max_iter=max_iter):
                kbound = KBound(max_iter=max_iter, **self.params).fit(self.X)
                self.assertFalse(kbound.converged_)
                self.assertEqual(kbound.n_iter_, max_iter)

                drift = np.linalg.norm(kbound.decision_centroids_ - kbound.centroids_)
                self.assertGreater(
                    drift, 1e-8,
                    "centroids_ and decision_centroids_ coincide, so this run "
                    "cannot detect the off-by-one it is meant to guard",
                )

                df = kbound.decision_path(self.X)
                self._assert_scenario_is_not_degenerate(kbound, df)
                self.assertEqual(kbound.fidelity_, 1.0)

    def test_decision_centroids_match_centroids_on_convergence(self):
        """The two centroid sets must not diverge when the fit settles."""
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        self.assertTrue(kbound.converged_)
        np.testing.assert_allclose(kbound.decision_centroids_, kbound.centroids_)

    def test_fidelity_is_exact_across_metrics(self):
        """The reconstruction follows distance_metric instead of assuming one."""
        cases = {
            "euclidean": (dict(distance_metric="euclidean"), None),
            "mahalanobis": (dict(distance_metric="mahalanobis",
                                 metric_params={"VI": np.linalg.inv(np.cov(self.X.T))}), None),
            "custom": (dict(distance_metric="custom",
                            metric_params={"func": euclidean}), None),
        }
        for name, (extra, _) in cases.items():
            with self.subTest(metric=name):
                kbound = KBound(max_iter=300, **self.params, **extra).fit(self.X)
                df = kbound.decision_path(self.X)
                self._assert_scenario_is_not_degenerate(kbound, df)
                self.assertEqual(kbound.fidelity_, 1.0)

    def test_fidelity_is_exact_without_dict_seeds(self):
        """No pinned seeds means every point is derived, none copied."""
        for seeds in (None, [self.X[10], self.X[80], self.X[150]]):
            with self.subTest(seeds=type(seeds).__name__):
                params = dict(self.params, seeds=seeds)
                kbound = KBound(max_iter=300, **params).fit(self.X)
                df = kbound.decision_path(self.X)
                self.assertFalse(df["is_seed"].any())
                self.assertEqual(kbound.fidelity_, 1.0)

    def test_gates_agree_with_their_boolean_columns(self):
        """The reported gate is the one the thresholds actually selected."""
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        df = kbound.decision_path(self.X)

        self.assertEqual(
            set(df["gate"]) - {"assigned", "rejected_density",
                               "rejected_distance", "seed_forced"},
            set(),
        )
        assigned = df[df["gate"] == "assigned"]
        self.assertTrue(assigned["density_passed"].all())
        self.assertTrue(assigned["distance_passed"].all())
        np.testing.assert_array_equal(assigned["reconstructed_label"],
                                      assigned["nearest_cluster"])

        # Density is tested before distance, so a density rejection is reported
        # even when the point also sits outside the distance ball.
        self.assertFalse(df.loc[df["gate"] == "rejected_density", "density_passed"].any())
        rejected_dist = df[df["gate"] == "rejected_distance"]
        self.assertTrue(rejected_dist["density_passed"].all())
        self.assertFalse(rejected_dist["distance_passed"].any())
        for gate in ("rejected_density", "rejected_distance"):
            self.assertTrue((df.loc[df["gate"] == gate, "reconstructed_label"] == -1).all())

    def test_margins_have_the_sign_their_gate_implies(self):
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        df = kbound.decision_path(self.X)
        self.assertTrue((df.loc[df["density_passed"], "density_margin"] >= 0).all())
        self.assertTrue((df.loc[~df["density_passed"], "density_margin"] < 0).all())
        self.assertTrue((df.loc[df["distance_passed"], "distance_margin"] >= 0).all())
        self.assertTrue((df.loc[~df["distance_passed"], "distance_margin"] < 0).all())
        self.assertTrue((df["margin"] >= 0).all(), "runner-up closer than the winner")

    def test_seed_points_are_reported_as_forced(self):
        """Seed labels are copied, not derived, and the table says so."""
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        df = kbound.decision_path(self.X)

        pinned = {row for rows in kbound.seed_indices_.values() for row in rows}
        self.assertTrue(pinned, "no seed was pinned, so nothing is being tested")
        seeds = df[df["is_seed"]]
        self.assertEqual(set(seeds["point_index"]), pinned)
        self.assertTrue((seeds["gate"] == "seed_forced").all())
        np.testing.assert_array_equal(seeds["reconstructed_label"], seeds["seed_cluster"])
        np.testing.assert_array_equal(seeds["label"], seeds["seed_cluster"])

    def test_local_density_aliases_point_densities(self):
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        np.testing.assert_array_equal(kbound.local_density_, kbound.point_densities_)
        self.assertEqual(len(kbound.local_density_), len(self.X))
        self.assertAlmostEqual(float(np.max(kbound.local_density_)), 1.0)

    def test_local_density_requires_a_fit(self):
        with self.assertRaises(AttributeError):
            KBound(**self.params).local_density_

    def test_decision_path_requires_a_fit(self):
        with self.assertRaises(RuntimeError):
            KBound(**self.params).decision_path(self.X)

    def test_decision_path_rejects_a_different_matrix(self):
        kbound = KBound(max_iter=300, **self.params).fit(self.X)
        with self.assertRaises(ValueError):
            kbound.decision_path(self.X[:50])
        with self.assertRaises(RuntimeError):
            kbound.decision_path(np.random.RandomState(1).randn(*self.X.shape))

    def test_fidelity_check_fires_on_a_drifted_reconstruction(self):
        """
        The guard must be load-bearing, not decoration: reinstating the old
        behaviour (rebuild against centroids_) has to be caught.
        """
        kbound = KBound(max_iter=1, **self.params).fit(self.X)
        kbound.decision_centroids_ = kbound.centroids_
        with self.assertRaises(RuntimeError):
            kbound.decision_path(self.X)

        df = kbound.decision_path(self.X, check_fidelity=False)
        self.assertEqual(len(df), len(self.X))
        self.assertLess(kbound.fidelity_, 1.0)
