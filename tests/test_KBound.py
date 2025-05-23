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


if __name__ == '__main__':
    unittest.main()