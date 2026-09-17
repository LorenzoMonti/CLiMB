import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles

# Import your modules (adjust paths as needed)
from CLiMB.core.CLiMB import CLiMB

class TestCLiMB(unittest.TestCase):
    
    def setUp(self):
        """Create synthetic datasets for testing"""
        # KBound still initialises from the global RNG when no seeds are given,
        # so pin it to keep the suite order-independent.
        np.random.seed(42)

        # Create dataset with known clusters
        self.X_blobs, self.y_blobs = make_blobs(
            n_samples=300, 
            centers=3, 
            n_features=3, 
            random_state=42
        )
        
        # Create complex dataset with non-convex shapes
        self.X_moons, self.y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
        # Add a third dimension to moons
        self.X_moons = np.column_stack((
            self.X_moons, 
            np.random.normal(0, 0.1, size=self.X_moons.shape[0])
        ))
        
        # Create mixed dataset with some known and some exploratory clusters
        self.X_mixed = np.vstack([
            self.X_blobs[:150],  # First half of blobs (known)
            self.X_moons  # Moons (to be discovered)
        ])
        
        # Standardize datasets
        self.scaler = StandardScaler()
        self.X_blobs_scaled = self.scaler.fit_transform(self.X_blobs)
        self.X_moons_scaled = self.scaler.fit_transform(self.X_moons)
        self.X_mixed_scaled = self.scaler.fit_transform(self.X_mixed)
        
        # Create seed points from known clusters
        self.seed_points = np.array([
            self.X_blobs[self.y_blobs == 0].mean(axis=0),
            self.X_blobs[self.y_blobs == 1].mean(axis=0),
            self.X_blobs[self.y_blobs == 2].mean(axis=0)
        ])
        
        # Create scaled seed points
        self.seed_points_scaled = self.scaler.transform(self.seed_points)
    
    def test_initialization(self):
        """Test proper initialization of CLiMB"""
        # Test default initialization
        climb = CLiMB()
        self.assertEqual(climb.constrained_clusters, 3)
        self.assertIsNone(climb.seed_points)
        self.assertEqual(climb.density_threshold, 0.2)
        
        # Test custom initialization
        custom_climb = CLiMB(
            constrained_clusters=5,
            seed_points=self.seed_points,
            density_threshold=0.3,
            distance_threshold=10,
            radial_threshold=2,
            convergence_tolerance=0.5
        )
        self.assertEqual(custom_climb.constrained_clusters, 5)
        self.assertIs(custom_climb.seed_points, self.seed_points)
        self.assertEqual(custom_climb.density_threshold, 0.3)
        self.assertEqual(custom_climb.distance_threshold, 10)
        self.assertEqual(custom_climb.radial_threshold, 2)
        self.assertEqual(custom_climb.convergence_tolerance, 0.5)
    
    def test_builder_setters(self):
        """Test builder pattern setters"""
        climb = CLiMB()
        climb.set_density(0.4).set_distance(20).set_radial(1.5).set_convergence(0.3)
        
        self.assertEqual(climb.density_threshold, 0.4)
        self.assertEqual(climb.distance_threshold, 20)
        self.assertEqual(climb.radial_threshold, 1.5)
        self.assertEqual(climb.convergence_tolerance, 0.3)
    
    def test_fit_blobs(self):
        """Test fitting on blob dataset"""
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=self.seed_points_scaled
        )
        
        climb.fit(self.X_blobs_scaled)
        
        # Check that attributes are properly set
        self.assertIsNotNone(climb.constrained_labels)
        self.assertIsNotNone(climb.constrained_centroids)
        self.assertIsNotNone(climb.signed_points)
        self.assertIsNotNone(climb.unassigned_points)
        
        # Get final labels
        labels = climb.get_labels()
        self.assertEqual(len(labels), len(self.X_blobs_scaled))
        
        # Check that we have both constrained and exploratory clusters
        unique_labels = np.unique(labels)
        self.assertTrue(len(unique_labels) > 0)
    
    def test_fit_mixed(self):
        """Test fitting on mixed dataset with known and unknown clusters"""
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=self.seed_points_scaled,
            density_threshold=0.3,
            distance_threshold=10
        )
        
        climb.fit(self.X_mixed_scaled)
        
        # Get final labels
        labels = climb.get_labels()
        self.assertEqual(len(labels), len(self.X_mixed_scaled))
        
        # Check that we have both constrained and exploratory clusters
        unique_labels = np.unique(labels)
        self.assertTrue(len(unique_labels) > 0)
        
        # There should be unassigned points for exploratory phase
        self.assertTrue(len(climb.unassigned_points) > 0)
        self.assertTrue(len(climb.exploratory_labels) > 0)
    
    def test_inverse_transform(self):
        """Test inverse transform of scaled data"""
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=self.seed_points_scaled
        )
        
        climb.fit(self.X_blobs_scaled)
        climb.inverse_transform(self.scaler)
        
        # Check that centroids are in original scale
        centroid_range = np.ptp(climb.constrained_centroids, axis=0)
        original_range = np.ptp(self.X_blobs, axis=0)
        
        # Ranges should be similar in original scale
        self.assertTrue(np.allclose(centroid_range, original_range, rtol=0.9))
    
    def test_visualization(self):
        """Test visualization methods"""
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=self.seed_points_scaled
        )
        
        climb.fit(self.X_blobs_scaled)
        
        # Test 3D visualization
        fig_3d = climb.plot_comprehensive_3d(axis_labels=['X', 'Y', 'Z'])
        self.assertIsNotNone(fig_3d)
        plt.close(fig_3d)
        
        # Test 2D visualization
        fig_2d = climb.plot_comprehensive_2d(dimensions=(0, 1), axis_labels=['X', 'Y'])
        self.assertIsNotNone(fig_2d)
        plt.close(fig_2d)
    
    def test_with_known_labels(self):
        """Test with known labels for matching"""
        # Create a simple CLiMB instance
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=self.seed_points_scaled
        )
        
        # Fit with known labels
        climb.fit(self.X_blobs_scaled, known_labels=self.y_blobs)
        
        # Check that the constrained labels match the known labels
        labels = climb.get_labels()
        mapped_labels = climb.mapped_labels
        
        # Verify mapping worked
        self.assertIsNotNone(mapped_labels)
        self.assertTrue(len(mapped_labels) > 0)

class TestCLiMBPhase1Primitives(unittest.TestCase):
    """
    CLiMB used to drop the fitted KBound on the floor, so anything wanting to
    explain Phase 1 had to re-derive the density kernel and reverse-engineer the
    pinned seeds by matching rows of X. Exposing kbound_ removes that coupling.
    """

    def setUp(self):
        X, y = make_blobs(n_samples=300, centers=4, n_features=3, random_state=7)
        self.X = StandardScaler().fit_transform(X)
        self.seeds = {
            tuple(self.X[y == c].mean(axis=0)):
                [self.X[np.where(y == c)[0][i]].tolist() for i in range(4)]
            for c in range(3)
        }
        self.climb = CLiMB(
            constrained_clusters=3, seed_points=self.seeds, density_threshold=0.03,
            distance_threshold=1.2, radial_threshold=2.0, convergence_tolerance=1e-6,
        ).fit(self.X)

    def test_fitted_kbound_is_exposed(self):
        kbound = self.climb.kbound_
        self.assertIsNotNone(kbound)
        self.assertIs(kbound.labels_, self.climb.constrained_labels)
        np.testing.assert_array_equal(kbound.centroids_, self.climb.constrained_centroids)

    def test_kbound_is_none_before_fit(self):
        self.assertIsNone(CLiMB(constrained_clusters=3).kbound_)

    def test_phase1_is_explainable_through_the_public_attributes(self):
        """The whole Phase-1 explanation without touching a private method."""
        kbound = self.climb.kbound_
        df = kbound.decision_path(self.X)

        self.assertEqual(kbound.fidelity_, 1.0)
        self.assertEqual(len(df), len(self.X))
        self.assertEqual(len(kbound.local_density_), len(self.X))
        self.assertTrue(kbound.seed_indices_)

    def test_density_primitive_matches_a_hand_rolled_replica(self):
        """
        local_density_ is what explain.py duplicated by hand. If the kernel in
        _compute_local_density ever changes, this is where the duplication would
        have started drifting in silence.
        """
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(self.X, metric="euclidean"))
        density = np.sum(np.exp(-0.5 * (distances / np.mean(distances)) ** 2), axis=1)
        np.testing.assert_allclose(density / np.max(density), self.climb.kbound_.local_density_)

    def test_rejected_points_are_exactly_what_phase2_receives(self):
        """The gate table accounts for the Phase-1/Phase-2 split with no remainder."""
        df = self.climb.kbound_.decision_path(self.X)
        rejected = df["label"] == -1
        self.assertEqual(int(rejected.sum()), len(self.climb.unassigned_points))
        np.testing.assert_array_equal(self.X[rejected.to_numpy()], self.climb.unassigned_points)
        self.assertTrue(
            df.loc[rejected, "gate"].isin(["rejected_density", "rejected_distance"]).all()
        )


if __name__ == '__main__':
    unittest.main()