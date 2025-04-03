import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles

# Import your modules (adjust paths as needed)
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from CLiMB.exploratory.HDBSCANExploratory import HDBSCANExploratory
from CLiMB.exploratory.OPTICSExploratory import OPTICSExploratory
from CLiMB.exploratory import ExploratoryClusteringBase

class TestExploratoryAlgorithms(unittest.TestCase):
    """Tests for specific exploratory clustering algorithms"""
    
    def setUp(self):
        """Create synthetic datasets for testing"""
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
        
        # Create circles dataset
        self.X_circles, self.y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
        # Add a third dimension to circles
        self.X_circles = np.column_stack((
            self.X_circles, 
            np.random.normal(0, 0.1, size=self.X_circles.shape[0])
        ))
        
        # Standardize datasets
        self.scaler = StandardScaler()
        self.X_blobs_scaled = self.scaler.fit_transform(self.X_blobs)
        self.X_moons_scaled = self.scaler.fit_transform(self.X_moons)
        self.X_circles_scaled = self.scaler.fit_transform(self.X_circles)
    
    def test_dbscan_exploratory(self):
        """Test DBSCANExploratory algorithm"""
        # Create algorithm with default parameters
        dbscan = DBSCANExploratory()
        
        # Test name
        self.assertEqual(dbscan.get_name(), "DBSCAN")
        
        # Test on blobs dataset
        blobs_labels = dbscan.fit_predict(self.X_blobs_scaled)
        self.assertEqual(len(blobs_labels), len(self.X_blobs_scaled))
        
        # There should be some clusters found
        unique_labels = np.unique(blobs_labels[blobs_labels >= 0])
        self.assertTrue(len(unique_labels) > 0)
        
        # Test with custom parameters
        custom_dbscan = DBSCANExploratory(eps=0.5, min_samples=10)
        custom_labels = custom_dbscan.fit_predict(self.X_blobs_scaled)
        self.assertEqual(len(custom_labels), len(self.X_blobs_scaled))
    
    def test_hdbscan_exploratory(self):
        """Test HDBSCANExploratory algorithm"""
        try:
            # HDBSCAN is optional, so check if it's available
            import hdbscan
            
            # Create algorithm with default parameters
            hdb = HDBSCANExploratory()
            
            # Test name
            self.assertEqual(hdb.get_name(), "HDBSCAN")
            
            # Test on blobs dataset
            blobs_labels = hdb.fit_predict(self.X_blobs_scaled)
            self.assertEqual(len(blobs_labels), len(self.X_blobs_scaled))
            
            # There should be some clusters found
            unique_labels = np.unique(blobs_labels[blobs_labels >= 0])
            self.assertTrue(len(unique_labels) > 0)
            
            # Test with custom parameters
            custom_hdb = HDBSCANExploratory(min_cluster_size=10, min_samples=5)
            custom_labels = custom_hdb.fit_predict(self.X_blobs_scaled)
            self.assertEqual(len(custom_labels), len(self.X_blobs_scaled))
            
        except ImportError:
            # Skip test if HDBSCAN is not available
            self.skipTest("HDBSCAN package not available")
    
    def test_optics_exploratory(self):
        """Test OPTICSExploratory algorithm"""
        # Create algorithm with default parameters
        optics = OPTICSExploratory()
        
        # Test name
        self.assertEqual(optics.get_name(), "OPTICS")
        
        # Test on blobs dataset
        blobs_labels = optics.fit_predict(self.X_blobs_scaled)
        self.assertEqual(len(blobs_labels), len(self.X_blobs_scaled))
        
        # There should be some clusters found
        unique_labels = np.unique(blobs_labels[blobs_labels >= 0])
        self.assertTrue(len(unique_labels) > 0)
        
        # Test with custom parameters
        custom_optics = OPTICSExploratory(min_samples=10)
        custom_labels = custom_optics.fit_predict(self.X_blobs_scaled)
        self.assertEqual(len(custom_labels), len(self.X_blobs_scaled))
    
    def test_algorithm_comparison(self):
        """Compare different exploratory algorithms on complex shapes"""
        # Create instances of each algorithm
        dbscan = DBSCANExploratory(eps=0.3, min_samples=3)
        optics = OPTICSExploratory(min_samples=15)
        
        # Run on moons dataset
        dbscan_moons = dbscan.fit_predict(self.X_moons_scaled)
        optics_moons = optics.fit_predict(self.X_moons_scaled)
        
        # Both should find clusters
        dbscan_clusters = len(np.unique(dbscan_moons[dbscan_moons >= 0]))
        optics_clusters = len(np.unique(optics_moons[optics_moons >= 0]))
        
        self.assertTrue(dbscan_clusters > 0)
        self.assertTrue(optics_clusters > 0)
        
        # Run on circles dataset
        dbscan_circles = dbscan.fit_predict(self.X_circles_scaled)
        optics_circles = optics.fit_predict(self.X_circles_scaled)
        
        # Both should find clusters
        dbscan_clusters = len(np.unique(dbscan_circles[dbscan_circles >= 0]))
        optics_clusters = len(np.unique(optics_circles[optics_circles >= 0]))
        
        self.assertTrue(dbscan_clusters > 0)
        self.assertTrue(optics_clusters > 0)
    
    def test_exploratory_integration(self):
        """Test integration with CLiMB main class"""
        from CLiMB.core.CLiMB import CLiMB
        
        # Create seed points from known clusters
        seed_points = np.array([
            self.X_blobs[self.y_blobs == 0].mean(axis=0),
            self.X_blobs[self.y_blobs == 1].mean(axis=0),
            self.X_blobs[self.y_blobs == 2].mean(axis=0)
        ])
        seed_points_scaled = self.scaler.transform(seed_points)
        
        # Create CLiMB instance with custom exploratory algorithm
        custom_dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
        climb = CLiMB(
            constrained_clusters=3,
            seed_points=seed_points_scaled
        )
        climb.set_exploratory_algorithm(custom_dbscan)
        
        # Fit on blob dataset
        climb.fit(self.X_blobs_scaled)
        
        # Check that the exploratory algorithm is properly set
        self.assertIs(climb.exploratory_algorithm, custom_dbscan)
        
        # Get the final labels
        labels = climb.get_labels()
        self.assertEqual(len(labels), len(self.X_blobs_scaled))

if __name__ == '__main__':
    unittest.main()