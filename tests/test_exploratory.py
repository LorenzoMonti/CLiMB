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

class TestExploratoryExplain(unittest.TestCase):
    """
    Phase 2's algorithms do not share a decision structure, so explain() is
    implemented separately for each rather than generalised. These tests check
    each one against the model it claims to describe, and check that the ones
    without a closed form say so instead of inventing a guarantee.
    """

    def setUp(self):
        np.random.seed(42)
        X, _ = make_blobs(n_samples=300, centers=3, n_features=3, random_state=0)
        self.X = StandardScaler().fit_transform(X)
        X_moons, _ = make_moons(n_samples=250, noise=0.07, random_state=0)
        self.X_moons = StandardScaler().fit_transform(X_moons)

    # --- DBSCAN ----------------------------------------------------------

    def test_dbscan_core_set_matches_sklearn(self):
        """
        The invariant: the reconstructed core set is sklearn's own core set.
        sklearn derives core_sample_indices_ inside its implementation, so
        agreeing with it is evidence of the same rule rather than a lookalike.
        """
        for name, X in (("blobs", self.X), ("moons", self.X_moons)):
            with self.subTest(data=name):
                dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
                dbscan.fit_predict(X)
                df = dbscan.explain(X)

                self.assertEqual(dbscan.fidelity_, 1.0)
                expected = np.zeros(len(X), dtype=bool)
                expected[dbscan.model.core_sample_indices_] = True
                np.testing.assert_array_equal(df["is_core"].to_numpy(), expected)

                # Non-degenerate: all three roles must actually occur, or the
                # agreement above is trivial.
                self.assertEqual(set(df["role"]), {"core", "border", "noise"})

    def test_dbscan_core_distance_is_the_radius_that_flips_core_status(self):
        dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
        dbscan.fit_predict(self.X)
        df = dbscan.explain(self.X)

        np.testing.assert_array_equal(df["core_distance"] <= dbscan.eps, df["is_core"])
        np.testing.assert_allclose(df["eps_margin"], dbscan.eps - df["core_distance"])
        self.assertTrue((df.loc[df["is_core"], "eps_margin"] >= 0).all())
        self.assertTrue((df.loc[~df["is_core"], "eps_margin"] < 0).all())
        self.assertTrue((df["n_neighbors_eps"] >= 1).all(), "self must be counted")

    def test_dbscan_roles_agree_with_the_labels(self):
        dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(self.X)
        df = dbscan.explain(self.X)

        np.testing.assert_array_equal(df["cluster"], labels)
        self.assertTrue((df.loc[df["role"] == "noise", "cluster"] == -1).all())
        self.assertTrue((df.loc[df["role"] != "noise", "cluster"] != -1).all())
        self.assertTrue(df.loc[df["role"] == "core", "is_core"].all())
        self.assertFalse(df.loc[df["role"] == "border", "is_core"].any())

    def test_dbscan_distance_to_nearest_core_matches_brute_force(self):
        """
        Border points take the nearest core; core points take the nearest core
        other than themselves. Computing it uniformly from the second neighbour
        -- correct only for core points, whose first neighbour is themselves --
        hands every border point its second-nearest core instead.
        """
        from scipy.spatial.distance import cdist

        dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(self.X)
        df = dbscan.explain(self.X)
        is_core = df["is_core"].to_numpy()

        expected = np.full(len(self.X), np.nan)
        for cluster in np.unique(labels[labels != -1]):
            members = np.where(labels == cluster)[0]
            cores = members[is_core[members]]
            for member in members:
                others = cores[cores != member]
                if len(others):
                    expected[member] = cdist(self.X[member:member + 1], self.X[others]).min()

        got = df["distance_to_nearest_core"].to_numpy()
        np.testing.assert_array_equal(np.isnan(expected), np.isnan(got))
        np.testing.assert_allclose(expected[~np.isnan(expected)], got[~np.isnan(got)])

        border = (~is_core) & (labels != -1)
        self.assertGreater(border.sum(), 0, "no border points: nothing is being tested")

    def test_dbscan_fidelity_check_fires_on_a_mismatched_matrix(self):
        dbscan = DBSCANExploratory(eps=0.3, min_samples=5)
        dbscan.fit_predict(self.X)
        with self.assertRaises(RuntimeError):
            dbscan.explain(np.random.RandomState(0).randn(*self.X.shape))
        df = dbscan.explain(np.random.RandomState(0).randn(*self.X.shape),
                            check_fidelity=False)
        self.assertEqual(len(df), len(self.X))
        self.assertLess(dbscan.fidelity_, 1.0)

    # --- OPTICS ----------------------------------------------------------

    def test_optics_core_distances_match_sklearn(self):
        optics = OPTICSExploratory(min_samples=5)
        optics.fit_predict(self.X)
        df = optics.explain(self.X)

        self.assertEqual(optics.fidelity_, 1.0)
        np.testing.assert_allclose(df["core_distance"], optics.model.core_distances_)

    def test_optics_order_position_inverts_the_ordering(self):
        """Sorting the table by order_position reproduces the reachability plot."""
        optics = OPTICSExploratory(min_samples=5)
        optics.fit_predict(self.X)
        df = optics.explain(self.X)

        order_position = df["order_position"].to_numpy()
        self.assertEqual(sorted(order_position.tolist()), list(range(len(self.X))))
        np.testing.assert_array_equal(np.argsort(order_position), optics.model.ordering_)

        profile = df.sort_values("order_position")["reachability"].to_numpy()
        expected = optics.model.reachability_[optics.model.ordering_]
        # The walk's first point was never reached, so its reachability is inf.
        np.testing.assert_allclose(profile[1:], expected[1:])
        self.assertTrue(np.isinf(profile[0]))

    def test_optics_reports_no_core_border_split(self):
        """OPTICS fixes no radius, so it must not borrow DBSCAN's vocabulary."""
        optics = OPTICSExploratory(min_samples=5)
        optics.fit_predict(self.X)
        df = optics.explain(self.X)
        self.assertNotIn("role", df.columns)
        self.assertNotIn("is_core", df.columns)
        self.assertNotIn("eps_margin", df.columns)

    # --- HDBSCAN ---------------------------------------------------------

    def test_hdbscan_declares_it_has_no_reconstruction(self):
        """
        HDBSCAN's labels come from the stability of a condensed tree; there is no
        closed form to replay. fidelity_ must stay None rather than report a 1.0
        that would describe nothing.
        """
        model = HDBSCANExploratory(min_cluster_size=10)
        model.fit_predict(self.X)
        model.explain(self.X)
        self.assertIsNone(model.fidelity_)

    def test_hdbscan_reports_membership_by_stability(self):
        model = HDBSCANExploratory(min_cluster_size=10)
        labels = model.fit_predict(self.X)
        df = model.explain(self.X)

        np.testing.assert_array_equal(df["cluster"], labels)
        self.assertTrue(((df["membership_probability"] >= 0) &
                         (df["membership_probability"] <= 1)).all())
        self.assertTrue((df.loc[df["is_noise"], "membership_probability"] == 0).all())
        self.assertTrue(df.loc[df["is_noise"], "cluster_persistence"].isna().all())
        self.assertFalse(df.loc[~df["is_noise"], "cluster_persistence"].isna().any())

        # Persistence is a per-cluster score, identical across a cluster's members.
        for _, group in df[~df["is_noise"]].groupby("cluster"):
            self.assertEqual(group["cluster_persistence"].nunique(), 1)

        self.assertNotIn("role", df.columns)
        self.assertNotIn("is_core", df.columns)

    # --- shared contract --------------------------------------------------

    def test_explain_is_required_by_the_base_class(self):
        class Incomplete(ExploratoryClusteringBase):
            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

            def get_name(self):
                return "Incomplete"

            def get_parameters(self):
                return ""

        with self.assertRaises(TypeError):
            Incomplete()

    def test_explain_guards_apply_to_every_implementation(self):
        for build in (lambda: DBSCANExploratory(eps=0.3, min_samples=5),
                      lambda: OPTICSExploratory(min_samples=5),
                      lambda: HDBSCANExploratory(min_cluster_size=10)):
            model = build()
            with self.subTest(algorithm=model.get_name()):
                with self.assertRaises(RuntimeError):
                    model.explain(self.X)          # not fitted yet

                model.fit_predict(self.X)
                with self.assertRaises(ValueError):
                    model.explain(self.X[:50])     # wrong number of rows

                df = model.explain(self.X)
                self.assertEqual(len(df), len(self.X))
                self.assertEqual(df["point_index"].tolist(), list(range(len(self.X))))
                self.assertIn("cluster", df.columns)
