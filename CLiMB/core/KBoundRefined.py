from .KBound import KBound
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KernelDensity

from ..utils.util import hungarian_match # Assuming this utility exists in the specified path, otherwise replace with your implementation

class KBoundRefined:
    def __init__(
        self,
        n_clusters,
        seeds=None,
        max_iter=300,
        density_threshold=0.5,
        distance_threshold=2.0,
        radial_threshold=1.0,
        convergence_tolerance=0.1,
        kde_bandwidth=None # Bandwidth for KDE, let's add this as parameter
    ):
        """
        Initialize 3D KBoundRefined (Density-Constrained K-Means + Mean Shift Refinement)

        Parameters:
        - n_clusters: Number of target clusters
        - seeds: Initial seed points for clustering
        - max_iter: Maximum iterations for KBound convergence
        - density_threshold: Minimum local density required for cluster assignment in KBound
        - distance_threshold: Maximum distance from centroid for point retention in KBound
        - radial_threshold: Maximum radial centroid's distance in KBound
        - convergence_tolerance: Convergence tolerance for KBound
        - kde_bandwidth: Bandwidth for Kernel Density Estimation, if None, estimated from data
        """
        self.kbound = KBound(
            n_clusters=n_clusters,
            seeds=seeds,
            max_iter=max_iter,
            density_threshold=density_threshold,
            distance_threshold=distance_threshold,
            radial_threshold=radial_threshold,
            convergence_tolerance=convergence_tolerance
        )
        self.kde_bandwidth = kde_bandwidth
        self.refined_labels_ = None
        self.refined_centroids_ = None

    def fit(self, X, known_labels=None, is_slight_movement=False):
        """
        Fit KBound and refine clusters using Mean Shift on KDE representation.

        Args:
        - X: Data points (numpy array)
        - known_labels: Labels corresponding to known centroids (numpy array of shape (n_clusters,))
        - is_slight_movement: boolean to control centroid movement constraint behavior in KBound.

        Returns:
        - self
        """
        # Run KBound clustering
        self.kbound.fit(X, known_labels=known_labels, is_slight_movement=is_slight_movement)

        # Generate KDE representation from KBound clusters
        kde_representation_points = []
        for cluster_label in range(self.kbound.n_clusters):
            cluster_points = X[self.kbound.labels_ == cluster_label]
            if len(cluster_points) > 0:
                # Option 1: Use cluster centroids as representation
                kde_representation_points.append(self.kbound.centroids_[cluster_label])

                # Option 2: Use points sampled from KDE (more complex, for future improvement)
                # kde = KernelDensity(bandwidth=self.kde_bandwidth if self.kde_bandwidth else 'scott', kernel='gaussian')
                # kde.fit(cluster_points)
                # num_samples = len(cluster_points) # Sample same number of points as in the cluster
                # samples = kde.sample(n_samples)
                # kde_representation_points.extend(samples)


        kde_representation_points = np.array(kde_representation_points)

        if len(kde_representation_points) > 0: # Proceed only if we have representation points
            # Apply Mean Shift on KDE representation
            bandwidth = estimate_bandwidth(kde_representation_points, quantile=0.2) if self.kde_bandwidth is None else self.kde_bandwidth
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True) # bin_seeding for speed
            ms.fit(kde_representation_points)
            self.refined_labels_ = ms.labels_
            self.refined_centroids_ = ms.cluster_centers_
        else:
            # Handle case where no KDE representation points are generated (e.g., empty clusters)
            self.refined_labels_ = np.array([]) # Or handle as appropriate
            self.refined_centroids_ = np.array([])

        return self

    def visualize_refined_clustering(self, X):
        """
        Visualize the refined clustering results alongside the original KBound results.
        """
        fig = plt.figure(figsize=(18, 6), dpi=100)

        # KBound Clustering Results Subplot
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=self.kbound.labels_,
            cmap='viridis',
            alpha=0.7
        )
        ax1.scatter(
            self.kbound.centroids_[:, 0],
            self.kbound.centroids_[:, 1],
            self.kbound.centroids_[:, 2],
            c='red',
            marker='^',
            s=100,
            label='KBound Centroids'
        )
        ax1.set_title('KBound Clustering')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()


        # Refined Clustering (Mean Shift) Results Subplot
        ax2 = fig.add_subplot(122, projection='3d')

        if self.refined_labels_ is not None and len(self.refined_centroids_) > 0 and len(self.refined_labels_) == len(self.kbound.centroids_):
            # Map refined labels back to original data points. Simple 1-to-1 mapping for now as we used centroids as representation.
            refined_data_labels = np.full(len(X), -1) # Initialize with unassigned
            for i, original_centroid in enumerate(self.kbound.centroids_):
                # Find points closest to the original centroid (approximation for mapping back)
                distances = np.linalg.norm(X - original_centroid, axis=1)
                closest_point_index = np.argmin(distances) #  This is a simplification. Better mapping needed for sampled KDE points.
                refined_data_labels[self.kbound.labels_ == i] = self.refined_labels_[i] # Assign refined label based on original cluster label

            scatter2 = ax2.scatter(
                X[:, 0], X[:, 1], X[:, 2],
                c=refined_data_labels,
                cmap='viridis',
                alpha=0.7
            )
            ax2.scatter(
                self.refined_centroids_[:, 0],
                self.refined_centroids_[:, 1],
                self.refined_centroids_[:, 2],
                c='red',
                marker='^',
                s=100,
                label='Refined Centroids (Mean Shift)'
            )
        else:
            ax2.text(0.5, 0.5, 'Mean Shift did not produce valid clusters.', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)


        ax2.set_title('Refined Clustering (Mean Shift on Centroids)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()


        plt.tight_layout()
        plt.show()
        return fig


# --- Utility function (assuming it's defined or replace with a dummy for testing if not available) ---
def hungarian_match(known_centroids, centroids, known_labels, filtered_labels):
    """
    Dummy implementation for testing. Replace with your actual hungarian_match logic.
    """
    n_clusters = len(known_centroids)
    cluster_mapping = {i: i for i in range(n_clusters)} # Identity mapping for dummy
    mapped_labels = filtered_labels # No actual mapping in dummy
    return cluster_mapping, mapped_labels


# --- Example Usage ---
if __name__ == '__main__':
    # Generate synthetic 3D data
    np.random.seed(42)
    cluster1 = np.random.randn(50, 3) + np.array([2, 2, 2])
    cluster2 = np.random.randn(50, 3) + np.array([-2, -2, -2])
    cluster3 = np.random.randn(50, 3) + np.array([2, -2, -2])
    X = np.vstack([cluster1, cluster2, cluster3])

    # Instantiate and fit KBoundRefined
    refined_kbound = KBoundRefined(n_clusters=3, seeds=np.array([[2, 2, 2], [-2, -2, -2], [2, -2, -2]]))
    refined_kbound.fit(X)

    # Visualize the refined clustering
    refined_kbound.visualize_refined_clustering(X)