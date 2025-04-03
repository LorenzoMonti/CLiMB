import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from ..utils.util import hungarian_match

class KBound:
    def __init__(
        self, 
        n_clusters, 
        seeds=None, 
        max_iter=300, 
        density_threshold=0.5,
        distance_threshold=2.0,
        radial_threshold=1.0,
        convergence_tolerance=0.1
    ):
        """
        Initialize 3D KBound (Constrained K-Means)
        
        Parameters:
        - n_clusters: Number of target clusters
        - seeds: Initial seed points for clustering
        - max_iter: Maximum iterations for convergence
        - density_threshold: Minimum local density required for cluster assignment
        - distance_threshold: Maximum distance from centroid for point retention
        - radial_threshold: Maximum radial centroid's distance
        - convergence_tolerance: defines the minimum movement required for centroids before the algorithm stops, 
        which helps balance computation efficiency and model accuracy.
        """

        self.n_clusters = n_clusters
        self.seeds = seeds
        self.max_iter = max_iter
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.radial_threshold = radial_threshold
        self.convergence_tolerance = convergence_tolerance
        self.mapped_labels_ = list()

    def _compute_local_density(self, X, sigma=None):
        """
        Calculate 3D local point density using Gaussian kernel
        
        Returns:
        - Normalized local density for each point
        """
        # Computes the pairwise Euclidean distance between all points in X
        # pdist return condensed distance matrix and squareform the expanded one
        distances = squareform(pdist(X))
        
        # sigma controls the sensitivity:
        # Small sigma → sharp kernel (local influence).
        # Large sigma → smoother kernel (global influence).
        if sigma is None:
            sigma = np.mean(distances)
        
        # Compute the density using a Gaussian kernel. 
        # It used to measure similarity between points based on their distance
        # This function assigns higher values to nearby points and lower values to distant points.
        density = np.sum(
            np.exp(-0.5 * (distances / sigma) ** 2), 
            axis=1
        )
        
        # normalization by max density point
        return density / np.max(density)
    
    def _initialize_centroids(self, X):
        """
        Intelligent centroid initialization strategy
        
        Handles various scenarios:
        - No seeds
        - Seed equal than clusters
        - Fewer seeds than clusters
        - More seeds than clusters
        """
        # No seeds: random initialization
        if self.seeds is None:
            return X[np.random.choice(len(X), self.n_clusters, replace=False)]
        
        # Convert seeds to numpy array for consistent handling
        seeds = np.array(self.seeds)

        # if len(seeds) == self.n_clusters
        centroids = seeds
        
        # More seeds than desired clusters
        if len(seeds) > self.n_clusters:
            # Select most dispersed seeds
            distances = cdist(seeds, seeds)
            np.fill_diagonal(distances, np.inf)
            
            selected_seed_indices = []
            while len(selected_seed_indices) < self.n_clusters:
                if not selected_seed_indices:
                    selected_seed_indices.append(0)
                else:
                    # Choose seed furthest from existing selections
                    candidates = [
                        i for i in range(len(seeds)) 
                        if i not in selected_seed_indices
                    ]
                    max_min_distance = -1
                    best_candidate = None
                    
                    for candidate in candidates:
                        min_dist = min(
                            cdist(
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
        
        # Fewer seeds than clusters
        elif len(seeds) < self.n_clusters:
            initial_centroids = seeds.copy()
            remaining_centroids = self.n_clusters - len(seeds)
            
            # Find points furthest from existing seeds
            distances_from_seeds = cdist(X, initial_centroids)
            furthest_point_indices = np.argsort(
                distances_from_seeds.min(axis=1)
            )[-remaining_centroids:]
            
            additional_centroids = X[furthest_point_indices]
            return np.vstack([initial_centroids, additional_centroids])
        
        # Exact number of seeds
        return centroids

    def fit(self, X, known_labels=None, is_adaptive=False):
        """
        Perform density-constrained clustering in 3 dimension with assignment constraints 
        (density & distance filtering) and movement constraints (adaptive radial threshold).
        
        Args:
        - X: Data points (numpy array)
        - known_labels: Labels corresponding to known centroids (numpy array of shape (n_clusters,))
        - is_adaptive: boolean to select adaptive radial threshold or static radial threshold (boolean)
        
        Return:
        - self
        """
        # Compute local point densities
        point_densities = self._compute_local_density(X)
        
        # Initialize centroids and store the original positions
        centroids = self._initialize_centroids(X)
        known_centroids = centroids.copy()
        initial_centroids = centroids.copy()  # Store initial positions
        
        for i in range(self.max_iter):
            # Compute distances to centroids
            distances = cdist(X, centroids)
            
            # Preliminary cluster assignments
            preliminary_labels = np.argmin(distances, axis=1)
            
            # Advanced filtering mechanism
            filtered_labels = preliminary_labels.copy()
            unassigned_mask = np.zeros(len(X), dtype=bool)
            
            for i in range(len(X)):
                # Density-based unassignment
                if point_densities[i] > 1 - self.density_threshold:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1  # Unassigned label
                    continue
                
                # Distance-based unassignment
                current_cluster = preliminary_labels[i]
                if distances[i, current_cluster] > self.distance_threshold:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1  # Unassigned label
                    continue
            
            # Compute new centroids
            new_centroids = np.array([
                X[filtered_labels == k].mean(axis=0) 
                if np.any(filtered_labels == k)
                else centroids[k]  # Fallback to previous centroid if no points
                for k in range(self.n_clusters)
            ])

            """
              Apply (adaptive) radial threshold constraint trying to prevents centroids from 
              drifting too much and maintains separation of clusters.
              If fixed radial_threshold doesn't work well, you could make it adaptive (is_adaptive=True) 
              based on local density trying to reduce centroid movement more in dense areas and allow more movement in sparse regions.
            """
            for k in range(self.n_clusters):
                displacement = new_centroids[k] - initial_centroids[k]
                distance_from_initial = np.linalg.norm(displacement)

                if is_adaptive:
                    # Compute adaptive threshold based on density
                    if np.any(filtered_labels == k):  # Avoid empty clusters
                        cluster_density = np.mean(point_densities[filtered_labels == k])
                        adaptive_threshold = self.radial_threshold * (1 - cluster_density)  # Reduce threshold in dense areas
                    else:
                        adaptive_threshold = self.radial_threshold  # Default if no points assigned

                    # If the centroid moves too far, scale back
                    if distance_from_initial > adaptive_threshold:
                        new_centroids[k] = initial_centroids[k] + (displacement / distance_from_initial) * adaptive_threshold
                    
                if distance_from_initial > self.radial_threshold:
                    # Scale displacement to stay within radial threshold
                    new_centroids[k] = initial_centroids[k] + (displacement / distance_from_initial) * self.radial_threshold

            # Updated convergence check, considering radial threshold constraint
            centroid_displacements = np.linalg.norm(new_centroids - centroids, axis=1)
            if np.all(centroid_displacements < self.convergence_tolerance) or np.all(centroid_displacements < self.radial_threshold):
                break
                
            centroids = new_centroids.copy()
        
        # If known_labels are provided, apply Hungarian algorithm for matching
        if known_labels is not None:
            cluster_mapping, mapped_labels = hungarian_match(known_centroids, centroids, known_labels, filtered_labels)
            self.mapped_labels_ = mapped_labels  # Corrected labels
            self.cluster_mapping_ = cluster_mapping  # Save mapping for reference
        else:
            self.mapped_labels_ = filtered_labels
            self.cluster_mapping_ = {i: i for i in range(self.n_clusters)}

        self.labels_ = filtered_labels
        self.original_centroids_ = known_centroids
        self.centroids_ = new_centroids  # Computed centroids
        self.point_densities_ = point_densities
        self.unassigned_mask_ = unassigned_mask

        return self

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
        if self.seeds is not None:
            ax1.scatter(
                self.seeds[:, 0],
                self.seeds[:, 1],
                self.seeds[:, 2],
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

        # Point Density Subplot
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=self.point_densities_,
            cmap='plasma',
            alpha=0.7
        )

        # Plot seed points with distinct marker style
        if self.seeds is not None:
            ax2.scatter(
                self.seeds[:, 0],
                self.seeds[:, 1],
                self.seeds[:, 2],
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
        if self.seeds is not None:
            ax3.scatter(
                self.seeds[:, 0],
                self.seeds[:, 1],
                self.seeds[:, 2],
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