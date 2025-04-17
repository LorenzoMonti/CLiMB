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
        convergence_tolerance=0.1,
        distance_metric='euclidean',  # Added distance_metric parameter
        metric_params=None # Added metric_params to pass parameters to the distance metric
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
        self.distance_metric = distance_metric # Store the distance metric
        self.metric_params = metric_params # Store metric parameters


    def _compute_local_density(self, X, sigma=None):
        """
        Calculate 3D local point density using Gaussian kernel
        (Using Euclidean distance for density calculation as it's about local proximity)

        Returns:
        - Normalized local density for each point
        """
        distances = squareform(pdist(X, metric='euclidean')) # Always euclidean for density

        if sigma is None:
            sigma = np.mean(distances)

        density = np.sum(
            np.exp(-0.5 * (distances / sigma) ** 2),
            axis=1
        )
        return density / np.max(density)

    def _initialize_centroids(self, X):
        """
        Intelligent centroid initialization strategy
        """
        if self.seeds is None:
            return X[np.random.choice(len(X), self.n_clusters, replace=False)]

        seeds = np.array(self.seeds)

        if len(seeds) == self.n_clusters:
            centroids = seeds

        elif len(seeds) > self.n_clusters:
            distances = self._cdist_custom(seeds, seeds) # Use custom distance
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
                            self._cdist_custom( # Use custom distance
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

            distances_from_seeds = self._cdist_custom(X, initial_centroids) # Use custom distance
            furthest_point_indices = np.argsort(
                distances_from_seeds.min(axis=1)
            )[-remaining_centroids:]

            additional_centroids = X[furthest_point_indices]
            return np.vstack([initial_centroids, additional_centroids])

        return centroids


    def _cdist_custom(self, XA, XB):
        """
        Wrapper for cdist with custom distance metric handling.
        """
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
                # cdist does not directly support arbitrary custom functions in the same way,
                # we might need to iterate and apply the custom function.
                # This is a simplified approach, might need more efficient implementation for large datasets.
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
        Perform density-constrained clustering with radial threshold constraints and custom distance metrics.
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
                # Handle non-invertible covariance matrix (e.g., use pseudoinverse)
                inv_covariance_matrix = np.linalg.pinv(covariance_matrix)
            self.metric_params = {'VI': inv_covariance_matrix} # Store for use in _cdist_custom


        for iteration in range(self.max_iter):
            print(f"\n--- Iterazione {iteration} ---")
            print("Centroidi all'inizio iterazione:\n", centroids)

            prev_centroids = centroids.copy()

            # Compute distances to centroids using custom distance function
            distances = self._cdist_custom(X, centroids)

            preliminary_labels = np.argmin(distances, axis=1)

            filtered_labels = preliminary_labels.copy()
            unassigned_mask = np.zeros(len(X), dtype=bool)
            for i in range(len(X)):
                if point_densities[i] > 1 - self.density_threshold:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1
                elif distances[i, preliminary_labels[i]] > self.distance_threshold:
                    unassigned_mask[i] = True
                    filtered_labels[i] = -1

            new_centroids_raw = np.array([
                X[filtered_labels == k].mean(axis=0) if np.any(filtered_labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])
            new_centroids = new_centroids_raw.copy()

            print("Nuovi centroidi PRIMA vincoli radiali:\n", new_centroids_raw)

            for k in range(self.n_clusters):
                displacement = new_centroids_raw[k] - initial_centroids[k]
                distance_from_initial = np.linalg.norm(displacement)

                if is_slight_movement:
                    if distance_from_initial > self.radial_threshold:
                        print(f"Centroide {k}: is_slight_movement=True - Limito spostamento")
                        new_centroids[k] = initial_centroids[k] + (displacement / distance_from_initial) * self.radial_threshold
                    else:
                        print(f"Centroide {k}: is_slight_movement=True - Spostamento OK")
                else:
                    if distance_from_initial > self.radial_threshold:
                        print(f"Centroide {k}: is_slight_movement=False - ROLLBACK CLUSTER {k}!")
                        new_centroids[k] = prev_centroids[k]
                    else:
                        print(f"Centroide {k}: is_slight_movement=False - Spostamento OK")


            print("Nuovi centroidi DOPO vincoli radiali:\n", new_centroids)
            centroid_displacements = np.linalg.norm(new_centroids - centroids, axis=1)
            print("Spostamenti centroidi:\n", centroid_displacements)

            if np.all(centroid_displacements < self.convergence_tolerance):
                print("Convergenza raggiunta!")
                break

            centroids = new_centroids.copy()
            print("Centroidi aggiornati:\n", centroids)

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