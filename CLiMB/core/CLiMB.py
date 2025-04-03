import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .KBound import KBound
from ..exploratory import ExploratoryClusteringBase
from ..exploratory.DBSCANExploratory import DBSCANExploratory

class CLiMB:
    """
    CLustering In Multiphase Boundaries (CLiMB)
    
    A two-phase clustering algorithm designed for datasets with both known 
    and exploratory components. First phase constrains clusters around known regions,
    second phase identifies new patterns in unassigned points.
    """
    
    def __init__(self, 
                 constrained_clusters=3, 
                 seed_points=None,
                 density_threshold=0.2, 
                 distance_threshold=15,
                 radial_threshold=1,
                 convergence_tolerance=0.4,
                 exploratory_algorithm=None):
        """
        Initialize CLiMB clustering algorithm
        
        Parameters:
        -----------
        constrained_clusters : int, default=3
            Number of clusters in first stage (constrained clustering)
            
        seed_points : array-like, default=None
            Initial known cluster centers (optional)
            
        density_threshold : float, default=0.2
            Minimum local density required for cluster assignment
            
        distance_threshold : float, default=15
            Maximum distance from centroid for point retention
            
        radial_threshold : float, default=1
            Maximum radial centroid's distance
            
        convergence_tolerance : float, default=0.4
            Minimum movement required for centroids before algorithm stops
            
        exploratory_algorithm : ExploratoryClusteringBase, default=None
            Algorithm for exploratory clustering phase. If None, defaults to DBSCAN.
        """
        self.constrained_clusters = constrained_clusters
        self.seed_points = seed_points
        self.density_threshold = density_threshold 
        self.distance_threshold = distance_threshold
        self.radial_threshold = radial_threshold
        self.convergence_tolerance = convergence_tolerance
        
        # Default to DBSCAN if no exploratory algorithm is provided
        if exploratory_algorithm is None:
            self.exploratory_algorithm = DBSCANExploratory(eps=0.5, min_samples=3)
        else:
            if not isinstance(exploratory_algorithm, ExploratoryClusteringBase):
                raise TypeError("exploratory_algorithm must be an instance of ExploratoryClusteringBase")
            self.exploratory_algorithm = exploratory_algorithm
        
        # Tracking clustering results
        self.mapped_labels = None
        self.constrained_labels = None
        self.density_constrained_labels = None
        self.constrained_seeds = None
        self.constrained_centroids = None
        self.original_centroids = None
        self.exploratory_labels = None
        self.signed_points = None
        self.unassigned_points = None
    
    def set_density(self, density):
        """Set density threshold parameter"""
        self.density_threshold = density
        return self
    
    def set_distance(self, distance):
        """Set distance threshold parameter"""
        self.distance_threshold = distance
        return self

    def set_radial(self, radial):
        """Set radial threshold parameter"""
        self.radial_threshold = radial
        return self

    def set_convergence(self, convergence):
        """Set convergence tolerance parameter"""
        self.convergence_tolerance = convergence
        return self
        
    def set_exploratory_algorithm(self, exploratory_algorithm):
        """
        Set the exploratory clustering algorithm
        
        Parameters:
        -----------
        exploratory_algorithm : ExploratoryClusteringBase
            Algorithm for exploratory clustering phase
        """
        if not isinstance(exploratory_algorithm, ExploratoryClusteringBase):
            raise TypeError("exploratory_algorithm must be an instance of ExploratoryClusteringBase")
        self.exploratory_algorithm = exploratory_algorithm
        return self

    def fit(self, X, known_labels=None, is_adaptive=False):
        """
        Execute two-stage clustering process
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to cluster.
            
        known_labels : array-like, default=None
            Known labels for seed points, if available.
            
        is_adaptive : bool, default=False
            Whether to use adaptive radial threshold.
            
        Returns:
        --------
        self : CLiMB
            Fitted estimator.
        """    
        # Stage 1: Constrained K-Means (KBound)
        constrained_kmeans = KBound(
            n_clusters=self.constrained_clusters, 
            seeds=self.seed_points,
            density_threshold=self.density_threshold,
            distance_threshold=self.distance_threshold,
            radial_threshold=self.radial_threshold,
            convergence_tolerance=self.convergence_tolerance,
        )
  
        constrained_kmeans.fit(X, known_labels, is_adaptive=is_adaptive)
        
        self.mapped_labels = constrained_kmeans.mapped_labels_
        self.constrained_labels = constrained_kmeans.labels_
        self.constrained_seeds = constrained_kmeans.seeds if hasattr(constrained_kmeans, 'seeds') else None
        self.constrained_centroids = constrained_kmeans.centroids_
        self.original_centroids = constrained_kmeans.original_centroids_
        
        # Identify points for each phase
        self.signed_points = X[self.constrained_labels != -1]
        self.unassigned_points = X[self.constrained_labels == -1]
        self.density_constrained_labels = constrained_kmeans.labels_[constrained_kmeans.labels_ != -1]
        
        # Stage 2: Exploratory Clustering
        if len(self.unassigned_points) > 0:
            exploratory_labels = self.exploratory_algorithm.fit_predict(self.unassigned_points)
            
            # Offset exploratory labels to avoid conflict with constrained labels
            max_constrained_label = np.max(self.constrained_labels) if np.max(self.constrained_labels) >= 0 else -1
            self.exploratory_labels = np.array([
                label if label == -1 else label + max_constrained_label + 1 
                for label in exploratory_labels
            ])
        else:
            self.exploratory_labels = np.array([])

        return self
    
    def inverse_transform(self, scaler):
        """
        Transform clustering results back to original scale
        
        Parameters:
        -----------
        scaler : object with inverse_transform method
            The scaler used to normalize the data
        """
        if self.signed_points is not None:
            self.signed_points = scaler.inverse_transform(self.signed_points)
        if self.constrained_seeds is not None:
            self.constrained_seeds = scaler.inverse_transform(self.constrained_seeds)
        if self.constrained_centroids is not None:
            self.constrained_centroids = scaler.inverse_transform(self.constrained_centroids)
        if self.original_centroids is not None:
            self.original_centroids = scaler.inverse_transform(self.original_centroids)
        if self.unassigned_points is not None and len(self.unassigned_points) > 0:
            self.unassigned_points = scaler.inverse_transform(self.unassigned_points)
        return self
    
    def get_labels(self):
        """
        Get the cluster labels for all points
        
        Returns:
        --------
        labels : ndarray
            Cluster labels for each point
        """
        # Create a label array of the correct size
        all_labels = np.full(self.constrained_labels.shape, -1)
        
        # Fill in the constrained labels
        constrained_indices = (self.constrained_labels != -1)
        all_labels[constrained_indices] = self.constrained_labels[constrained_indices]
        
        # Fill in the exploratory labels
        if self.exploratory_labels is not None and len(self.exploratory_labels) > 0:
            exploratory_indices = (self.constrained_labels == -1)
            all_labels[exploratory_indices] = self.exploratory_labels
            
        return all_labels

    def plot_comprehensive_3d(self, axis_labels=None, save_path=None, figsize=(15, 6), dpi=300):
        """
        Visualize both clustering stages in 3D
        
        Parameters:
        -----------
        axis_labels : list, default=None
            Labels for X, Y, and Z axes
            
        save_path : str, default=None
            Path to save the figure
            
        figsize : tuple, default=(15, 6)
            Figure size
            
        dpi : int, default=300
            Figure resolution
        """
        if self.unassigned_points is None:
            raise ValueError("Must call fit() first")
        
        if axis_labels is None:
            axis_labels = ['X', 'Y', 'Z']
        
        # Check dimensionality
        if self.signed_points.shape[1] < 3:
            raise ValueError("3D visualization requires at least 3 dimensions in data")
        
        # 3D visualization
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # Constrained Clustering Subplot
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(
            self.signed_points[:, 0], self.signed_points[:, 1], self.signed_points[:, 2],
            c=self.density_constrained_labels, 
            cmap='rainbow',
            s=1
        )
        ax1.set_title('Constrained 3D Clustering')
        ax1.set_xlabel(axis_labels[0])
        ax1.set_ylabel(axis_labels[1])
        ax1.set_zlabel(axis_labels[2])
        fig.colorbar(scatter1, ax=ax1, shrink=0.6)
        
        # Plot original centroids with distinct marker style
        if self.original_centroids is not None:
            ax1.scatter(
                self.original_centroids[:, 0],
                self.original_centroids[:, 1],
                self.original_centroids[:, 2],
                c='red',
                marker='x',
                s=50,
                linewidth=3,
                label='Original Centroids'
            )

        # Plot final centroids
        ax1.scatter(
            self.constrained_centroids[:, 0],
            self.constrained_centroids[:, 1],
            self.constrained_centroids[:, 2],
            c='black',
            marker='^',
            s=50,
            label='Centroids'
        )
        ax1.legend()

        # Exploratory Clustering Subplot
        ax2 = fig.add_subplot(122, projection='3d')
        if len(self.unassigned_points) > 0:
            scatter2 = ax2.scatter(
                self.unassigned_points[:, 0], 
                self.unassigned_points[:, 1],
                self.unassigned_points[:, 2],
                c=self.exploratory_labels, 
                cmap='plasma',
                s=0.4,
                alpha=0.8
            )
            ax2.set_title(f'Exploratory 3D Clustering ({self.exploratory_algorithm.get_name()})')
            ax2.set_xlabel(axis_labels[0])
            ax2.set_ylabel(axis_labels[1])
            ax2.set_zlabel(axis_labels[2])
            fig.colorbar(scatter2, ax=ax2, shrink=0.6)
        else:
            ax2.set_title('No Unassigned Points for Exploratory Clustering')
            ax2.set_xlabel(axis_labels[0])
            ax2.set_ylabel(axis_labels[1])
            ax2.set_zlabel(axis_labels[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig

    def plot_comprehensive_2d(self, dimensions=(0, 1), axis_labels=None, save_path=None, figsize=(15, 6), dpi=300):
        """
        Visualize both clustering stages in 2D
        
        Parameters:
        -----------
        dimensions : tuple, default=(0, 1)
            Indices of dimensions to plot
            
        axis_labels : list, default=None
            Labels for X and Y axes
            
        save_path : str, default=None
            Path to save the figure
            
        figsize : tuple, default=(15, 6)
            Figure size
            
        dpi : int, default=300
            Figure resolution
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.unassigned_points is None:
            raise ValueError("Must call fit() first")
        
        if axis_labels is None:
            axis_labels = ['X', 'Y']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        dim1, dim2 = dimensions
        
        # Constrained Clustering Subplot
        scatter1 = ax1.scatter(
            self.signed_points[:, dim1], self.signed_points[:, dim2],
            c=self.density_constrained_labels, 
            cmap='rainbow',
            s=1
        )
        ax1.set_title('Constrained 2D Clustering')
        ax1.set_xlabel(axis_labels[0])
        ax1.set_ylabel(axis_labels[1])
        fig.colorbar(scatter1, ax=ax1, shrink=0.6)
        
        # Plot cluster original centroids
        if self.original_centroids is not None:
            ax1.scatter(
                self.original_centroids[:, dim1],
                self.original_centroids[:, dim2],
                c='red',
                marker='x',
                s=100,
                linewidth=3,
                label='Original centroids'
            )

        # Plot cluster centroids
        ax1.scatter(
            self.constrained_centroids[:, dim1],
            self.constrained_centroids[:, dim2],
            c='black',
            marker='^',
            s=100,
            label='Constrained centroids'
        )
        ax1.legend()

        # Exploratory Clustering Subplot
        if len(self.unassigned_points) > 0:
            scatter2 = ax2.scatter(
                self.unassigned_points[:, dim1], 
                self.unassigned_points[:, dim2],
                c=self.exploratory_labels, 
                cmap='viridis',
                s=0.4,
                alpha=0.8
            )
            ax2.set_title(f'Exploratory 2D Clustering ({self.exploratory_algorithm.get_name()})')
            ax2.set_xlabel(axis_labels[0])
            ax2.set_ylabel(axis_labels[1])
            fig.colorbar(scatter2, ax=ax2, shrink=0.6)
        else:
            ax2.set_title('No Unassigned Points for Exploratory Clustering')
            ax2.set_xlabel(axis_labels[0])
            ax2.set_ylabel(axis_labels[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig