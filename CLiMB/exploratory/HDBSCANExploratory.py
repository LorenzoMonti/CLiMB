import hdbscan
from . import ExploratoryClusteringBase

class HDBSCANExploratory(ExploratoryClusteringBase):
    """
    HDBSCAN for exploratory clustering
    """
    def __init__(self, min_cluster_size=5, min_samples=None):
        """
        Initialize HDBSCAN exploratory clustering
        
        Parameters:
        -----------
        min_cluster_size : int, default=5
            The minimum size of clusters to be considered.
            
        min_samples : int, default=None
            The number of samples in a neighborhood for a point to be considered
            as a core point.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
    def fit_predict(self, X):
        """
        Perform HDBSCAN clustering on X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to cluster.
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. Noisy samples are given the label -1.
        """
        return self.model.fit_predict(X)
        
    def get_name(self):
        return f"HDBSCAN"
    
    def get_parameters(self):
        return f"min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}"

