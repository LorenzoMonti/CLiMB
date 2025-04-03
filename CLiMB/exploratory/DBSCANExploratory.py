from sklearn.cluster import DBSCAN
from . import ExploratoryClusteringBase

class DBSCANExploratory(ExploratoryClusteringBase):
    """
    DBSCAN for exploratory clustering
    """
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN exploratory clustering
        
        Parameters:
        -----------
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
            
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered
            as a core point. This includes the point itself.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
    def fit_predict(self, X):
        """
        Perform DBSCAN clustering on X.
        
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
        return f"DBSCAN"
    
    def get_parameters(self):
        return f"eps={self.eps}, min_samples={self.min_samples}"