from sklearn.cluster import OPTICS
from . import ExploratoryClusteringBase

class OPTICSExploratory(ExploratoryClusteringBase):
    """
    OPTICS for exploratory clustering
    """
    def __init__(self, min_samples=5):
        """
        Initialize OPTICS exploratory clustering
        
        Parameters:
        -----------
            
        min_samples : int, default=None
            The number of samples in a neighborhood for a point to be considered
            as a core point.
        """
        self.min_samples = min_samples
        self.model = OPTICS(
            min_samples=min_samples
        )
        
    def fit_predict(self, X):
        """
        Perform OPTICS clustering on X.
        
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
        return f"OPTICS"
    
    def get_parameters(self):
        return f"min_samples={self.min_samples}"
