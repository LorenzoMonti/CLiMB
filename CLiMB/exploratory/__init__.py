from abc import ABC, abstractmethod

class ExploratoryClusteringBase(ABC):
    """
    Base abstract class for exploratory clustering algorithms
    """
    
    @abstractmethod
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to cluster.
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point.
        """
        pass
        
    @abstractmethod
    def get_name(self):
        """
        Returns the name of the clustering algorithm
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Returns the parameters of the clustering algorithm
        """
        pass