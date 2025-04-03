from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np

def hungarian_match(known_centroids, centroids, known_labels, filtered_labels):
    # Hungarian Algorithm (Munkres): match computed centroids to known centroids
    distance_matrix = cdist(known_centroids, centroids)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create a mapping {computed cluster index -> known label}
    cluster_mapping = {col: known_labels[row] for row, col in zip(row_ind, col_ind)}

    # Apply the mapping to assign correct labels
    mapped_labels = np.array([
        cluster_mapping[label] if label in cluster_mapping else 0 
        for label in filtered_labels
    ])
    
    return cluster_mapping, mapped_labels
