from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


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

def split_points_by_labels(x, y, labels):
    """
    Splits points based on their labels.
    
    Parameters:
    - x: list or numpy array of x coordinates
    - y: list or numpy array of y coordinates
    - labels: list or numpy array of corresponding labels
    
    Returns:
    - A dictionary where keys are unique labels and values are numpy arrays of shape (N, 2) with x and y coordinates.
    """
    label_dict = defaultdict(list)
    
    for xi, yi, label in zip(x, y, labels):
        label_dict[label].append([xi, yi])
    
    # Convert lists to numpy arrays for easier processing
    return {label: np.array(points) for label, points in label_dict.items()}

def compare_blob(blob_df, blobs_dict):
    """
    Compares a given blob (Pandas DataFrame) with all blobs returned from split_points_by_labels.
    
    Parameters:
    - blob_df: Pandas DataFrame with columns ['x', 'y'] representing the x, y coordinates of a blob.
    - blobs_dict: dictionary of labeled blobs returned by split_points_by_labels.
    
    Returns:
    Returns:
    - A dictionary where keys are labels and values are booleans indicating if exact matches for energy and lz values were found.
    """
    blob = blob_df[['Lz', 'Energy']].to_numpy()
    results = {}
    
    for label, points in blobs_dict.items():
        if points.size == 0:
            results[label] = False
        else:
            match = any(np.all(points == row, axis=1).any() for row in blob)
            results[label] = match
    
    return results

def plot_blobs(blobs_dict, blob_df, filename, axis_names, hiding_cluster, save_path="."):
    """
    Plots data from different blobs with different colors and a legend.
    
    Parameters:
    - blobs_dict: dictionary of labeled blobs returned by split_points_by_labels.
    - blob_df: Pandas DataFrame with 2 columns representing the x, y coordinates of a blob.
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(blobs_dict))
    
    # Plot imported blob in red
    plt.scatter(blob_df[axis_names[0]], blob_df[axis_names[1]], 
                color='red', label='Imported Blob', marker='x', s=100)

    for i, (label, points) in enumerate(blobs_dict.items()):
        if label == hiding_cluster: # not assigned
            continue
        plt.scatter(points[:, 0], points[:, 1], 
                    label=f'Label {label}', color=colors(i), alpha=0.6)
    
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.title('Blobs Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/compare_{filename}_plot.png")