import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Import the CLiMB algorithm and exploratory strategies
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory.HDBSCANExploratory import HDBSCANExploratory 
from CLiMB.exploratory.DBSCANExploratory import DBSCANExploratory
from CLiMB.exploratory.OPTICSExploratory import OPTICSExploratory

def complex_mixed_example():
    """
    Complex example using mixed data with both convex and non-convex clusters
    """
    print("Running complex mixed data example...")
    
    # Generate blobs for known clusters
    X_blobs, y_blobs = make_blobs(
        n_samples=300, 
        centers=3,
        n_features=3,
        random_state=42
    )
    
    # Generate moons for exploratory clustering
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
    # Add a third dimension to moons
    X_moons = np.column_stack((
        X_moons, 
        np.random.normal(0, 0.1, size=X_moons.shape[0])
    ))
    
    # Combine datasets
    X_combined = np.vstack([X_blobs, X_moons])
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Create seed points from known blob clusters
    unique_labels = np.unique(y_blobs)
    seed_points = np.array([
        X_blobs[y_blobs == label].mean(axis=0) for label in unique_labels
    ])
    seed_points_scaled = scaler.transform(seed_points)
    
    # Initialize CLiMB with HDBSCAN exploratory algorithm
    hdbscan_exploratory = HDBSCANExploratory(min_cluster_size=5, min_samples=3)
    
    climb = CLiMB(
        constrained_clusters=len(unique_labels),
        seed_points=seed_points_scaled,
        density_threshold=0.3,
        distance_threshold=3.0,
        exploratory_algorithm=hdbscan_exploratory
    )
    
    # Fit the model
    climb.fit(X_scaled)
    
    # Get the cluster labels
    labels = climb.get_labels()
    
    # Transform back to original scale
    climb.inverse_transform(scaler)
    
    # Visualize results
    fig = climb.plot_comprehensive_3d(
        axis_labels=['Feature 1', 'Feature 2', 'Feature 3']
    )
    plt.savefig('mixed_example_3d.png')
    plt.close(fig)
    
    fig = climb.plot_comprehensive_2d(
        dimensions=(0, 1),
        axis_labels=['Feature 1', 'Feature 2']
    )
    plt.savefig('mixed_example_2d.png')
    plt.close(fig)
    
    # Print statistics
    constrained_points = len(X_scaled) - len(climb.unassigned_points)
    print(f"Points assigned in constrained phase: {constrained_points} ({constrained_points/len(X_scaled):.1%})")
    print(f"Points assigned in exploratory phase: {len(climb.unassigned_points)} ({len(climb.unassigned_points)/len(X_scaled):.1%})")
    print(f"Total clusters found: {len(np.unique(labels))}")
    print(f"Expected: {len(unique_labels)} known + 2 moons = {len(unique_labels) + 2}")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":   
    complex_mixed_example()