import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Import the CLIMB algorithm and exploratory strategies
from CLiMB.core.CLiMB import CLiMB
from CLiMB.exploratory import DBSCANExploratory, HDBSCANExploratory, OPTICSExploratory

def simple_blobs_example():
    """
    Simple example using synthetic data with clear clusters
    """
    print("Running simple blobs example...")
    
    # Generate synthetic data with clear clusters
    X, y = make_blobs(
        n_samples=500, 
        centers=4,
        n_features=3,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create seed points from the first few points of each cluster
    unique_labels = np.unique(y)
    seed_points = np.array([
        X[y == label][:1].mean(axis=0) for label in unique_labels
    ])
    seed_points_scaled = scaler.transform(seed_points)
    
    # Initialize CLIMB with default parameters
    climb = CLiMB(
        constrained_clusters=len(unique_labels),
        seed_points=seed_points_scaled,
        density_threshold=0.2,
        distance_threshold=2.0
    )
    
    # Fit the model
    climb.fit(X_scaled)
    
    # Get the cluster labels
    labels = climb.get_labels()
    
    # Transform back to original scale for visualization
    climb.inverse_transform(scaler)
    
    # Visualize results
    fig = climb.plot_comprehensive_3d(
        axis_labels=['Feature 1', 'Feature 2', 'Feature 3']
    )
    plt.savefig('blobs_example_3d.png')
    plt.close(fig)
    
    fig = climb.plot_comprehensive_2d(
        dimensions=(0, 1),
        axis_labels=['Feature 1', 'Feature 2']
    )
    plt.savefig('blobs_example_2d.png')
    plt.close(fig)
    
    # Print statistics
    constrained_points = len(X_scaled) - len(climb.unassigned_points)
    print(f"Points assigned in constrained phase: {constrained_points} ({constrained_points/len(X_scaled):.1%})")
    print(f"Points assigned in exploratory phase: {len(climb.unassigned_points)} ({len(climb.unassigned_points)/len(X_scaled):.1%})")
    print(f"Total clusters found: {len(np.unique(labels))} (expected: {len(unique_labels)})")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    simple_blobs_example()